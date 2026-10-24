// omdet_rt: OmDet-Turbo (omlab/omdet-turbo-swin-tiny-hf) on the Modalix MLA.
//
// Runs the programs of model/omdet_export.py with SimaLMM's MLAModelWithBuffer:
//   bb*        12 Swin-T blocks on tokens in window layout; between them the CPU gathers tokens into
//              the next block's layout (and 2x2 neighbourhoods for patch merging), and copies the
//              projected level features out in row-major order (manifest "backbone" steps).
//   aifi       the hybrid encoder's attention layer on the stride-32 level
//   ccfm       FPN + PAN + the decoder's channel projections; its three outputs land in one buffer,
//              one level after the other (the decoder's token order)
//   decin      the decoder layers' value projections, the encoder head's class features and box MLP
//   dec_pre_k  query position, self-attention over [queries | task tokens], sampling offsets/weights
//   dec_post_k deformable output projection, FFN, box head
//   text/task  CLIP text encoder (8 sequences per run) and the task head
// CPU: deformable sampling (msda.hpp, 3 levels), cosine class similarity, the top-900 selection,
// box updates in logit space, final class logits.
//
// Protocol (stdin/stdout, little endian), one request at a time:
//   request : "OMDT" u32 cmd u32 nbytes payload
//   response: "OMDT" u32 status(0 ok) u32 nbytes payload
//   cmd 1 INFO     -> JSON
//   cmd 2 TEXT     u32 n_seq, u32 n_task, u32 eot[n_seq], f32 emb[n_seq][T][512]
//                  (sequences 0..n_seq-2 are the classes, the last one is the task prompt) -> f32 ms
//   cmd 3 DETECT   u8 rgb[S][S][3] (squashed to S x S)
//   cmd 6 DETECT_CACHED  same image, new prompt
//                  -> u32 C, f32 logits[900][C], f32 boxes[900][4] (cx, cy, w, h), f32 timings[16]
//   cmd 4 QUIT
#include <sima_lmm/mla_buffer.hpp>
#include <sima_lmm/mla_model.hpp>

#include <nlohmann/json.hpp>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "dino/bf16.hpp"
#include "dino/msda.hpp"

namespace fs = std::filesystem;
namespace llima = simaai::llima;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using dino::from_bf16;
using dino::to_bf16;

namespace omdet {

constexpr int D = 256, NQ = 900, CT = 512;   // model width, queries, CLIP width

static double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

static std::vector<char> read_file(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + p.string());
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct Buffer {
  Buffer(const std::string& name, size_t elems) : n(elems), b(name, {1, elems}, "bfloat16", false) {
    b.allocate();
    b.clear();
  }
  ~Buffer() { b.free(); }
  uint16_t* p() { return static_cast<uint16_t*>(b.get_virtual_addr()); }
  void flush() { b.flush_cache(0, n * 2); }
  void invalidate() { b.invalidate_cache(0, n * 2); }
  llima::MLABufferSlice all() { return llima::MLABufferSlice(&b); }
  llima::MLABufferSlice slice(size_t off, size_t len) {
    return llima::MLABufferSlice(&b, {0u, (uint32_t)off}, {1u, (uint32_t)len});
  }
  size_t n;
  llima::MLABuffer b;
};

struct Stage {
  std::unique_ptr<llima::MLAModelWithBuffer> model;
  double last_ms = 0;
  void run() {
    auto t = Clock::now();
    model->run();
    last_ms = ms_since(t);
  }
};

struct Step {
  std::string op, name;
  int k = 1, c = 0, src_stride = 0, src_off = 0, level = 0, grid = 0;
  std::vector<int32_t> table;
  Buffer* src = nullptr;   // gather / level: the previous block's output
  Buffer* dst = nullptr;   // patchify / gather: the next block's input
  size_t in_elems = 0, out_elems = 0;
};

static float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
static float inv_sigmoid(float x) {
  x = std::clamp(x, 0.f, 1.f);
  return std::log(std::max(x, 1e-5f) / std::max(1.f - x, 1e-5f));
}

class Model {
 public:
  explicit Model(const fs::path& dir) {
    const json m = json::parse(read_file(dir / "manifest.json"));
    S_ = m.at("size");
    TT_ = m.at("task_tokens");
    TXT_T_ = m.at("text_tokens");
    TXT_B_ = m.at("text_batch");
    N_ = 0;
    for (auto& l : m.at("levels")) {
      dino::Level lv;
      lv.h = l[0];
      lv.w = l[1];
      lv.start = l[2];
      levels_.push_back(lv);
      N_ += (size_t)lv.h * lv.w;
    }
    auto raw = read_file(dir / "host.f32");
    host_.resize(raw.size() / 4);
    std::memcpy(host_.data(), raw.data(), raw.size());
    for (auto& [k, v] : m.at("host").items()) index_[k] = v.at("offset").get<size_t>();
    const json order = m.value("inputs", json::object()), outs = m.value("outputs", json::object());
    const auto elf = [&](const std::string& n) { return (dir / "elf" / (n + ".elf")).string(); };
    auto by_name = [&](const std::string& prog, std::map<std::string, llima::MLABufferSlice> named,
                       const json& o) {
      std::vector<llima::MLABufferSlice> v;
      for (const auto& nm : o.at(prog)) v.push_back(named.at(nm.get<std::string>()));
      return v;
    };
    const auto t0 = Clock::now();
    int loaded = 0;
    auto load = [&](const std::string& n, std::vector<llima::MLABufferSlice> in,
                    std::vector<llima::MLABufferSlice> out) {
      auto s = std::make_unique<Stage>();
      s->model = std::make_unique<llima::MLAModelWithBuffer>(elf(n), in, out);
      std::fprintf(stderr, "OMDTLOAD %d %s\n", loaded++, n.c_str());
      s->model->load();
      stages_[n] = std::move(s);
    };

    // ---- backbone: per block its own input/output buffers
    for (auto& js : m.at("backbone")) {
      Step st;
      st.op = js.at("op");
      st.name = js.value("name", "");
      st.k = js.value("k", 1);
      st.c = js.value("c", 0);
      st.src_stride = js.value("src_stride", 0);
      st.src_off = js.value("src_off", 0);
      st.level = js.value("level", 0);
      st.grid = js.value("grid", 0);
      st.in_elems = js.value("in_elems", 0);
      st.out_elems = js.value("out_elems", 0);
      if (js.contains("table")) {
        auto t = read_file(dir / js.at("table").get<std::string>());
        st.table.resize(t.size() / 4);
        std::memcpy(st.table.data(), t.data(), t.size());
      }
      steps_.push_back(std::move(st));
    }
    for (size_t i = 0; i < 3; ++i)
      lvl_.push_back(std::make_unique<Buffer>("level", (size_t)levels_[i].h * levels_[i].w * D));
    for (size_t i = 0; i < steps_.size(); ++i) {
      Step& st = steps_[i];
      if (st.op != "run") continue;
      bb_in_.push_back(std::make_unique<Buffer>("bb_in", st.in_elems));
      bb_out_.push_back(std::make_unique<Buffer>("bb_out", st.out_elems));
      load(st.name, {bb_in_.back()->all()}, {bb_out_.back()->all()});
      st.src = bb_in_.back().get();
      st.dst = bb_out_.back().get();
    }
    // wire the CPU steps to their neighbouring blocks
    for (size_t i = 0; i < steps_.size(); ++i) {
      Step& st = steps_[i];
      if (st.op == "run") continue;
      for (size_t j = i + 1; j < steps_.size() && !st.dst; ++j)
        if (steps_[j].op == "run") st.dst = steps_[j].src;
      for (size_t j = i; j-- > 0 && !st.src;)
        if (steps_[j].op == "run") st.src = steps_[j].dst;
    }
    // ---- encoder
    aifi_ = std::make_unique<Buffer>("aifi", (size_t)levels_[2].h * levels_[2].w * D);
    load("aifi", {lvl_[2]->all()}, {aifi_->all()});
    dec_ = std::make_unique<Buffer>("dec_levels", N_ * D);
    std::map<std::string, llima::MLABufferSlice> lv_out;
    for (int i = 0; i < 3; ++i)
      lv_out.emplace("o" + std::to_string(i), dec_->slice(levels_[i].start * D, (size_t)levels_[i].h * levels_[i].w * D));
    load("ccfm", by_name("ccfm", {{"l0", lvl_[0]->all()}, {"l1", lvl_[1]->all()}, {"l2", aifi_->all()}}, order),
         by_name("ccfm", lv_out, outs));
    di_ = std::make_unique<Buffer>("decin", 8 * N_ * D);
    std::map<std::string, llima::MLABufferSlice> lv_in;
    for (int i = 0; i < 3; ++i)
      lv_in.emplace("l" + std::to_string(i), dec_->slice(levels_[i].start * D, (size_t)levels_[i].h * levels_[i].w * D));
    load("decin", by_name("decin", lv_in, order), {di_->all()});
    // ---- decoder
    pre_in_ = std::make_unique<Buffer>("pre_in", (size_t)(2 * NQ + TT_ + 1) * D);
    pre_out_ = std::make_unique<Buffer>("pre_out", (size_t)(3 * NQ + TT_) * D);
    post_in_ = std::make_unique<Buffer>("post_in", (size_t)2 * NQ * D);
    post_out_ = std::make_unique<Buffer>("post_out", (size_t)2 * NQ * D);
    for (int k = 0; k < 6; ++k) {
      load("dec_pre" + std::to_string(k), {pre_in_->all()}, {pre_out_->all()});
      load("dec_post" + std::to_string(k), {post_in_->all()}, {post_out_->all()});
    }
    // ---- text
    txt_in_ = std::make_unique<Buffer>("text_in", (size_t)TXT_B_ * TXT_T_ * CT);
    txt_out_ = std::make_unique<Buffer>("text_out", (size_t)TXT_B_ * TXT_T_ * CT);
    load("text", {txt_in_->all()}, {txt_out_->all()});
    task_in_ = std::make_unique<Buffer>("task_in", (size_t)TT_ * CT);
    task_out_ = std::make_unique<Buffer>("task_out", (size_t)TT_ * D);
    load("task", {task_in_->all()}, {task_out_->all()});
    load_s_ = ms_since(t0) / 1000.0;
    std::fprintf(stderr, "omdet_rt: S=%d, %zu programs loaded in %.1f s\n", S_, stages_.size(), load_s_);
  }

  ~Model() {
    for (auto& [n, s] : stages_) s->model->free();
  }

  json info() const {
    return {{"size", S_}, {"queries", NQ}, {"task_tokens", TT_}, {"text_tokens", TXT_T_}, {"text_batch", TXT_B_},
            {"programs", stages_.size()}, {"load_seconds", load_s_}};
  }

  float set_text(const char* p, size_t n) {
    const auto t0 = Clock::now();
    uint32_t n_seq, n_task;
    std::memcpy(&n_seq, p, 4);
    std::memcpy(&n_task, p + 4, 4);
    if (n_seq < 2) throw std::runtime_error("TEXT needs at least one class and the task");
    std::vector<uint32_t> eot(n_seq);
    std::memcpy(eot.data(), p + 8, 4 * n_seq);
    const float* emb = reinterpret_cast<const float*>(p + 8 + 4 * n_seq);
    if (n != 8 + 4 * n_seq + (size_t)n_seq * TXT_T_ * CT * 4) throw std::runtime_error("TEXT: bad payload size");
    // CLIP on all sequences, TXT_B_ per run
    std::vector<float> hid((size_t)n_seq * TXT_T_ * CT);
    for (uint32_t s0 = 0; s0 < n_seq; s0 += TXT_B_) {
      const uint32_t cnt = std::min<uint32_t>(TXT_B_, n_seq - s0);
      std::memset(txt_in_->p(), 0, txt_in_->n * 2);
      dino::to_bf16_span(emb + (size_t)s0 * TXT_T_ * CT, txt_in_->p(), (size_t)cnt * TXT_T_ * CT);
      txt_in_->flush();
      stages_.at("text")->run();
      txt_out_->invalidate();
      dino::from_bf16_span(txt_out_->p(), hid.data() + (size_t)s0 * TXT_T_ * CT, (size_t)cnt * TXT_T_ * CT);
    }
    // classes: end token @ text_projection, then the encoder / last decoder class heads, normalised
    C_ = (int)n_seq - 1;
    enc_cls_.assign((size_t)C_ * D, 0.f);
    dec_cls_.assign((size_t)C_ * D, 0.f);
    const float* P = w("text_proj");
    std::vector<float> ce(CT);
    for (int c = 0; c < C_; ++c) {
      const float* h = hid.data() + ((size_t)c * TXT_T_ + std::min<uint32_t>(eot[c], TXT_T_ - 1)) * CT;
      for (int j = 0; j < CT; ++j) {
        float s = 0;
        for (int i = 0; i < CT; ++i) s += h[i] * P[(size_t)i * CT + j];
        ce[j] = s;
      }
      head(ce.data(), w("enc_cls.w"), w("enc_cls.b"), enc_cls_.data() + (size_t)c * D);
      head(ce.data(), w("dec_cls.w"), w("dec_cls.b"), dec_cls_.data() + (size_t)c * D);
    }
    // task tokens -> task head
    n_task_ = std::min<int>(n_task, TT_);
    std::vector<float> ti((size_t)TT_ * CT, 0.f);
    std::memcpy(ti.data(), hid.data() + (size_t)C_ * TXT_T_ * CT, (size_t)n_task_ * CT * 4);
    dino::to_bf16_span(ti.data(), task_in_->p(), ti.size());
    task_in_->flush();
    stages_.at("task")->run();
    task_out_->invalidate();
    task_.assign(task_out_->p(), task_out_->p() + (size_t)TT_ * D);
    have_text_ = true;
    return (float)ms_since(t0);
  }

  std::vector<char> detect(const uint8_t* rgb) {
    if (!have_text_) throw std::runtime_error("DETECT before TEXT");
    std::vector<float> t(16, 0.f);
    const auto t0 = Clock::now();
    run_backbone(rgb, t.data());
    auto te = Clock::now();
    stages_.at("aifi")->run();
    stages_.at("ccfm")->run();
    stages_.at("decin")->run();
    t[3] = (float)ms_since(te);
    has_image_ = true;
    return decode(t, t0);
  }

  std::vector<char> detect_cached() {
    if (!has_image_) throw std::runtime_error("no image yet");
    std::vector<float> t(16, 0.f);
    return decode(t, Clock::now());
  }

 private:
  const float* w(const std::string& k) const { return host_.data() + index_.at(k); }

  // y = normalise(W x + b), W (256, 512) row-major
  static void head(const float* x, const float* W, const float* b, float* y) {
    double nn = 0;
    for (int o = 0; o < D; ++o) {
      float s = b[o];
      const float* wr = W + (size_t)o * CT;
      for (int i = 0; i < CT; ++i) s += wr[i] * x[i];
      y[o] = s;
      nn += (double)s * s;
    }
    const float inv = 1.f / std::max(std::sqrt((float)nn), 1e-12f);
    for (int o = 0; o < D; ++o) y[o] *= inv;
  }

  void run_backbone(const uint8_t* rgb, float* timing) {
    double mla = 0, cpu = 0;
    for (auto& st : steps_) {
      const auto t = Clock::now();
      if (st.op == "run") {
        Stage& s = *stages_.at(st.name);
        s.run();
        mla += s.last_ms;
        continue;
      }
      if (st.op == "patchify") {
        uint16_t* dst = st.dst->p();
        const int g = st.grid;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < st.table.size(); ++i) {
          float px[48];
          const int32_t o = st.table[i];
          if (o < 0) {
            std::fill(px, px + 48, 0.f);
          } else {
            const int y4 = o / g, x4 = o % g;
            for (int py = 0; py < 4; ++py)
              for (int pxx = 0; pxx < 4; ++pxx) {
                const uint8_t* p = rgb + ((size_t)(y4 * 4 + py) * S_ + x4 * 4 + pxx) * 3;
                for (int c = 0; c < 3; ++c) px[c * 16 + py * 4 + pxx] = p[c] * (1.0f / 255.0f);
              }
          }
          dino::to_bf16_span(px, dst + i * 48, 48);
        }
        st.dst->flush();
        timing[0] = (float)ms_since(t);
      } else if (st.op == "gather") {
        st.src->invalidate();
        const uint16_t* in = st.src->p();
        uint16_t* out = st.dst->p();
        const size_t c = st.c, rows = st.table.size() / st.k;
#pragma omp parallel for schedule(static)
        for (size_t r = 0; r < rows; ++r)
          for (int j = 0; j < st.k; ++j) {
            const int32_t idx = st.table[r * st.k + j];
            uint16_t* d = out + (r * st.k + j) * c;
            if (idx < 0)
              std::memset(d, 0, c * 2);
            else
              std::memcpy(d, in + (size_t)idx * st.src_stride + st.src_off, c * 2);
          }
        st.dst->flush();
      } else if (st.op == "level") {
        st.src->invalidate();
        const uint16_t* in = st.src->p();
        Buffer& L = *lvl_[st.level];
        uint16_t* out = L.p();
#pragma omp parallel for schedule(static)
        for (size_t r = 0; r < st.table.size(); ++r)
          std::memcpy(out + r * D, in + (size_t)st.table[r] * st.src_stride + st.src_off, D * 2);
        L.flush();
      }
      cpu += ms_since(t);
    }
    timing[1] = (float)mla;
    timing[2] = (float)cpu;
  }

  std::vector<char> decode(std::vector<float>& t, Clock::time_point t0) {
    auto ts = Clock::now();
    const size_t N = N_;
    di_->invalidate();
    // encoder head of every token: class features and box (sequential copies out of device memory)
    cf_.resize(N * D);
    std::memcpy(cf_.data(), di_->p() + 6 * N * D, N * D * 2);
    box_.resize(N * D);
    std::memcpy(box_.data(), di_->p() + 7 * N * D, N * D * 2);
    const float* anchor = w("anchors");
    const float* valid = w("valid");
    const float* cfi = w("cf_invalid");
    std::vector<float> cfi_n(D);
    float inv_sim = -1e30f;
    {
      double nn = 0;
      for (int i = 0; i < D; ++i) nn += (double)cfi[i] * cfi[i];
      const float inv = 1.f / std::max(std::sqrt((float)nn), 1e-12f);
      for (int i = 0; i < D; ++i) cfi_n[i] = cfi[i] * inv;
      for (int c = 0; c < C_; ++c) {
        float s = 0;
        for (int i = 0; i < D; ++i) s += cfi_n[i] * enc_cls_[(size_t)c * D + i];
        inv_sim = std::max(inv_sim, s);
      }
    }
    std::vector<float> sim(N);
#pragma omp parallel for schedule(static)
    for (size_t k = 0; k < N; ++k) {
      if (valid[k] < 0.5f) {
        sim[k] = inv_sim;
        continue;
      }
      float x[D];
      dino::from_bf16_span(cf_.data() + k * D, x, D);
      float nn = 0;
      for (int i = 0; i < D; ++i) nn += x[i] * x[i];
      const float inv = 1.f / std::max(std::sqrt(nn), 1e-12f);
      float best = -1e30f;
      for (int c = 0; c < C_; ++c) {
        const float* e = enc_cls_.data() + (size_t)c * D;
        float s = 0;
#pragma omp simd reduction(+ : s)
        for (int i = 0; i < D; ++i) s += x[i] * e[i];
        best = std::max(best, s * inv);
      }
      sim[k] = best;
    }
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + NQ, idx.end(),
                      [&](int a, int b) { return sim[a] > sim[b] || (sim[a] == sim[b] && a < b); });
    // queries, reference boxes, task tokens
    ref_.assign((size_t)NQ * 4, 1.f);
    uint16_t* pin = pre_in_->p();
    std::vector<uint16_t> cfi_bf(D);
    for (int i = 0; i < D; ++i) cfi_bf[i] = to_bf16(cfi[i]);
    for (int q = 0; q < NQ; ++q) {
      const int k = idx[q];
      const bool ok = valid[k] >= 0.5f;
      std::memcpy(pin + (size_t)q * D, ok ? cf_.data() + (size_t)k * D : cfi_bf.data(), D * 2);
      if (ok)
        for (int j = 0; j < 4; ++j) ref_[q * 4 + j] = sigmoid(from_bf16(box_[(size_t)k * D + j]) + anchor[k * 4 + j]);
    }
    write_refs();
    std::memcpy(pin + (size_t)2 * NQ * D, task_.data(), (size_t)TT_ * D * 2);
    uint16_t* km = pin + (size_t)(2 * NQ + TT_) * D;
    for (int i = 0; i < TT_; ++i) km[i] = to_bf16(i < n_task_ ? 0.f : -1e4f);
    pre_in_->flush();
    t[4] = (float)ms_since(ts);
    // decoder
    double mla = 0, cpu = 0;
    dino::MsdaArgs a;
    a.ref = ref_.data();
    a.ref_dims = 4;
    a.queries = NQ;
    a.n_levels = 3;
    for (int l = 0; l < 3; ++l) a.levels[l] = levels_[l];
    for (int k = 0; k < 6; ++k) {
      Stage& pre = *stages_.at("dec_pre" + std::to_string(k));
      pre.run();
      mla += pre.last_ms;
      auto tc = Clock::now();
      pre_out_->invalidate();
      const uint16_t* po = pre_out_->p();
      a.value = di_->p() + (size_t)k * N * D;
      a.offsets = po + (size_t)(NQ + TT_) * D;
      a.logits = po + (size_t)(2 * NQ + TT_) * D;
      a.out = post_in_->p() + (size_t)NQ * D;
      dino::msda(a);
      std::memcpy(post_in_->p(), po, (size_t)NQ * D * 2);                       // H2
      std::memcpy(pin + (size_t)2 * NQ * D, po + (size_t)NQ * D, (size_t)TT_ * D * 2);   // next task tokens
      post_in_->flush();
      cpu += ms_since(tc);
      Stage& post = *stages_.at("dec_post" + std::to_string(k));
      post.run();
      mla += post.last_ms;
      tc = Clock::now();
      post_out_->invalidate();
      const uint16_t* qo = post_out_->p();
      for (int q = 0; q < NQ; ++q)
        for (int j = 0; j < 4; ++j)
          ref_[q * 4 + j] = sigmoid(from_bf16(qo[((size_t)NQ + q) * D + j]) + inv_sigmoid(ref_[q * 4 + j]));
      if (k < 5) {
        std::memcpy(pin, qo, (size_t)NQ * D * 2);
        write_refs();
        pre_in_->flush();
      }
      cpu += ms_since(tc);
    }
    t[5] = (float)mla;
    // class logits of the last layer's queries: cosine with the class heads / 0.07
    auto tl = Clock::now();
    const uint16_t* hq = post_out_->p();
    std::vector<float> logits((size_t)NQ * C_);
#pragma omp parallel for schedule(static)
    for (int q = 0; q < NQ; ++q) {
      float x[D];
      dino::from_bf16_span(hq + (size_t)q * D, x, D);
      float nn = 0;
      for (int i = 0; i < D; ++i) nn += x[i] * x[i];
      const float inv = 1.f / std::max(std::sqrt(nn), 1e-12f);
      for (int c = 0; c < C_; ++c) {
        const float* e = dec_cls_.data() + (size_t)c * D;
        float s = 0;
        for (int i = 0; i < D; ++i) s += x[i] * e[i];
        logits[(size_t)q * C_ + c] = s * inv / 0.07f;
      }
    }
    cpu += ms_since(tl);
    t[6] = (float)cpu;
    t[7] = (float)ms_since(t0);
    std::vector<char> out(4 + (size_t)NQ * C_ * 4 + (size_t)NQ * 16 + 16 * 4);
    uint32_t c = C_;
    std::memcpy(out.data(), &c, 4);
    std::memcpy(out.data() + 4, logits.data(), logits.size() * 4);
    std::memcpy(out.data() + 4 + logits.size() * 4, ref_.data(), ref_.size() * 4);
    std::memcpy(out.data() + out.size() - 64, t.data(), 64);
    return out;
  }

  void write_refs() {
    uint16_t* r = pre_in_->p() + (size_t)NQ * D;
    for (int q = 0; q < NQ; ++q)
      for (int j = 0; j < 4; ++j) r[(size_t)q * D + j] = to_bf16(ref_[q * 4 + j]);
  }

  int S_ = 0, TT_ = 64, TXT_T_ = 64, TXT_B_ = 8, C_ = 0, n_task_ = 0;
  size_t N_ = 0;
  double load_s_ = 0;
  bool have_text_ = false, has_image_ = false;
  std::vector<dino::Level> levels_;
  std::vector<float> host_, enc_cls_, dec_cls_, ref_;
  std::vector<uint16_t> task_, cf_, box_;
  std::map<std::string, size_t> index_;
  std::vector<Step> steps_;
  std::vector<std::unique_ptr<Buffer>> bb_in_, bb_out_, lvl_;
  std::unique_ptr<Buffer> aifi_, dec_, di_, pre_in_, pre_out_, post_in_, post_out_, txt_in_, txt_out_, task_in_,
      task_out_;
  std::map<std::string, std::unique_ptr<Stage>> stages_;
};

static bool read_all(int fd, void* p, size_t n) {
  auto* c = static_cast<char*>(p);
  while (n) {
    const ssize_t r = ::read(fd, c, n);
    if (r <= 0) return false;
    c += r;
    n -= r;
  }
  return true;
}

static void write_all(int fd, const void* p, size_t n) {
  auto* c = static_cast<const char*>(p);
  while (n) {
    const ssize_t r = ::write(fd, c, n);
    if (r <= 0) throw std::runtime_error("write to client failed");
    c += r;
    n -= r;
  }
}

static volatile std::sig_atomic_t g_stop = 0;

static void reply(int fd, uint32_t status, const std::vector<char>& payload) {
  const uint32_t n = payload.size();
  write_all(fd, "OMDT", 4);
  write_all(fd, &status, 4);
  write_all(fd, &n, 4);
  if (n) write_all(fd, payload.data(), n);
}

}  // namespace omdet

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: omdet_rt <bundle dir>\n");
    return 2;
  }
  std::signal(SIGTERM, [](int) { omdet::g_stop = 1; });
  std::signal(SIGINT, [](int) { omdet::g_stop = 1; });
  std::signal(SIGPIPE, SIG_IGN);
  const int proto = ::dup(1);
  ::dup2(2, 1);
  int rc = 0;
  llima::connect_mla_rt({});
  try {
    omdet::Model model(argv[1]);
    std::fwrite("OMDTREADY\n", 1, 10, stderr);
    std::fflush(stderr);
    while (!omdet::g_stop) {
      char magic[4];
      uint32_t cmd = 0, n = 0;
      if (!omdet::read_all(0, magic, 4) || !omdet::read_all(0, &cmd, 4) || !omdet::read_all(0, &n, 4)) break;
      if (std::memcmp(magic, "OMDT", 4) != 0) throw std::runtime_error("bad request header");
      std::vector<char> in(n);
      if (n && !omdet::read_all(0, in.data(), n)) break;
      try {
        std::vector<char> out;
        if (cmd == 1) {
          const std::string s = model.info().dump();
          out.assign(s.begin(), s.end());
        } else if (cmd == 2) {
          const float ms = model.set_text(in.data(), n);
          out.resize(4);
          std::memcpy(out.data(), &ms, 4);
        } else if (cmd == 3) {
          out = model.detect(reinterpret_cast<const uint8_t*>(in.data()));
        } else if (cmd == 6) {
          out = model.detect_cached();
        } else if (cmd == 4) {
          omdet::reply(proto, 0, {});
          break;
        } else {
          throw std::runtime_error("unknown command " + std::to_string(cmd));
        }
        omdet::reply(proto, 0, out);
      } catch (const std::exception& e) {
        const std::string s = e.what();
        omdet::reply(proto, 1, std::vector<char>(s.begin(), s.end()));
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "omdet_rt: fatal: %s\n", e.what());
    rc = 1;
  }
  llima::disconnect_mla_rt();
  return rc;
}
