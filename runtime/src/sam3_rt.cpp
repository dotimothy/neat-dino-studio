// sam3_rt: SAM 3 (facebook/sam3) image path on the Modalix MLA.
//
// Runs the compiled programs of model/sam3_export.py with SimaLMM's MLAModelWithBuffer:
// 8 ViT programs (4 groups of 7 windowed layers, 4 global layers), 3 neck levels, the CLIP
// text encoder, the 6 DETR encoder layers, 6 decoder layers, and the mask decoder (prompt
// cross-attention, pixel decoder, query . pixel). The CPU does the rest: patchify into window
// layout, window -> image order for the neck, and per decoder layer the query position (sine
// of the boxes + MLP), the box relative-position bias, the box update; then presence,
// text-query scoring and the mask embeddings.
//
// ViT tokens stay in window layout (window-major, row-major inside the 24x24 window); a global
// layer reads the same buffer as (1, C, 1, all tokens), which is the same bytes in HWC.
//
// Protocol (stdin/stdout, little endian), one request at a time:
//   request : "SAM3" u32 cmd u32 nbytes payload
//   response: "SAM3" u32 status(0 ok) u32 nbytes payload
//   cmd 1 INFO     -> JSON
//   cmd 2 TEXT     payload f32 packed[T][1024 + T] (token-major: embeddings, then additive mask
//                  row), u32 n_valid -> f32 timing_ms
//   (the mask dot product query . pixel runs on the CPU for kept queries)
//   cmd 3 DETECT   payload f32 mask_min_score, u8 rgb[S][S][3] (already resized to S x S)
//   cmd 6 DETECT_CACHED  payload f32 mask_min_score (same image, new prompt)
//                  -> f32 logits[Q], f32 presence_logit, f32 boxes[Q][4] (cx,cy,w,h),
//                     u32 n_masks, n_masks x (u32 query, u8 mask[M][M] (logit > 0)), f32 timings[16]
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
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "dino/bf16.hpp"

namespace fs = std::filesystem;
namespace llima = simaai::llima;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using dino::from_bf16;
using dino::to_bf16;

namespace sam3 {

constexpr int D = 256, kBiasIn = 16;

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
  size_t n;
  llima::MLABuffer b;
};

struct Stage {
  std::string name;
  std::unique_ptr<llima::MLAModelWithBuffer> model;
  double last_ms = 0;
  void run() {
    auto t = Clock::now();
    model->run();
    last_ms = ms_since(t);
  }
};

// y[r][o] = b[o] + sum_i W[o][i] x[r][i]   (W row-major (out, in)), optional ReLU
static void linear(const float* x, int rows, int in, const float* W, const float* b, int out, float* y,
                   bool relu = false) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + (size_t)r * in;
    float* yr = y + (size_t)r * out;
    for (int o = 0; o < out; ++o) {
      const float* w = W + (size_t)o * in;
      float s = 0.f;
#pragma omp simd reduction(+ : s)
      for (int i = 0; i < in; ++i) s += w[i] * xr[i];
      s += b ? b[o] : 0.f;
      yr[o] = relu ? std::max(s, 0.f) : s;
    }
  }
}

static void layer_norm(float* x, int rows, int c, const float* w, const float* b, float eps) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; ++r) {
    float* v = x + (size_t)r * c;
    double mu = 0, var = 0;
    for (int i = 0; i < c; ++i) mu += v[i];
    mu /= c;
    for (int i = 0; i < c; ++i) var += (v[i] - mu) * (v[i] - mu);
    const float inv = 1.f / std::sqrt((float)(var / c) + eps);
    for (int i = 0; i < c; ++i) v[i] = (float)((v[i] - mu) * inv) * w[i] + b[i];
  }
}

// SAM3_DUMP=<dir>: write intermediate buffers (raw bf16, HWC) for comparison with the export's records.
static const char* g_dump = std::getenv("SAM3_DUMP");
static void dump(const std::string& name, Buffer& b) {
  if (!g_dump) return;
  b.invalidate();
  std::ofstream(std::string(g_dump) + "/" + name + ".bf16", std::ios::binary).write((const char*)b.p(), b.n * 2);
}

class Model {
 public:
  explicit Model(const fs::path& dir) : dir_(dir) {
    const json m = json::parse(read_file(dir / "manifest.json"));
    S_ = m.at("size");
    G_ = m.at("grid");
    WS_ = m.at("window");
    Q_ = m.at("queries");
    T_ = m.at("text_tokens");
    H_ = m.at("heads");
    M_ = m.at("mask_size");
    sc_scale_ = m.at("score_scale");
    for (auto& [k, v] : m.at("eps").items()) eps_[k] = v.get<float>();
    const int nW = (G_ / WS_) * (G_ / WS_), W2 = WS_ * WS_, N = G_ * G_;
    // host weights
    auto raw = read_file(dir / "host.f32");
    host_.resize(raw.size() / 4);
    std::memcpy(host_.data(), raw.data(), raw.size());
    for (auto& [k, v] : m.at("host").items()) index_[k] = v.at("offset").get<size_t>();
    auto pos = read_file(dir / "pos.bf16");
    pos_.resize(pos.size() / 2);
    std::memcpy(pos_.data(), pos.data(), pos.size());
    // window layout: token t of window w -> spatial index
    order_.resize(N);
    for (int w = 0, i = 0; w < nW; ++w)
      for (int t = 0; t < W2; ++t, ++i) {
        const int wy = w / (G_ / WS_), wx = w % (G_ / WS_);
        order_[i] = (wy * WS_ + t / WS_) * G_ + wx * WS_ + t % WS_;
      }

    // ---- device buffers (bf16, HWC element order)
    const int P = 14, pc = 3 * P * P;
    patches_ = std::make_unique<Buffer>("patches", (size_t)N * pc);
    va_ = std::make_unique<Buffer>("vit_a", (size_t)N * 1024);
    vb_ = std::make_unique<Buffer>("vit_b", (size_t)N * 1024);
    sp_ = std::make_unique<Buffer>("spatial", (size_t)N * 1024);
    f4_ = std::make_unique<Buffer>("f4", (size_t)16 * N * D);
    f2_ = std::make_unique<Buffer>("f2", (size_t)4 * N * D);
    f1_ = std::make_unique<Buffer>("f1", (size_t)N * D);
    tin_ = std::make_unique<Buffer>("text_in", (size_t)T_ * (1024 + T_));
    text_ = std::make_unique<Buffer>("text", (size_t)T_ * D);
    tmask_ = std::make_unique<Buffer>("tmask", (size_t)T_);
    enc_ = std::make_unique<Buffer>("enc", (size_t)N * D);
    visk_ = std::make_unique<Buffer>("vis_k", (size_t)N * D);
    h0_ = std::make_unique<Buffer>("h0", (size_t)(Q_ + 1) * D);
    for (auto& b : dout_) b = std::make_unique<Buffer>("dec_out", (size_t)(2 * Q_ + 1) * D);   // [h' | box delta]
    qsine_ = std::make_unique<Buffer>("qsine", (size_t)Q_ * 512);
    dy_ = std::make_unique<Buffer>("dy", (size_t)G_ * Q_ * kBiasIn);   // 2 values + zeros (see box_log_deltas)
    dx_ = std::make_unique<Buffer>("dx", (size_t)G_ * Q_ * kBiasIn);
    e2_ = std::make_unique<Buffer>("e2", (size_t)N * D);
    inst_ = std::make_unique<Buffer>("inst", (size_t)16 * N * D);

    // ---- programs (multi-input programs take their inputs in the order the compiler chose,
    // recorded by name in the manifest)
    const json order = m.value("inputs", json::object());
    const auto elf = [&](const std::string& n) { return (dir / "elf" / (n + ".elf")).string(); };
    auto by_name = [&](const std::string& prog, std::map<std::string, llima::MLABufferSlice> named) {
      std::vector<llima::MLABufferSlice> v;
      for (const auto& nm : order.at(prog)) v.push_back(named.at(nm.get<std::string>()));
      return v;
    };
    auto add = [&](const std::string& n, std::vector<llima::MLABufferSlice> in, std::vector<llima::MLABufferSlice> out) {
      auto s = std::make_unique<Stage>();
      s->name = n;
      s->model = std::make_unique<llima::MLAModelWithBuffer>(elf(n), in, out);
      std::fprintf(stderr, "SAM3LOAD %zu %s\n", stages_.size(), n.c_str());
      s->model->load();
      stages_[n] = std::move(s);
    };
    const auto t0 = Clock::now();
    vit_ = m.at("vit").get<std::vector<std::string>>();
    Buffer* src = patches_.get();
    for (size_t i = 0; i < vit_.size(); ++i) {
      Buffer* dst = (i % 2 == 0) ? va_.get() : vb_.get();
      add(vit_[i], {src->all()}, {dst->all()});
      src = dst;
    }
    vit_out_ = src;
    add("neck4", {sp_->all()}, {f4_->all()});
    add("neck2", {sp_->all()}, {f2_->all()});
    add("neck1", {sp_->all()}, {f1_->all()});
    add("text", {tin_->all()}, {text_->all()});
    add("enc", by_name("enc", {{"x", f1_->all()}, {"text", text_->all()}, {"tmask", tmask_->all()}}), {enc_->all()});
    // decoder layers (with their query position, box bias and box head): layer k reads h from the
    // first 1 + Q tokens of layer k-1's output, so nothing is copied between layers
    for (int k = 0; k < 6; ++k) {
      const std::string n = "dec" + std::to_string(k);
      llima::MLABufferSlice hin = k == 0 ? h0_->all()
                                         : llima::MLABufferSlice(&dout_[(k - 1) % 2]->b, {0u, 0u}, {1u, (uint32_t)((Q_ + 1) * D)});
      add(n, by_name(n, {{"h", hin}, {"qsine", qsine_->all()}, {"text", text_->all()}, {"tmask", tmask_->all()},
                         {"enc", enc_->all()}, {"vis_k", visk_->all()}, {"dy", dy_->all()}, {"dx", dx_->all()}}),
          {dout_[k % 2]->all()});
    }
    add("maskpre", by_name("maskpre", {{"enc", enc_->all()}, {"text", text_->all()}, {"tmask", tmask_->all()}}),
        {e2_->all()});
    add("pixdec", by_name("pixdec", {{"e2", e2_->all()}, {"f2", f2_->all()}, {"f4", f4_->all()}}), {inst_->all()});
    load_s_ = ms_since(t0) / 1000.0;
    std::fprintf(stderr, "sam3_rt: S=%d, %zu programs loaded in %.1f s\n", S_, stages_.size(), load_s_);
  }

  ~Model() {
    for (auto& [n, s] : stages_) s->model->free();
  }

  json info() const {
    return {{"size", S_}, {"grid", G_}, {"queries", Q_}, {"text_tokens", T_}, {"mask_size", M_},
            {"programs", stages_.size()}, {"load_seconds", load_s_}};
  }

  float set_text(const float* packed, int n_valid) {
    const auto t0 = Clock::now();
    dino::to_bf16_span(packed, tin_->p(), (size_t)T_ * (1024 + T_));
    tin_->flush();
    stages_.at("text")->run();
    dump("text_in", *tin_);
    dump("text", *text_);
    text_->invalidate();
    n_valid_ = n_valid;
    for (int i = 0; i < T_; ++i) tmask_->p()[i] = to_bf16(i < n_valid ? 0.f : -1e4f);
    tmask_->flush();
    // scoring, text side: t = LN(text + MLP(text)), mean over valid tokens, projected
    std::vector<float> t((size_t)T_ * D), h((size_t)T_ * 2048), o((size_t)T_ * D);
    dino::from_bf16_span(text_->p(), t.data(), t.size());
    linear(t.data(), T_, D, w("sc_mlp.l1.w"), w("sc_mlp.l1.b"), 2048, h.data(), true);
    linear(h.data(), T_, 2048, w("sc_mlp.l2.w"), w("sc_mlp.l2.b"), D, o.data());
    for (size_t i = 0; i < o.size(); ++i) o[i] += t[i];
    layer_norm(o.data(), T_, D, w("sc_ln.w"), w("sc_ln.b"), eps_.at("sc_ln"));
    std::vector<float> pooled(D, 0.f);
    for (int r = 0; r < n_valid; ++r)
      for (int c = 0; c < D; ++c) pooled[c] += o[(size_t)r * D + c] / n_valid;
    proj_text_.assign(D, 0.f);
    linear(pooled.data(), 1, D, w("sc_text.w"), w("sc_text.b"), D, proj_text_.data());
    has_text_ = true;
    return (float)ms_since(t0);
  }

  // Full image: ViT + neck, then the detector and masks.
  std::vector<char> detect(const uint8_t* rgb, float mask_min) {
    tm_.assign(16, 0.f);
    auto t0 = Clock::now();
    patchify(rgb);
    dump("patches", *patches_);
    tm_[0] = (float)ms_since(t0);
    t0 = Clock::now();
    for (size_t i = 0; i < vit_.size(); ++i) {
      stages_.at(vit_[i])->run();
      if (g_dump) dump("out_" + vit_[i], i % 2 == 0 ? *va_ : *vb_);
    }
    tm_[1] = (float)ms_since(t0);
    t0 = Clock::now();
    to_image_order();
    tm_[2] = (float)ms_since(t0);
    t0 = Clock::now();
    dump("spatial", *sp_);
    for (const char* n : {"neck4", "neck2", "neck1"}) stages_.at(n)->run();
    dump("f4", *f4_);
    dump("f2", *f2_);
    dump("f1", *f1_);
    tm_[3] = (float)ms_since(t0);
    have_image_ = true;
    return detector(mask_min);
  }

  std::vector<char> detect_cached(float mask_min) {
    if (!have_image_) throw std::runtime_error("no image yet");
    tm_.assign(16, 0.f);
    return detector(mask_min);
  }

 private:
  const float* w(const std::string& k) const {
    auto it = index_.find(k);
    if (it == index_.end()) throw std::runtime_error("host weight missing: " + k);
    return host_.data() + it->second;
  }

  void patchify(const uint8_t* rgb) {
    const int P = 14, pc = 3 * P * P, W2 = WS_ * WS_, nWx = G_ / WS_;
    uint16_t* dst = patches_->p();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < G_ * G_; ++i) {
      const int w = i / W2, t = i % W2;
      const int gy = (w / nWx) * WS_ + t / WS_, gx = (w % nWx) * WS_ + t % WS_;
      uint16_t* o = dst + (size_t)i * pc;
      for (int c = 0; c < 3; ++c)
        for (int py = 0; py < P; ++py)
          for (int px = 0; px < P; ++px) {
            const uint8_t v = rgb[((size_t)(gy * P + py) * S_ + gx * P + px) * 3 + c];
            o[c * P * P + py * P + px] = to_bf16(v / 127.5f - 1.f);
          }
    }
    patches_->flush();
  }

  void to_image_order() {
    vit_out_->invalidate();
    const uint16_t* s = vit_out_->p();
    uint16_t* d = sp_->p();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < G_ * G_; ++i) std::memcpy(d + (size_t)order_[i] * 1024, s + (size_t)i * 1024, 1024 * 2);
    sp_->flush();
  }

  // sine embedding of boxes (cx, cy, w, h) -> 512 values, order (y, x, w, h) as HF encode_boxes
  void sine_boxes(const float* ref, float* out) const {
    const int F = D / 2;
    static float dim_t[128];
    static bool init = false;
    if (!init) {
      for (int i = 0; i < F; ++i) dim_t[i] = std::pow(10000.f, 2.f * (i / 2) / F);
      init = true;
    }
#pragma omp parallel for schedule(static)
    for (int q = 0; q < Q_; ++q) {
      const float* b = ref + q * 4;
      const float vals[4] = {b[1], b[0], b[2], b[3]};
      for (int k = 0; k < 4; ++k)
        for (int i = 0; i < F; i += 2) {
          const float a = vals[k] * 2.f * (float)M_PI;
          out[(size_t)q * 4 * F + k * F + i] = std::sin(a / dim_t[i]);
          out[(size_t)q * 4 * F + k * F + i + 1] = std::cos(a / dim_t[i + 1]);
        }
    }
  }

  // the box bias MLPs' inputs: log-scaled distances from each grid coordinate to the box's two edges,
  // as (1, 16, G, Q) tensors (HWC: [coordinate][query][edge, then 14 zeros: a 2-channel MLA input
  // miscompiles]); the MLPs and the expansion run on the MLA. Buffers start cleared, so the zeros stay.
  void box_log_deltas(const float* ref) {
    auto lg = [](float d) {
      const float v = d * 8.f;
      return (v > 0 ? 1.f : (v < 0 ? -1.f : 0.f)) * std::log2(std::fabs(v) + 1.f) / 3.f;
    };
    uint16_t* py = dy_->p();
    uint16_t* px = dx_->p();
#pragma omp parallel for schedule(static)
    for (int c = 0; c < G_; ++c) {
      const float coord = (float)c / G_;
      for (int q = 0; q < Q_; ++q) {
        const float* b = ref + q * 4;
        const size_t o = ((size_t)c * Q_ + q) * kBiasIn;
        py[o] = to_bf16(lg(coord - (b[1] - b[3] / 2)));
        py[o + 1] = to_bf16(lg(coord - (b[1] + b[3] / 2)));
        px[o] = to_bf16(lg(coord - (b[0] - b[2] / 2)));
        px[o + 1] = to_bf16(lg(coord - (b[0] + b[2] / 2)));
      }
    }
    dy_->flush();
    dx_->flush();
  }

  std::vector<char> detector(float mask_min) {
    if (!has_text_) throw std::runtime_error("no prompt yet");
    const int N = G_ * G_;
    auto t0 = Clock::now();
    stages_.at("enc")->run();
    dump("enc", *enc_);
    dump("tmask", *tmask_);
    tm_[4] = (float)ms_since(t0);
    // vis_k = enc + pos
    t0 = Clock::now();
    enc_->invalidate();
    {
      const uint16_t* e = enc_->p();
      uint16_t* k = visk_->p();
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < (size_t)N * D; ++i) k[i] = to_bf16(from_bf16(e[i]) + from_bf16(pos_[i]));
      visk_->flush();
    }
    // decoder
    std::vector<float> ref(w("ref0"), w("ref0") + (size_t)Q_ * 4), sine((size_t)Q_ * 512),
        hq((size_t)(Q_ + 1) * D), q((size_t)Q_ * D), delta((size_t)Q_ * D);
    {
      uint16_t* h0 = h0_->p();
      for (int c = 0; c < D; ++c) h0[c] = to_bf16(w("presence_token")[c]);
      dino::to_bf16_span(w("query_embed"), h0 + D, (size_t)Q_ * D);
      h0_->flush();
    }
    double mla = 0;
    for (int k = 0; k < 6; ++k) {
      sine_boxes(ref.data(), sine.data());
      dino::to_bf16_span(sine.data(), qsine_->p(), sine.size());
      qsine_->flush();
      box_log_deltas(ref.data());
      auto tm = Clock::now();
      stages_.at("dec" + std::to_string(k))->run();
      mla += ms_since(tm);
      Buffer& out = *dout_[k % 2];
      out.invalidate();
      if (g_dump) dump("dec" + std::to_string(k) + "_out", out);
      dino::from_bf16_span(out.p() + (size_t)(Q_ + 1) * D, delta.data(), delta.size());
      for (int i = 0; i < Q_; ++i)
        for (int j = 0; j < 4; ++j) {
          const float r = std::min(std::max(ref[i * 4 + j], 1e-3f), 1.f - 1e-3f);
          ref[i * 4 + j] = 1.f / (1.f + std::exp(-(std::log(r / (1.f - r)) + delta[(size_t)i * D + j])));
        }
      if (k == 5) {
        dino::from_bf16_span(out.p(), hq.data(), hq.size());
        std::memcpy(q.data(), hq.data() + D, (size_t)Q_ * D * 4);
        layer_norm(q.data(), Q_, D, w("out_ln.w"), w("out_ln.b"), eps_.at("out_ln"));
      }
    }
    tm_[5] = (float)mla;
    // presence and scores
    std::vector<float> p0(hq.begin(), hq.begin() + D), pa(D), pb(D);
    layer_norm(p0.data(), 1, D, w("pres_ln.w"), w("pres_ln.b"), eps_.at("pres_ln"));
    linear(p0.data(), 1, D, w("pres_head.l1.w"), w("pres_head.l1.b"), D, pa.data(), true);
    linear(pa.data(), 1, D, w("pres_head.l2.w"), w("pres_head.l2.b"), D, pb.data(), true);
    float presence = 0;
    linear(pb.data(), 1, D, w("pres_head.l3.w"), w("pres_head.l3.b"), 1, &presence);
    presence = std::min(std::max(presence, -10.f), 10.f);
    std::vector<float> pq((size_t)Q_ * D), logits(Q_);
    linear(q.data(), Q_, D, w("sc_query.w"), w("sc_query.b"), D, pq.data());
    for (int i = 0; i < Q_; ++i) {
      float s = 0;
      for (int c = 0; c < D; ++c) s += pq[(size_t)i * D + c] * proj_text_[c];
      logits[i] = std::min(std::max(s * sc_scale_, -12.f), 12.f);
    }
    tm_[6] = (float)ms_since(t0) - tm_[5];
    // masks for queries whose score (sigmoid(logit) * sigmoid(presence)) reaches mask_min
    t0 = Clock::now();
    const float pres_p = 1.f / (1.f + std::exp(-presence));
    std::vector<int> keep;
    for (int i = 0; i < Q_; ++i)
      if (pres_p / (1.f + std::exp(-logits[i])) >= mask_min) keep.push_back(i);
    std::sort(keep.begin(), keep.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    if (keep.size() > 100) keep.resize(100);
    const int MM = M_ * M_;
    std::vector<uint8_t> maskbits;
    if (!keep.empty()) {
      stages_.at("maskpre")->run();
      stages_.at("pixdec")->run();
      std::vector<float> m1((size_t)Q_ * D), m2((size_t)Q_ * D), m3((size_t)Q_ * D);
      linear(q.data(), Q_, D, w("mask_embed.0.w"), w("mask_embed.0.b"), D, m1.data(), true);
      linear(m1.data(), Q_, D, w("mask_embed.1.w"), w("mask_embed.1.b"), D, m2.data(), true);
      linear(m2.data(), Q_, D, w("mask_embed.2.w"), w("mask_embed.2.b"), D, m3.data());
      // query . pixel on the CPU, only for the kept queries (an MLA program with two dynamic
      // inputs and no weights does not compile)
      inst_->invalidate();
      const uint16_t* ip = inst_->p();
      maskbits.resize(keep.size() * (size_t)MM);
#pragma omp parallel for schedule(static)
      for (int p = 0; p < MM; ++p) {
        float px[256];
        dino::from_bf16_span(ip + (size_t)p * D, px, D);
        for (size_t j = 0; j < keep.size(); ++j) {
          const float* e = m3.data() + (size_t)keep[j] * D;
          float s = 0;
#pragma omp simd reduction(+ : s)
          for (int c = 0; c < D; ++c) s += e[c] * px[c];
          maskbits[j * MM + p] = s > 0 ? 1 : 0;
        }
      }
    }
    tm_[7] = (float)ms_since(t0);
    tm_[8] = (float)keep.size();
    // reply
    std::vector<char> out;
    auto put = [&](const void* p, size_t n) {
      const char* c = static_cast<const char*>(p);
      out.insert(out.end(), c, c + n);
    };
    put(logits.data(), (size_t)Q_ * 4);
    put(&presence, 4);
    put(ref.data(), (size_t)Q_ * 16);
    const uint32_t nm = keep.size();
    put(&nm, 4);
    for (size_t j = 0; j < keep.size(); ++j) {
      const uint32_t qi = keep[j];
      put(&qi, 4);
      put(maskbits.data() + j * MM, MM);
    }
    put(tm_.data(), 16 * 4);
    return out;
  }

  fs::path dir_;
  int S_ = 0, G_ = 0, WS_ = 0, Q_ = 0, T_ = 0, H_ = 0, M_ = 0, n_valid_ = 0;
  float sc_scale_ = 1;
  std::map<std::string, float> eps_;
  double load_s_ = 0;
  bool has_text_ = false, have_image_ = false;
  std::vector<float> host_, proj_text_, tm_;
  std::map<std::string, size_t> index_;
  std::vector<uint16_t> pos_;
  std::vector<int> order_;
  std::vector<std::string> vit_;
  std::unique_ptr<Buffer> patches_, va_, vb_, sp_, f4_, f2_, f1_, tin_, text_, tmask_, enc_, visk_, h0_, qsine_, dy_, dx_, e2_,
      inst_;
  std::unique_ptr<Buffer> dout_[2];
  Buffer* vit_out_ = nullptr;
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
  write_all(fd, "SAM3", 4);
  write_all(fd, &status, 4);
  write_all(fd, &n, 4);
  if (n) write_all(fd, payload.data(), n);
}

}  // namespace sam3

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: sam3_rt <bundle dir>\n");
    return 2;
  }
  std::signal(SIGTERM, [](int) { sam3::g_stop = 1; });
  std::signal(SIGINT, [](int) { sam3::g_stop = 1; });
  std::signal(SIGPIPE, SIG_IGN);
  const int proto = ::dup(1);
  ::dup2(2, 1);
  int rc = 0;
  llima::connect_mla_rt({});
  try {
    sam3::Model model(argv[1]);
    std::fwrite("SAM3READY\n", 1, 10, stderr);
    std::fflush(stderr);
    while (!sam3::g_stop) {
      char magic[4];
      uint32_t cmd = 0, n = 0;
      if (!sam3::read_all(0, magic, 4) || !sam3::read_all(0, &cmd, 4) || !sam3::read_all(0, &n, 4)) break;
      if (std::memcmp(magic, "SAM3", 4) != 0) throw std::runtime_error("bad request header");
      std::vector<char> in(n);
      if (n && !sam3::read_all(0, in.data(), n)) break;
      try {
        std::vector<char> out;
        if (cmd == 1) {
          const std::string s = model.info().dump();
          out.assign(s.begin(), s.end());
        } else if (cmd == 2) {
          const int nv = *reinterpret_cast<const uint32_t*>(in.data() + n - 4);
          const float ms = model.set_text(reinterpret_cast<const float*>(in.data()), nv);
          out.resize(4);
          std::memcpy(out.data(), &ms, 4);
        } else if (cmd == 3) {
          float mm;
          std::memcpy(&mm, in.data(), 4);
          out = model.detect(reinterpret_cast<const uint8_t*>(in.data() + 4), mm);
        } else if (cmd == 6) {
          float mm;
          std::memcpy(&mm, in.data(), 4);
          out = model.detect_cached(mm);
        } else if (cmd == 4) {
          sam3::reply(proto, 0, {});
          break;
        } else {
          throw std::runtime_error("unknown command " + std::to_string(cmd));
        }
        sam3::reply(proto, 0, out);
      } catch (const std::exception& e) {
        const std::string s = e.what();
        sam3::reply(proto, 1, std::vector<char>(s.begin(), s.end()));
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "sam3_rt: fatal: %s\n", e.what());
    rc = 1;
  }
  llima::disconnect_mla_rt();
  return rc;
}
