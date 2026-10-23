// dino_rt: Grounding DINO on the Modalix MLA.
//
// Runs the 21 compiled stage ELFs (text, backbone, 7 encoder, decoder values,
// 6 + 6 decoder) with SimaLMM's MLAModelWithBuffer, and does on the CPU what
// the MLA cannot: deformable-attention sampling (NEON, msda.hpp), the two-stage
// top-900 selection, and the reference-box update between decoder layers.
//
// Every stage tensor is tokens x 256 bf16 in device memory (plain HWC layout of
// a (1, 256, 1, W) tensor); stages are packed along W (see model/dino_mla.py).
// Host code reads and writes those buffers in place.
//
// Protocol (stdin/stdout, little endian). Replies come back in request order, but a
// client may send the next DETECT before the last reply arrived (Live mode): with two
// "slots" (DINO_SLOTS=2, the default) two frames are in flight, one on the MLA while the
// other's deformable sampling runs on the CPU. The slots share the loaded programs
// (SimaLMM keys them by ELF path) and have their own device buffers.
//   request : "DINO" u32 cmd u32 nbytes payload
//   response: "DINO" u32 status(0 ok) u32 nbytes payload
//   cmd 1 INFO     -> JSON
//   cmd 2 TEXT     payload f32 text_in[64][832] + f32 tp[193][256]  -> f32 timing_ms
//   cmd 3 DETECT   payload u32 valid_w, u32 valid_h, u8 rgb[S][S][3] (top-left letterbox)
//                  -> f32 logits[900][64], f32 boxes[900][4] (cx,cy,w,h), f32 timings[32]
//   cmd 4 QUIT
//   cmd 6 DETECT_CACHED  rerun on the last image (prompt changed) -> as DETECT
//   cmd 5 DETECT_TOKENS (debug) payload f32 backbone tokens[N][256] -> as DETECT
// Logs go to stderr.
#include <sima_lmm/mla_buffer.hpp>
#include <sima_lmm/mla_model.hpp>

#include <nlohmann/json.hpp>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <set>
#include <thread>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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

namespace dino {

constexpr int T = 64, NQ = 900, TPW = 193;

static double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

static std::vector<char> read_file(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + p.string());
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

template <typename V>
static std::vector<V> read_vec(const fs::path& p, size_t count) {
  auto raw = read_file(p);
  if (raw.size() != count * sizeof(V))
    throw std::runtime_error(p.string() + ": expected " + std::to_string(count * sizeof(V)) + " bytes, got " +
                             std::to_string(raw.size()));
  std::vector<V> v(count);
  std::memcpy(v.data(), raw.data(), raw.size());
  return v;
}

// A bf16 device buffer of `tokens` x 256 values.
struct Buffer {
  Buffer(const std::string& name, size_t elems) : n(elems), b(name, {1, elems}, "bfloat16", false) {
    b.allocate();
    b.clear();
  }
  ~Buffer() { b.free(); }
  uint16_t* p() { return static_cast<uint16_t*>(b.get_virtual_addr()); }
  uint16_t* tok(size_t t) { return p() + t * kD; }
  void flush(size_t tok0, size_t ntok) { b.flush_cache(tok0 * kD * 2, ntok * kD * 2); }
  void invalidate(size_t tok0, size_t ntok) { b.invalidate_cache(tok0 * kD * 2, ntok * kD * 2); }
  size_t n;
  llima::MLABuffer b;
};

static llima::MLABufferSlice span(Buffer& b, size_t tok0, size_t ntok) {
  if (tok0 == 0 && ntok * kD == b.n) return llima::MLABufferSlice(&b.b);
  return llima::MLABufferSlice(&b.b, {0u, (uint32_t)(tok0 * kD)}, {1u, (uint32_t)(ntok * kD)});
}

static llima::MLABufferSlice elem_span(Buffer& b, size_t off, size_t n) {
  if (off == 0 && n == b.n) return llima::MLABufferSlice(&b.b);
  return llima::MLABufferSlice(&b.b, {0u, (uint32_t)off}, {1u, (uint32_t)n});
}

// One step of the backbone program (manifest "backbone", written by tools/make_bundle.py).
struct BbStep {
  std::string op, name, src, dst;
  size_t in_elems = 0, out_elems = 0, dst_off = 0, tokens = 0;
  int k = 1, c = 0, grid = 0;
  std::vector<int32_t> table;
};

// One MLA program runs at a time. When two slots want the MLA, the older request (lower
// sequence number) goes first, so the frame that is further along finishes first.
class MlaArbiter {
 public:
  void acquire(uint64_t prio) {
    std::unique_lock<std::mutex> l(m_);
    auto it = waiting_.insert(prio);
    cv_.wait(l, [&] { return !busy_ && *waiting_.begin() == prio; });
    waiting_.erase(it);
    busy_ = true;
  }
  void release() {
    {
      std::lock_guard<std::mutex> l(m_);
      busy_ = false;
    }
    cv_.notify_all();
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  std::multiset<uint64_t> waiting_;
  bool busy_ = false;
};

static MlaArbiter g_mla;
// Slots in a CPU phase right now share the cores; a slot waiting for or running an MLA
// program leaves them to the other.
static std::atomic<int> g_cpu{0};
static int g_max_threads = 1;
static bool g_multi_slot = false;   // queued back-to-back programs would hold the MLA too long

// DINO_TRACE=<file>: every MLA turn (slot, wait start, start, end in us) for timeline plots.
struct TraceEv {
  int slot;
  uint64_t seq;
  double wait, start, end;
  std::string what;
};
static std::mutex g_trace_m;
static std::vector<TraceEv> g_trace;
static const char* g_trace_path = std::getenv("DINO_TRACE");
static const Clock::time_point g_t0 = Clock::now();
static double now_us() { return std::chrono::duration<double, std::micro>(Clock::now() - g_t0).count(); }
static thread_local int t_slot = 0;
static thread_local bool t_in_cpu = false;
static void cpu_phase() {
  if (!t_in_cpu) {
    ++g_cpu;
    t_in_cpu = true;
  }
  omp_set_num_threads(std::max(1, g_max_threads / std::max(1, g_cpu.load())));
}
static void cpu_leave() {
  if (t_in_cpu) {
    --g_cpu;
    t_in_cpu = false;
  }
}

struct MlaTurn {
  explicit MlaTurn(uint64_t prio, const char* what = "mla") : prio_(prio), what_(what) {
    cpu_leave();
    if (g_trace_path) wait_ = now_us();
    g_mla.acquire(prio);
    if (g_trace_path) start_ = now_us();
  }
  ~MlaTurn() {
    g_mla.release();
    if (g_trace_path) {
      std::lock_guard<std::mutex> l(g_trace_m);
      g_trace.push_back({t_slot, prio_, wait_, start_, now_us(), what_});
    }
  }
  uint64_t prio_;
  const char* what_;
  double wait_ = 0, start_ = 0;
};

struct Stage {
  std::string name;
  std::unique_ptr<llima::MLAModelWithBuffer> model;
  double last_ms = 0;
  void run(uint64_t prio) {
    MlaTurn turn(prio, name.c_str());
    auto t = Clock::now();
    model->run();
    last_ms = ms_since(t);
  }
};

class Detector {
 public:
  // slot 0 reports load progress; further slots bind the same (already loaded) ELFs to
  // their own buffers.
  Detector(const fs::path& bundle, const fs::path& text_bundle, int slot = 0) : dir_(bundle), slot_(slot) {
    const json m = json::parse(read_file(bundle / "manifest.json"));
    S_ = m.at("size");
    N_ = m.at("tokens");
    for (int l = 0; l < kLevels; ++l) {
      const auto& lv = m.at("levels").at(l);
      levels_[l] = Level{lv.at(0).get<int>(), lv.at(1).get<int>(), lv.at(2).get<size_t>()};
    }
    const size_t N = N_;
    pos_ = read_vec<uint16_t>(bundle / "pos.bf16", N * kD);
    ref_ = read_vec<float>(bundle / "ref.f32", N * 2);
    proposals_ = read_vec<float>(bundle / "proposals.f32", N * 4);
    valid_ = read_vec<uint8_t>(bundle / "valid.u8", N);
    invalid_query_ = read_vec<float>(bundle / "invalid_query.f32", kD);
    query_embed_ = read_vec<uint16_t>(bundle / "query_embed.bf16", NQ * kD);

    // ---- backbone program: Swin blocks in window layout + CPU token gathers
    size_t need_a = 0, need_b = 0, need_p = 0;
    for (const auto& j : m.at("backbone")) {
      BbStep st;
      st.op = j.at("op");
      st.name = j.value("name", "");
      st.src = j.value("src", "");
      st.dst = j.value("dst", "");
      st.in_elems = j.value("in_elems", 0);
      st.out_elems = j.value("out_elems", 0);
      st.dst_off = j.value("dst_off", 0);
      st.tokens = j.value("tokens", 0);
      st.k = j.value("k", 1);
      st.c = j.value("c", 0);
      st.grid = j.value("grid", 0);
      if (j.contains("table")) {
        const fs::path tp = bundle / j.at("table").get<std::string>();
        st.table = read_vec<int32_t>(tp, fs::file_size(tp) / 4);
      }
      auto grow = [](size_t& v, const std::string& b, const std::string& want, size_t n) {
        if (b == want) v = std::max(v, n);
      };
      if (st.op == "patchify") need_a = std::max(need_a, st.table.size() * 48);
      if (st.op == "run") {
        for (auto [v, w] : {std::pair<size_t*, std::string>{&need_a, "A"}, {&need_b, "B"}, {&need_p, "P"}}) {
          grow(*v, st.src, w, st.in_elems);
          grow(*v, st.dst, w, st.out_elems);
        }
      }
      if (st.op == "gather") {
        const size_t out = st.table.size() * st.c;
        for (auto [v, w] : {std::pair<size_t*, std::string>{&need_a, "A"}, {&need_b, "B"}, {&need_p, "P"}})
          grow(*v, st.dst, w, out);
      }
      bb_.push_back(std::move(st));
    }
    bb_a_ = std::make_unique<Buffer>("bb_a", need_a);
    bb_b_ = std::make_unique<Buffer>("bb_b", need_b);
    bb_p_ = std::make_unique<Buffer>("bb_p", std::max<size_t>(need_p, kD));

    // ---- device buffers (tokens x 256 bf16)
    txt_in_ = std::make_unique<Buffer>("txt_in", (size_t)T * 832);
    txt_out_ = std::make_unique<Buffer>("txt_out", (size_t)T * kD);
    e0_in_ = std::make_unique<Buffer>("e0_in", (2 * N + TPW) * kD);          // [X | P | TP]
    for (auto& b : ek_in_) b = std::make_unique<Buffer>("ek_in", (3 * N + TPW) * kD);  // [X | S | P | TP]
    e_out_ = std::make_unique<Buffer>("e_out", (4 * N + T) * kD);           // [X1 | V | OFF | AW | T]
    e6_out_ = std::make_unique<Buffer>("e6_out", (2 * N + T) * kD);         // [HEAD | M | T]
    dv_out_ = std::make_unique<Buffer>("dv_out", 6 * N * kD);               // [V0..V5]
    for (auto& b : dpre_in_) b = std::make_unique<Buffer>("dpre_in", (3 * NQ + TPW) * kD);  // [H|QS0|QS1|TP]
    dpre_out_ = std::make_unique<Buffer>("dpre_out", 3 * NQ * kD);          // [H2 | OFF | AW]
    dpost_in_ = std::make_unique<Buffer>("dpost_in", (2 * NQ + TPW) * kD);  // [H2 | S | TP]
    dpost_out_ = std::make_unique<Buffer>("dpost_out", 2 * NQ * kD);        // [H' | HEAD]

    // persistent regions: position embeddings, decoder query embeddings
    std::memcpy(e0_in_->tok(N), pos_.data(), N * kD * 2);
    e0_in_->flush(N, N);
    for (auto& b : ek_in_) {
      std::memcpy(b->tok(2 * N), pos_.data(), N * kD * 2);
      b->flush(2 * N, N);
    }
    std::memcpy(dpre_in_[0]->tok(0), query_embed_.data(), NQ * kD * 2);
    dpre_in_[0]->flush(0, NQ);

    // ---- stages
    const auto elf = [&](const std::string& n) { return (bundle / "elf" / (n + ".elf")).string(); };
    auto add = [&](const std::string& n, const std::string& path, std::vector<llima::MLABufferSlice> in,
                   std::vector<llima::MLABufferSlice> out) {
      auto s = std::make_unique<Stage>();
      s->name = n;
      s->model = std::make_unique<llima::MLAModelWithBuffer>(path, in, out);
      stages_[n] = std::move(s);
    };
    add("text", (text_bundle / "text_t64.elf").string(), {span(*txt_in_, 0, T * 832 / kD)},
        {span(*txt_out_, 0, T)});
    for (auto& st : bb_) {
      if (st.op != "run") continue;
      if (!fs::exists(elf(st.name))) {
        std::fprintf(stderr, "dino_rt: %s.elf missing; only DETECT_TOKENS works\n", st.name.c_str());
        has_backbone_ = false;
        continue;
      }
      add(st.name, elf(st.name), {elem_span(buf(st.src), 0, st.in_elems)},
          {st.dst == "X" ? elem_span(*e0_in_, st.dst_off * kD, st.out_elems) : elem_span(buf(st.dst), 0, st.out_elems)});
    }
    add("enc0", elf("enc0"), {span(*e0_in_, 0, 2 * N + TPW)}, {span(*e_out_, 0, 4 * N + T)});
    for (int k = 1; k <= 5; ++k)
      add("enc" + std::to_string(k), elf("enc" + std::to_string(k)), {span(*ek_in_[k % 2], 0, 3 * N + TPW)},
          {span(*e_out_, 0, 4 * N + T)});
    add("enc6", elf("enc6"), {span(*ek_in_[0], 0, 3 * N + TPW)}, {span(*e6_out_, 0, 2 * N + T)});
    add("dec_values", elf("dec_values"), {span(*e6_out_, N, N)}, {span(*dv_out_, 0, 6 * N)});
    for (int k = 0; k < 6; ++k) {
      add("dec_pre" + std::to_string(k), elf("dec_pre" + std::to_string(k)),
          {span(*dpre_in_[k % 2], 0, 3 * NQ + TPW)}, {span(*dpre_out_, 0, 3 * NQ)});
      add("dec_post" + std::to_string(k), elf("dec_post" + std::to_string(k)), {span(*dpost_in_, 0, 2 * NQ + TPW)},
          {span(*dpost_out_, 0, 2 * NQ)});
    }
    auto t = Clock::now();
    // Progress lines for the app's loading bar: "DINOLOAD <done> <total> <stage>".
    size_t done = 0;
    for (auto& [n, s] : stages_) {
      if (slot_ == 0) {
        std::fprintf(stderr, "DINOLOAD %zu %zu %s\n", done, stages_.size(), n.c_str());
        std::fflush(stderr);
      }
      s->model->load();
      ++done;
    }
    if (slot_ == 0) std::fprintf(stderr, "DINOLOAD %zu %zu done\n", done, stages_.size());
    std::fprintf(stderr, "dino_rt: slot %d: S=%d N=%zu, %zu stages loaded in %.0f ms\n", slot_, S_, N_,
                 stages_.size(), ms_since(t));
  }

  // SimaLMM keys programs by path: load() of an already loaded ELF and a second free() do
  // nothing, so whichever slot goes first frees the shared programs.
  ~Detector() {
    for (auto& [n, s] : stages_) s->model->free();
  }

  uint64_t prio = 0;  // sequence number of the request being served (MLA priority)

  json info() const {
    return {{"size", S_}, {"tokens", N_}, {"queries", NQ}, {"text_tokens", T}, {"stages", stages_.size()},
            {"threads", g_max_threads}};
  }

  // text_in [64][832] (embeddings | additive mask, key on C); tp [193][256] (T columns ignored)
  double set_text(const float* text_in, const float* tp) {
    auto t0 = Clock::now();
    to_bf16_span(text_in, txt_in_->p(), (size_t)T * 832);
    txt_in_->b.flush_cache();
    stages_["text"]->run(prio);
    txt_out_->b.invalidate_cache();
    put_text(txt_out_->p(), tp);
    return ms_since(t0);
  }
  const uint16_t* text_features() { return txt_out_->p(); }

  // text features (T x 256 bf16, the text stage's output) + text pack -> every stage input
  void put_text(const uint16_t* feats, const float* tp) {
    std::vector<uint16_t> tpb((size_t)TPW * kD);
    to_bf16_span(tp, tpb.data(), tpb.size());
    std::memcpy(tpb.data(), feats, (size_t)T * kD * 2);   // text features into cols 0..63
    const size_t N = N_;
    auto put = [&](Buffer& b, size_t at) {
      std::memcpy(b.tok(at), tpb.data(), tpb.size() * 2);
      b.flush(at, TPW);
    };
    put(*e0_in_, 2 * N);
    for (auto& b : ek_in_) put(*b, 3 * N);
    for (auto& b : dpre_in_) put(*b, 3 * NQ);
    put(*dpost_in_, 2 * NQ);
    have_text_ = true;
  }

  // rgb: S x S x 3 uint8, image at the top-left, valid_w x valid_h; the rest is padding.
  void detect(const uint8_t* rgb, int valid_w, int valid_h, float* logits, float* boxes, float* timing) {
    if (!have_text_) throw std::runtime_error("no prompt set");
    const size_t N = N_;
    std::fill(timing, timing + 32, 0.f);
    auto t0 = Clock::now();
    // layer 0's decoder queries are constant; dpre_in_[0] is reused by layers 2 and 4
    std::memcpy(dpre_in_[0]->tok(0), query_embed_.data(), (size_t)NQ * kD * 2);
    if (rgb) {
      if (!has_backbone_) throw std::runtime_error("this bundle has no backbone");
      run_backbone(rgb, valid_w, valid_h, timing);
      has_image_ = true;
    }

    // ---- encoder
    double mla = 0, cpu = 0;
    for (int k = 0; k <= 6; ++k) {
      Stage& st = *stages_["enc" + std::to_string(k)];
      st.run(prio);
      mla += st.last_ms;
      if (k == 6) break;
      auto tc = Clock::now();
      cpu_phase();
      Buffer& next = (k + 1 == 6) ? *ek_in_[0] : *ek_in_[(k + 1) % 2];
      e_out_->invalidate(0, 4 * N + T);
      // X1 -> next X; T -> next TP text columns; sampled -> next S
      std::memcpy(next.tok(0), e_out_->tok(0), N * kD * 2);
      std::memcpy(next.tok(3 * N), e_out_->tok(4 * N), (size_t)T * kD * 2);
      MsdaArgs a;
      a.value = e_out_->tok(N);
      a.offsets = e_out_->tok(2 * N);
      a.logits = e_out_->tok(3 * N);
      a.out = next.tok(N);
      a.ref = ref_.data();
      a.ref_dims = 2;
      a.queries = (int)N;
      std::copy(levels_, levels_ + kLevels, a.levels);
      msda(a);
      next.flush(0, 2 * N);
      next.flush(3 * N, T);
      cpu += ms_since(tc);
    }
    timing[2] = mla;
    timing[3] = cpu;

    // ---- decoder value maps (bound to the memory columns of enc6's output)
    stages_["dec_values"]->run(prio);
    timing[4] = stages_["dec_values"]->last_ms;

    // ---- two-stage top-k on the host
    auto tk = Clock::now();
    cpu_phase();
    e6_out_->invalidate(0, 2 * N + T);
    dv_out_->b.invalidate_cache();
    std::vector<float> t6((size_t)T * kD);
    from_bf16_span(e6_out_->tok(2 * N), t6.data(), t6.size());
    float inv_logit[T];
    for (int t = 0; t < T; ++t) {
      double s = 0;
      for (int c = 0; c < kD; ++c) s += (double)invalid_query_[c] * t6[(size_t)t * kD + c];
      inv_logit[t] = (float)s;
    }
    std::vector<float> score(N);
    const uint16_t* head = e6_out_->tok(0);
#pragma omp parallel for schedule(static)
    for (size_t n = 0; n < N; ++n) {
      float best = -INFINITY;
      for (int t = 0; t < T; ++t) {
        if (!token_mask_[t]) continue;
        const float v = valid_[n] ? from_bf16(head[n * kD + t]) : inv_logit[t];
        best = v > best ? v : best;
      }
      score[n] = best;
    }
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + NQ, idx.end(), [&](int a, int b) {
      return score[a] > score[b] || (score[a] == score[b] && a < b);
    });
    std::vector<float> L(NQ * 4), ref4(NQ * 4);
    for (int q = 0; q < NQ; ++q) {
      const size_t n = idx[q];
      for (int i = 0; i < 4; ++i) {
        const float d = from_bf16(head[n * kD + T + i]);
        L[q * 4 + i] = d + proposals_[n * 4 + i];  // +inf where the proposal is invalid
        ref4[q * 4 + i] = sigmoid(L[q * 4 + i]);
      }
    }
    // decoder text: T6 into the TP text columns of every decoder input
    for (auto& b : dpre_in_) {
      std::memcpy(b->tok(3 * NQ), e6_out_->tok(2 * N), (size_t)T * kD * 2);
      b->flush(3 * NQ, T);
    }
    std::memcpy(dpost_in_->tok(2 * NQ), e6_out_->tok(2 * N), (size_t)T * kD * 2);
    dpost_in_->flush(2 * NQ, T);
    timing[5] = ms_since(tk);

    // ---- decoder
    double dmla = 0, dcpu = 0;
    std::vector<float> qs((size_t)NQ * 512);
    for (int k = 0; k < 6; ++k) {
      auto tc = Clock::now();
      cpu_phase();
      Buffer& pin = *dpre_in_[k % 2];
      sine_box(ref4.data(), qs.data());
#pragma omp parallel for schedule(static)
      for (int q = 0; q < NQ; ++q) {
        to_bf16_span(qs.data() + (size_t)q * 512, pin.tok(NQ + q), kD);
        to_bf16_span(qs.data() + (size_t)q * 512 + kD, pin.tok(2 * NQ + q), kD);
      }
      pin.flush(0, 3 * NQ);
      dcpu += ms_since(tc);
      Stage& pre = *stages_["dec_pre" + std::to_string(k)];
      pre.run(prio);
      dmla += pre.last_ms;
      tc = Clock::now();
      dpre_out_->b.invalidate_cache();
      std::memcpy(dpost_in_->tok(0), dpre_out_->tok(0), (size_t)NQ * kD * 2);
      MsdaArgs a;
      a.value = dv_out_->tok((size_t)k * N);
      a.offsets = dpre_out_->tok(NQ);
      a.logits = dpre_out_->tok(2 * NQ);
      a.out = dpost_in_->tok(NQ);
      a.ref = ref4.data();
      a.ref_dims = 4;
      a.queries = NQ;
      std::copy(levels_, levels_ + kLevels, a.levels);
      msda(a);
      dpost_in_->flush(0, 2 * NQ);
      dcpu += ms_since(tc);
      Stage& post = *stages_["dec_post" + std::to_string(k)];
      post.run(prio);
      dmla += post.last_ms;
      tc = Clock::now();
      dpost_out_->b.invalidate_cache();
      const uint16_t* h = dpost_out_->tok(NQ);
      if (k < 5) {
        // next layer's H, and the box update in logit space (HF: sigmoid(delta + logit(ref, 1e-5)))
        std::memcpy(dpre_in_[(k + 1) % 2]->tok(0), dpost_out_->tok(0), (size_t)NQ * kD * 2);
        for (int q = 0; q < NQ; ++q)
          for (int i = 0; i < 4; ++i)
            ref4[q * 4 + i] = sigmoid(from_bf16(h[(size_t)q * kD + i]) + logit(ref4[q * 4 + i]));
      } else {
        for (int q = 0; q < NQ; ++q) {
          for (int i = 0; i < 4; ++i)
            boxes[q * 4 + i] = sigmoid(from_bf16(h[(size_t)q * kD + 4 + i]) + logit(ref4[q * 4 + i]));
          for (int t = 0; t < T; ++t)
            logits[q * T + t] = token_mask_[t] ? from_bf16(h[(size_t)q * kD + 8 + t]) : -INFINITY;
        }
      }
      dcpu += ms_since(tc);
    }
    timing[6] = dmla;
    timing[7] = dcpu;
    timing[8] = ms_since(t0);
  }

  // Debug path: put backbone output tokens (N x 256 floats) in place of running the backbone.
  void put_tokens(const float* tokens) {
    to_bf16_span(tokens, e0_in_->p(), N_ * kD);
    e0_in_->flush(0, N_);
    has_image_ = true;
  }
  size_t tokens() const { return N_; }
  bool has_image() const { return has_image_; }

  void set_token_mask(const float* tp) {
    // TP column 192 holds the additive token mask on channels 0..63.
    for (int t = 0; t < T; ++t) token_mask_[t] = tp[(size_t)192 * kD + t] > -1.0f;
  }

  int size() const { return S_; }

 private:
  static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
  static float logit(float x) {
    const float e = 1e-5f;
    x = std::min(std::max(x, e), 1.0f - e);
    return std::log(x / (1.0f - x));
  }

  // get_sine_pos_embed(ref, 128): per coordinate 128 dims, sin on even, cos on odd;
  // output order (y, x, w, h) because HF swaps x and y.
  static void sine_box(const float* ref, float* out) {
    static float inv_dim[128];
    static bool init = false;
    if (!init) {
      for (int i = 0; i < 128; ++i) inv_dim[i] = (float)(1.0 / std::pow(10000.0, 2.0 * (i / 2) / 128.0));
      init = true;
    }
    const int order[4] = {1, 0, 2, 3};
#pragma omp parallel for schedule(static)
    for (int q = 0; q < NQ; ++q)
      for (int c = 0; c < 4; ++c) {
        const float v = ref[q * 4 + order[c]] * 6.283185307179586f;
        float* o = out + (size_t)q * 512 + c * 128;
        for (int i = 0; i < 128; i += 2) {
          o[i] = std::sin(v * inv_dim[i]);
          o[i + 1] = std::cos(v * inv_dim[i + 1]);
        }
      }
  }

  Buffer& buf(const std::string& n) {
    if (n == "A") return *bb_a_;
    if (n == "B") return *bb_b_;
    if (n == "P") return *bb_p_;
    if (n == "X") return *e0_in_;
    throw std::runtime_error("unknown buffer " + n);
  }

  // The backbone program: patchify, Swin blocks (MLA), token gathers (CPU), level projections.
  void run_backbone(const uint8_t* rgb, int vw, int vh, float* timing) {
    double mla = 0, cpu = 0;
    for (size_t si = 0; si < bb_.size(); ++si) {
      auto& st = bb_[si];
      auto t = Clock::now();
      if (st.op == "run" && queue_ && !g_multi_slot && si + 1 < bb_.size() && bb_[si + 1].op == "run") {
        // Back-to-back MLA programs (no CPU step between): one queued submission.
        size_t sj = si;
        MlaTurn turn(prio);
        t = Clock::now();
        while (sj < bb_.size() && bb_[sj].op == "run") stages_[bb_[sj++].name]->model->add_to_queue();
        llima::MLAModelWithBuffer::run_queue();
        mla += ms_since(t);
        si = sj - 1;
        continue;
      }
      cpu_phase();
      if (st.op == "patchify") {
        patchify(rgb, vw, vh, st.table, st.grid);
        cpu += ms_since(t);
        timing[0] = (float)ms_since(t);
      } else if (st.op == "run") {
        Stage& s = *stages_[st.name];
        s.run(prio);
        mla += s.last_ms;
      } else if (st.op == "gather") {
        Buffer& src = buf(st.src);
        Buffer& dst = buf(st.dst);
        src.b.invalidate_cache();
        const size_t c = st.c, rows = st.table.size() / st.k;
        uint16_t* out = dst.p() + st.dst_off * kD;
        const uint16_t* in = src.p();
#pragma omp parallel for schedule(static)
        for (size_t r = 0; r < rows; ++r)
          for (int j = 0; j < st.k; ++j) {
            const int32_t idx = st.table[r * st.k + j];
            uint16_t* d = out + (r * st.k + j) * c;
            if (idx < 0)
              std::memset(d, 0, c * 2);
            else
              std::memcpy(d, in + (size_t)idx * c, c * 2);
          }
        dst.b.flush_cache(st.dst_off * kD * 2, rows * st.k * c * 2);
        cpu += ms_since(t);
      }
    }
    timing[1] = (float)mla;
    timing[9] = (float)cpu;
  }

  // Normalise + 4x4 space-to-depth into buffer A, one 48-value token per entry of `table`
  // (spatial index y * grid + x of stage-1 token t in window layout), channel c*16 + py*4 + px.
  void patchify(const uint8_t* rgb, int vw, int vh, const std::vector<int32_t>& table, int grid) {
    static const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
    const int S = S_;
    uint16_t* dst = bb_a_->p();
#pragma omp parallel for schedule(static)
    for (size_t t = 0; t < table.size(); ++t) {
      float px[48];
      const int y4 = table[t] / grid, x4 = table[t] % grid;
      for (int py = 0; py < 4; ++py)
        for (int pxx = 0; pxx < 4; ++pxx) {
          const int y = y4 * 4 + py, x = x4 * 4 + pxx;
          const bool in = x < vw && y < vh && y < S && x < S;
          const uint8_t* p = rgb + ((size_t)std::min(y, S - 1) * S + std::min(x, S - 1)) * 3;
          for (int c = 0; c < 3; ++c)
            px[c * 16 + py * 4 + pxx] = in ? (p[c] * (1.0f / 255.0f) - mean[c]) / stdv[c] : 0.0f;
        }
      to_bf16_span(px, dst + t * 48, 48);
    }
    bb_a_->b.flush_cache();
  }

  fs::path dir_;
  int slot_ = 0;
  int S_ = 0;
  size_t N_ = 0;
  Level levels_[kLevels];
  std::vector<uint16_t> pos_, query_embed_;
  std::vector<float> ref_, proposals_, invalid_query_;
  std::vector<uint8_t> valid_;
  bool token_mask_[T] = {};
  bool have_text_ = false;
  bool has_image_ = false;
  std::vector<BbStep> bb_;
  // Queued back-to-back programs hold the MLA longer; with two slots the other frame
  // would wait, so the queue is only used with one slot.
  bool queue_ = std::getenv("DINO_NO_QUEUE") == nullptr;
  bool has_backbone_ = true;
  std::unique_ptr<Buffer> bb_a_, bb_b_, bb_p_, txt_in_, txt_out_, e0_in_, ek_in_[2], e_out_, e6_out_, dv_out_, dpre_in_[2],
      dpre_out_, dpost_in_, dpost_out_;
  std::map<std::string, std::unique_ptr<Stage>> stages_;
};

// ------------------------------------------------------------------ protocol

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

// Reads requests on the main thread; DETECT-type requests go to the slot with less work,
// everything else waits until the slots are idle. Replies are written in request order.
class Server {
 public:
  Server(std::vector<std::unique_ptr<Detector>>& det, int out) : det_(det), out_(out), slots_(det.size()) {
    for (size_t i = 0; i < det_.size(); ++i) workers_.emplace_back([this, i] { work(i); });
  }
  ~Server() {
    {
      std::lock_guard<std::mutex> l(m_);
      quit_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }

  int run() {
    const int in = 0;
    while (!g_stop) {
      char magic[4];
      uint32_t cmd = 0, n = 0;
      if (!read_all(in, magic, 4) || !read_all(in, &cmd, 4) || !read_all(in, &n, 4)) break;
      if (std::memcmp(magic, "DINO", 4) != 0) {
        std::fprintf(stderr, "dino_rt: bad request header\n");
        drain();
        return 1;
      }
      Job job{next_seq_++, cmd, std::vector<char>(n)};
      if (n && !read_all(in, job.payload.data(), n)) break;
      if (cmd == 3 || cmd == 5 || cmd == 6) {
        std::lock_guard<std::mutex> l(m_);
        size_t k;
        if (cmd == 6) {
          k = last_image_slot_;
        } else {
          // the slot with less queued work; on a tie, not the one used last
          k = (last_image_slot_ + 1) % slots_.size();
          for (size_t i = 0; i < slots_.size(); ++i)
            if (load(i) < load(k)) k = i;
          last_image_slot_ = k;
        }
        slots_[k].q.push_back(std::move(job));
        cv_.notify_all();
        continue;
      }
      drain();
      try {
        if (cmd == 1) {
          json j = det_[0]->info();
          j["slots"] = det_.size();
          const std::string s = j.dump();
          reply(job.seq, 0, s.data(), s.size());
        } else if (cmd == 2) {
          const size_t need = ((size_t)T * 832 + (size_t)TPW * kD) * 4;
          if (n != need) throw std::runtime_error("TEXT payload size");
          const float* f = reinterpret_cast<const float*>(job.payload.data());
          const float* tp = f + (size_t)T * 832;
          for (auto& d : det_) d->set_token_mask(tp);
          det_[0]->prio = job.seq;
          const float ms = (float)det_[0]->set_text(f, tp);
          for (size_t i = 1; i < det_.size(); ++i) det_[i]->put_text(det_[0]->text_features(), tp);
          reply(job.seq, 0, &ms, 4);
        } else if (cmd == 4) {
          reply(job.seq, 0, nullptr, 0);
          return 0;
        } else {
          throw std::runtime_error("unknown command " + std::to_string(cmd));
        }
      } catch (const std::exception& e) {
        std::fprintf(stderr, "dino_rt: %s\n", e.what());
        reply(job.seq, 1, e.what(), std::strlen(e.what()));
      }
    }
    drain();
    return 0;
  }

 private:
  struct Job {
    uint64_t seq;
    uint32_t cmd;
    std::vector<char> payload;
  };
  struct Slot {
    std::deque<Job> q;
    bool busy = false;
  };

  size_t load(size_t i) const { return slots_[i].q.size() + (slots_[i].busy ? 1 : 0); }

  void drain() {
    std::unique_lock<std::mutex> l(m_);
    cv_.wait(l, [&] {
      for (size_t i = 0; i < slots_.size(); ++i)
        if (load(i)) return false;
      return true;
    });
  }

  void work(size_t k) {
    Detector& det = *det_[k];
    t_slot = (int)k;
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [&] { return quit_ || !slots_[k].q.empty(); });
        if (slots_[k].q.empty()) return;
        job = std::move(slots_[k].q.front());
        slots_[k].q.pop_front();
        slots_[k].busy = true;
      }
      det.prio = job.seq;
      std::vector<float> res((size_t)NQ * T + NQ * 4 + 32);
      std::string error;
      try {
        float* lg = res.data();
        float* bx = res.data() + (size_t)NQ * T;
        float* tm = res.data() + (size_t)NQ * T + NQ * 4;
        const size_t n = job.payload.size();
        if (job.cmd == 3) {
          const size_t S = det.size();
          if (n != 8 + S * S * 3) throw std::runtime_error("DETECT payload size");
          uint32_t vw, vh;
          std::memcpy(&vw, job.payload.data(), 4);
          std::memcpy(&vh, job.payload.data() + 4, 4);
          det.detect(reinterpret_cast<const uint8_t*>(job.payload.data() + 8), (int)vw, (int)vh, lg, bx, tm);
        } else if (job.cmd == 5) {
          // DETECT_TOKENS (debug): f32 backbone tokens [N][256] -> same reply as DETECT
          if (n != det.tokens() * kD * 4) throw std::runtime_error("DETECT_TOKENS payload size");
          det.put_tokens(reinterpret_cast<const float*>(job.payload.data()));
          det.detect(nullptr, 0, 0, lg, bx, tm);
        } else {
          // DETECT_CACHED: the same image as last time (its backbone output is still in this
          // slot's buffers), e.g. after a prompt change -> same reply as DETECT
          if (!det.has_image()) throw std::runtime_error("no image yet");
          det.detect(nullptr, 0, 0, lg, bx, tm);
        }
        tm[10] = (float)k;
      } catch (const std::exception& e) {
        error = e.what();
        std::fprintf(stderr, "dino_rt: %s\n", e.what());
      }
      cpu_leave();
      if (error.empty())
        reply(job.seq, 0, res.data(), res.size() * 4);
      else
        reply(job.seq, 1, error.data(), error.size());
      {
        std::lock_guard<std::mutex> l(m_);
        slots_[k].busy = false;
      }
      cv_.notify_all();
    }
  }

  // Write a reply once every earlier request has been answered.
  void reply(uint64_t seq, uint32_t status, const void* data, size_t n) {
    std::unique_lock<std::mutex> l(reply_m_);
    reply_cv_.wait(l, [&] { return next_reply_ == seq; });
    const char magic[4] = {'D', 'I', 'N', 'O'};
    const uint32_t len = (uint32_t)n;
    if (!broken_) {
      try {
        write_all(out_, magic, 4);
        write_all(out_, &status, 4);
        write_all(out_, &len, 4);
        if (n) write_all(out_, data, n);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "dino_rt: %s\n", e.what());   // client gone: stop after this
        broken_ = true;
        g_stop = 1;
      }
    }
    ++next_reply_;
    reply_cv_.notify_all();
  }

  std::vector<std::unique_ptr<Detector>>& det_;
  int out_;
  std::vector<Slot> slots_;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_;
  bool quit_ = false;
  size_t last_image_slot_ = 0;
  uint64_t next_seq_ = 0;
  std::mutex reply_m_;
  std::condition_variable reply_cv_;
  uint64_t next_reply_ = 0;
  bool broken_ = false;
};

}  // namespace dino

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: dino_rt <detector bundle dir> <text bundle dir>\n");
    return 2;
  }
  std::signal(SIGTERM, [](int) { dino::g_stop = 1; });
  std::signal(SIGINT, [](int) { dino::g_stop = 1; });
  std::signal(SIGPIPE, SIG_IGN);
  // Responses go to the original stdout; anything libraries print goes to stderr.
  const int proto = ::dup(1);
  ::dup2(2, 1);
  int rc = 0;
  llima::connect_mla_rt({});
  dino::g_max_threads = omp_get_max_threads();
  try {
    std::vector<std::unique_ptr<dino::Detector>> det;
    det.push_back(std::make_unique<dino::Detector>(argv[1], argv[2], 0));
    const char* ns = std::getenv("DINO_SLOTS");
    const int slots = std::max(1, std::min(2, ns ? std::atoi(ns) : 2));
    for (int k = 1; k < slots; ++k) {
      try {
        det.push_back(std::make_unique<dino::Detector>(argv[1], argv[2], k));
      } catch (const std::exception& e) {
        std::fprintf(stderr, "dino_rt: slot %d not available (%s); one frame at a time\n", k, e.what());
        break;
      }
    }
    dino::g_multi_slot = det.size() > 1;
    const char ready[] = "DINOREADY\n";
    std::fwrite(ready, 1, sizeof(ready) - 1, stderr);
    {
      dino::Server server(det, proto);
      rc = server.run();
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dino_rt: fatal: %s\n", e.what());
    rc = 1;
  }
  if (dino::g_trace_path) {
    std::ofstream f(dino::g_trace_path);
    for (auto& e : dino::g_trace)
      f << e.slot << ' ' << e.seq << ' ' << e.wait << ' ' << e.start << ' ' << e.end << ' ' << e.what << '\n';
  }
  llima::disconnect_mla_rt();
  return rc;
}
