// esam_rt: EfficientSAM (yformer/EfficientSAM, Ti or S) on the Modalix MLA, for Grounded EfficientSAM:
// Grounding DINO (dino_rt) finds the boxes, this turns each box into a mask.
//
// Programs (model/esam_export.py), run with SimaLMM's MLAModelWithBuffer:
//   enc      host patch tokens (1, 768, 1, G*G) -> ViT (12 global layers) + neck 1x1 conv + LN
//   neck     the same bytes as (1, 256, G, G) -> neck 3x3 conv + LN = image embedding
//   dec<B>   image embedding + B boxes' tokens (1, 256, B, 16) -> (1, 48, B, G*G + 1):
//            3 mask logit maps at 4G x 4G as 16 sub-pixel groups x 3 masks on C, plus the 3 IoU
//            scores in the last column. Several box counts can be compiled; the runtime times each
//            at load and covers n boxes with the cheapest mix.
// The decoder tokens per box are the IoU + 4 mask tokens and the 4 padding-point embeddings (constants
// from host.f32), the two corner embeddings and 5 zero tokens the ELF masks out.
// The CPU does the patchify, the prompt embedding of the box corners (random-Fourier PE), the
// pick of the best of the 3 masks by predicted IoU, and a 2x bilinear upsample before the threshold.
//
// Protocol (stdin/stdout, little endian), one request at a time:
//   request : "ESAM" u32 cmd u32 nbytes payload
//   response: "ESAM" u32 status(0 ok) u32 nbytes payload
//   cmd 1 INFO     -> JSON
//   cmd 2 ENCODE   payload u8 rgb[S][S][3] (the image squashed to S x S) -> f32 timings[8]
//   cmd 3 MASKS    payload u32 n, f32 boxes[n][4] (x0, y0, x1, y1 in S x S pixels)
//                  -> u32 M, n x (f32 iou, u8 mask[M][M] (logit > 0)), f32 timings[8]
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
#include <tuple>
#include <unistd.h>
#include <vector>

#include "dino/bf16.hpp"

namespace fs = std::filesystem;
namespace llima = simaai::llima;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using dino::from_bf16;
using dino::to_bf16;

namespace esam {

constexpr int D = 256, P = 16, PC = 3 * P * P, NM = 3, OC = 48, TOK = 16;   // tokens per box: 5 + 2 + 4 + 5 masked

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
  std::unique_ptr<llima::MLAModelWithBuffer> model;
  double last_ms = 0;
  void run() {
    auto t = Clock::now();
    model->run();
    last_ms = ms_since(t);
  }
};

struct Decoder {
  int B;
  std::unique_ptr<Buffer> box, out;
  Stage stage;
  double ms = 0;   // measured at load
};

class Model {
 public:
  explicit Model(const fs::path& dir) {
    const json m = json::parse(read_file(dir / "manifest.json"));
    S_ = m.at("size");
    G_ = S_ / P;
    N_ = G_ * G_;
    M_ = 4 * G_;
    up_ = m.value("mask_upsample", 2);
    variant_ = m.value("variant", "");
    auto raw = read_file(dir / "host.f32");
    host_.resize(raw.size() / 4);
    std::memcpy(host_.data(), raw.data(), raw.size());
    for (auto& [k, v] : m.at("host").items()) index_[k] = v.at("offset").get<size_t>();

    for (const auto& [k, dst, n] : {std::tuple{"tok_head", &tok_head_, 5}, std::tuple{"tok_tail", &tok_tail_, 4}}) {
      dst->resize((size_t)n * D);
      dino::to_bf16_span(w(k), dst->data(), dst->size());
    }
    tokens_ = std::make_unique<Buffer>("tokens", (size_t)N_ * PC);
    mid_ = std::make_unique<Buffer>("mid", (size_t)N_ * D);
    emb_ = std::make_unique<Buffer>("emb", (size_t)N_ * D);
    const json order = m.value("inputs", json::object());
    const auto elf = [&](const std::string& n) { return (dir / "elf" / (n + ".elf")).string(); };
    const auto t0 = Clock::now();
    auto load = [&](Stage& s, const std::string& n, std::vector<llima::MLABufferSlice> in,
                    std::vector<llima::MLABufferSlice> out) {
      s.model = std::make_unique<llima::MLAModelWithBuffer>(elf(n), in, out);
      std::fprintf(stderr, "ESAMLOAD %s\n", n.c_str());
      s.model->load();
    };
    load(enc_, "enc", {tokens_->all()}, {mid_->all()});
    load(neck_, "neck", {mid_->all()}, {emb_->all()});
    for (int B : m.at("decoders").get<std::vector<int>>()) {
      auto d = std::make_unique<Decoder>();
      d->B = B;
      d->box = std::make_unique<Buffer>("tok", (size_t)B * TOK * D);
      d->out = std::make_unique<Buffer>("dec_out", (size_t)B * (N_ + 1) * OC);
      const std::string n = "dec" + std::to_string(B);
      std::map<std::string, llima::MLABufferSlice> named{{"img", emb_->all()}, {"tok", d->box->all()}};
      std::vector<llima::MLABufferSlice> in;
      for (const auto& nm : order.at(n)) in.push_back(named.at(nm.get<std::string>()));
      load(d->stage, n, in, {d->out->all()});
      decs_.push_back(std::move(d));
    }
    std::sort(decs_.begin(), decs_.end(), [](auto& a, auto& b) { return a->B < b->B; });
    load_s_ = ms_since(t0) / 1000.0;
    for (auto& d : decs_) {   // time each decoder (second run, after the first one warmed it)
      for (int i = 0; i < 2; ++i) d->stage.run();
      d->ms = d->stage.last_ms;
      std::fprintf(stderr, "esam_rt: dec%d %.2f ms\n", d->B, d->ms);
    }
    std::fprintf(stderr, "esam_rt: %s S=%d, %zu programs loaded in %.1f s\n", variant_.c_str(), S_, decs_.size() + 2,
                 load_s_);
  }

  ~Model() {
    enc_.model->free();
    neck_.model->free();
    for (auto& d : decs_) d->stage.model->free();
  }

  json info() const {
    json d = json::array();
    for (auto& x : decs_) d.push_back({{"boxes", x->B}, {"ms", x->ms}});
    return {{"variant", variant_}, {"size", S_}, {"grid", G_}, {"mask_size", M_ * up_}, {"decoders", d},
            {"load_seconds", load_s_}};
  }

  std::vector<float> encode(const uint8_t* rgb) {
    std::vector<float> t(8, 0.f);
    auto t0 = Clock::now();
    uint16_t* dst = tokens_->p();
    // token (gy, gx), channel (c, py, px): the patch conv's weight order (normalisation is in the ELF)
#pragma omp parallel for schedule(static)
    for (int tk = 0; tk < N_; ++tk) {
      const int gy = tk / G_, gx = tk % G_;
      uint16_t* o = dst + (size_t)tk * PC;
      for (int c = 0; c < 3; ++c)
        for (int py = 0; py < P; ++py) {
          const uint8_t* row = rgb + ((size_t)(gy * P + py) * S_ + gx * P) * 3 + c;
          for (int px = 0; px < P; ++px) o[(c * P + py) * P + px] = to_bf16(row[px * 3] * (1.f / 255.f));
        }
    }
    tokens_->flush();
    t[0] = ms_since(t0);
    enc_.run();
    t[1] = enc_.last_ms;
    neck_.run();
    t[2] = neck_.last_ms;
    t[7] = ms_since(t0);
    encoded_ = true;
    return t;
  }

  std::vector<char> masks(const float* boxes, int n) {
    if (!encoded_) throw std::runtime_error("MASKS before ENCODE");
    std::vector<float> t(8, 0.f);
    const auto t0 = Clock::now();
    const int Mo = M_ * up_;
    std::vector<char> out(4 + (size_t)n * (4 + (size_t)Mo * Mo) + 8 * 4);
    uint32_t mo = Mo;
    std::memcpy(out.data(), &mo, 4);
    // cover n boxes with the cheapest mix of compiled box counts
    std::vector<double> best(n + 1, 1e30);
    std::vector<int> pick(n + 1, -1);
    best[0] = 0;
    for (int k = 1; k <= n; ++k)
      for (size_t i = 0; i < decs_.size(); ++i) {
        const double c = decs_[i]->ms + best[std::max(0, k - decs_[i]->B)];
        if (c < best[k]) best[k] = c, pick[k] = (int)i;
      }
    std::vector<float> logit((size_t)M_ * M_);
    int done = 0, k = n;
    double dec_ms = 0, post_ms = 0;
    while (k > 0) {
      Decoder& d = *decs_[pick[k]];
      const int cnt = std::min(d.B, k);
      // box corner prompt embeddings; unused slots repeat the first box
      uint16_t* bx = d.box->p();
      for (int s = 0; s < d.B; ++s) {
        const float* b = boxes + 4 * (done + std::min(s, cnt - 1));
        uint16_t* tk = bx + (size_t)s * TOK * D;
        std::memcpy(tk, tok_head_.data(), 5 * D * 2);
        for (int corner = 0; corner < 2; ++corner) {
          const float x = std::max(0.f, b[2 * corner]), y = std::max(0.f, b[2 * corner + 1]);
          point_embedding(x, y, corner, tk + (size_t)(5 + corner) * D);
        }
        std::memcpy(tk + 7 * D, tok_tail_.data(), 4 * D * 2);   // tokens 11..15 stay zero
      }
      d.box->flush();
      d.stage.run();
      dec_ms += d.stage.last_ms;
      const auto tp = Clock::now();
      d.out->invalidate();
      // one sequential copy out of the device buffer (strided reads of it are slow)
      const size_t row_n = (size_t)(N_ + 1) * OC;
      host_out_.resize(row_n * cnt);
      std::memcpy(host_out_.data(), d.out->p(), host_out_.size() * 2);
      const uint16_t* o = host_out_.data();
      for (int s = 0; s < cnt; ++s) {
        const uint16_t* row = o + (size_t)s * row_n;                          // [w][c] for this box
        const uint16_t* iou = row + (size_t)N_ * OC;
        int bm = 0;
        for (int i = 1; i < NM; ++i)
          if (from_bf16(iou[i]) > from_bf16(iou[bm])) bm = i;
        const float score = from_bf16(iou[bm]);
        // un-shuffle: channel ((a*2+b)*4 + a'*2+b')*3 + m at token (i, j) -> pixel (4i+2a+a', 4j+2b+b')
#pragma omp parallel for schedule(static)
        for (int tk = 0; tk < N_; ++tk) {
          const int gi = tk / G_, gj = tk % G_;
          const uint16_t* c = row + (size_t)tk * OC;
          for (int sub = 0; sub < 16; ++sub) {
            const int a = sub >> 3, bb = (sub >> 2) & 1, a2 = (sub >> 1) & 1, b2 = sub & 1;
            logit[(size_t)(4 * gi + 2 * a + a2) * M_ + 4 * gj + 2 * bb + b2] = from_bf16(c[sub * NM + bm]);
          }
        }
        char* dst = out.data() + 4 + (size_t)(done + s) * (4 + (size_t)Mo * Mo);
        std::memcpy(dst, &score, 4);
        upsample_threshold(logit.data(), reinterpret_cast<uint8_t*>(dst + 4));
      }
      post_ms += ms_since(tp);
      done += cnt;
      k -= cnt;
    }
    t[3] = dec_ms;
    t[4] = post_ms;
    t[7] = ms_since(t0);
    std::memcpy(out.data() + out.size() - 32, t.data(), 32);
    return out;
  }

 private:
  const float* w(const std::string& k) const { return host_.data() + index_.at(k); }

  // PromptEncoder._embed_points for one corner: random-Fourier PE of the pixel centre + corner embedding
  void point_embedding(float x, float y, int corner, uint16_t* dst) const {
    const float* g = w("pe_gauss");                                       // (2, 128)
    const float* e = w(corner == 0 ? "corner_tl" : "corner_br");          // (256)
    const float cx = 2.f * ((x + 0.5f) / S_) - 1.f, cy = 2.f * ((y + 0.5f) / S_) - 1.f;
    for (int k = 0; k < D / 2; ++k) {
      const float f = 2.f * (float)M_PI * (cx * g[k] + cy * g[D / 2 + k]);
      dst[k] = to_bf16(std::sin(f) + e[k]);
      dst[D / 2 + k] = to_bf16(std::cos(f) + e[D / 2 + k]);
    }
  }

  // bilinear (align_corners = False) M x M -> (up*M)^2, then logit > 0
  void upsample_threshold(const float* l, uint8_t* dst) const {
    const int Mo = M_ * up_;
    const float inv = 1.f / up_;
#pragma omp parallel for schedule(static)
    for (int y = 0; y < Mo; ++y) {
      const float sy = std::clamp((y + 0.5f) * inv - 0.5f, 0.f, (float)(M_ - 1));
      const int y0 = (int)sy, y1 = std::min(y0 + 1, M_ - 1);
      const float fy = sy - y0;
      for (int x = 0; x < Mo; ++x) {
        const float sx = std::clamp((x + 0.5f) * inv - 0.5f, 0.f, (float)(M_ - 1));
        const int x0 = (int)sx, x1 = std::min(x0 + 1, M_ - 1);
        const float fx = sx - x0;
        const float v = (1 - fy) * ((1 - fx) * l[y0 * M_ + x0] + fx * l[y0 * M_ + x1]) +
                        fy * ((1 - fx) * l[y1 * M_ + x0] + fx * l[y1 * M_ + x1]);
        dst[(size_t)y * Mo + x] = v > 0.f;
      }
    }
  }

  int S_ = 0, G_ = 0, N_ = 0, M_ = 0, up_ = 2;
  std::string variant_;
  double load_s_ = 0;
  bool encoded_ = false;
  std::vector<float> host_;
  std::vector<uint16_t> tok_head_, tok_tail_, host_out_;
  std::map<std::string, size_t> index_;
  std::unique_ptr<Buffer> tokens_, mid_, emb_;
  Stage enc_, neck_;
  std::vector<std::unique_ptr<Decoder>> decs_;
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
  write_all(fd, "ESAM", 4);
  write_all(fd, &status, 4);
  write_all(fd, &n, 4);
  if (n) write_all(fd, payload.data(), n);
}

}  // namespace esam

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: esam_rt <bundle dir>\n");
    return 2;
  }
  std::signal(SIGTERM, [](int) { esam::g_stop = 1; });
  std::signal(SIGINT, [](int) { esam::g_stop = 1; });
  std::signal(SIGPIPE, SIG_IGN);
  const int proto = ::dup(1);
  ::dup2(2, 1);
  int rc = 0;
  llima::connect_mla_rt({});
  try {
    esam::Model model(argv[1]);
    std::fwrite("ESAMREADY\n", 1, 10, stderr);
    std::fflush(stderr);
    while (!esam::g_stop) {
      char magic[4];
      uint32_t cmd = 0, n = 0;
      if (!esam::read_all(0, magic, 4) || !esam::read_all(0, &cmd, 4) || !esam::read_all(0, &n, 4)) break;
      if (std::memcmp(magic, "ESAM", 4) != 0) throw std::runtime_error("bad request header");
      std::vector<char> in(n);
      if (n && !esam::read_all(0, in.data(), n)) break;
      try {
        std::vector<char> out;
        if (cmd == 1) {
          const std::string s = model.info().dump();
          out.assign(s.begin(), s.end());
        } else if (cmd == 2) {
          const auto t = model.encode(reinterpret_cast<const uint8_t*>(in.data()));
          out.resize(t.size() * 4);
          std::memcpy(out.data(), t.data(), out.size());
        } else if (cmd == 3) {
          uint32_t nb;
          std::memcpy(&nb, in.data(), 4);
          if (n != 4 + nb * 16) throw std::runtime_error("MASKS: bad payload size");
          out = model.masks(reinterpret_cast<const float*>(in.data() + 4), (int)nb);
        } else if (cmd == 4) {
          esam::reply(proto, 0, {});
          break;
        } else {
          throw std::runtime_error("unknown command " + std::to_string(cmd));
        }
        esam::reply(proto, 0, out);
      } catch (const std::exception& e) {
        const std::string s = e.what();
        esam::reply(proto, 1, std::vector<char>(s.begin(), s.end()));
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "esam_rt: fatal: %s\n", e.what());
    rc = 1;
  }
  llima::disconnect_mla_rt();
  return rc;
}
