// Benchmark the runtime's msda() (bf16 in/out) on synthetic data with realistic locality.
//   msda_bench <S> [dec]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "dino/msda.hpp"

using namespace dino;

int main(int argc, char** argv) {
  const int S = argc > 1 ? atoi(argv[1]) : 576;
  const bool dec = argc > 2;
  Level lv[kLevels];
  size_t N = 0;
  for (int l = 0; l < kLevels; ++l) {
    lv[l].h = lv[l].w = (S / 8 + (1 << l) - 1) >> l;
    lv[l].start = N;
    N += (size_t)lv[l].h * lv[l].w;
  }
  const int Q = dec ? 900 : (int)N;
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0, 3.0f);
  std::uniform_real_distribution<float> ud(0, 1);
  std::vector<uint16_t> val(N * kD), off((size_t)Q * kD), lg((size_t)Q * kD), out((size_t)Q * kD);
  for (auto& v : val) v = to_bf16(ud(rng));
  std::vector<float> ref((size_t)Q * (dec ? 4 : 2));
  for (int q = 0; q < Q; ++q) {
    if (dec) {
      for (int i = 0; i < 4; ++i) ref[q * 4 + i] = i < 2 ? ud(rng) : 0.05f + 0.3f * ud(rng);
    } else {
      int l = kLevels - 1;
      while (l > 0 && (size_t)q < lv[l].start) --l;
      const int i = q - (int)lv[l].start;
      ref[q * 2] = ((i % lv[l].w) + 0.5f) / lv[l].w;
      ref[q * 2 + 1] = ((i / lv[l].w) + 0.5f) / lv[l].h;
    }
    for (int c = 0; c < kD; ++c) {
      const int l = (c / 8) % 4;
      off[(size_t)q * kD + c] = to_bf16(l < 2 ? nd(rng) : (ud(rng) - 0.5f) * lv[l].w);
      lg[(size_t)q * kD + c] = to_bf16(nd(rng) * 0.3f);
    }
  }
  MsdaArgs a;
  a.value = val.data(); a.offsets = off.data(); a.logits = lg.data(); a.out = out.data();
  a.ref = ref.data(); a.ref_dims = dec ? 4 : 2; a.queries = Q;
  for (int l = 0; l < kLevels; ++l) a.levels[l] = lv[l];
  for (int variant = 0; variant < 1; ++variant) {
    auto fn = msda;
    fn(a);
    double best = 1e9;
    for (int r = 0; r < 9; ++r) {
      auto t = std::chrono::steady_clock::now();
      fn(a);
      best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count());
    }
    double cs = 0;
    for (auto v : out) cs += from_bf16(v);
    std::printf("S=%d Q=%d %s %s: %.2f ms (checksum %.4f)\n", S, Q, dec ? "decoder" : "encoder",
                "", best, cs);
  }
}
