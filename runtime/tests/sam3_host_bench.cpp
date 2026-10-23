// Time the per-decoder-layer host work of sam3_rt (CPU only, random weights, no MLA).
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "dino/bf16.hpp"
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
static void linear(const float* x, int rows, int in, const float* W, const float* b, int out, float* y, bool relu = false) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + (size_t)r * in;
    float* yr = y + (size_t)r * out;
    for (int o = 0; o < out; ++o) {
      const float* w = W + (size_t)o * in;
      float s = 0.f;
#pragma omp simd reduction(+ : s)
      for (int i = 0; i < in; ++i) s += w[i] * xr[i];
      s += b[o];
      yr[o] = relu ? std::max(s, 0.f) : s;
    }
  }
}
int main() {
  const int Q = 200, D = 256, G = 48, H = 8, N = G * G;
  std::mt19937 rng(0);
  std::normal_distribution<float> nd(0, 0.05f);
  auto rnd = [&](size_t n) { std::vector<float> v(n); for (auto& x : v) x = nd(rng); return v; };
  auto W1 = rnd(256 * 512), B1 = rnd(256), W2 = rnd(256 * 256), B2 = rnd(256);           // ref head
  auto R1 = rnd(256 * 2), RB1 = rnd(256), R2 = rnd(8 * 256), RB2 = rnd(8);                // rpb mlp (one axis)
  auto H1 = rnd(256 * 256), H2 = rnd(256 * 256), H3 = rnd(4 * 256), HB = rnd(256);        // box head
  std::vector<float> ref(Q * 4, 0.3f), sine(Q * 512), hid(Q * 256), qp(Q * 256), t1(Q * 256), t2(Q * 256), dl(Q * 4);
  std::vector<uint16_t> rpb((size_t)H * (Q + 1) * N);
  std::vector<float> by(Q * G * H), bx(Q * G * H);
  double t_sine = 0, t_ref = 0, t_mlp = 0, t_exp = 0, t_box = 0;
  for (int it = 0; it < 12; ++it) {
    auto t = Clock::now();
#pragma omp parallel for
    for (int q = 0; q < Q; ++q)
      for (int i = 0; i < 512; ++i) sine[q * 512 + i] = (i & 1) ? std::cos(ref[q * 4 + i / 128] * 6.28f / std::pow(10000.f, (i % 128) / 128.f)) : std::sin(ref[q * 4 + i / 128] * 6.28f / std::pow(10000.f, (i % 128) / 128.f));
    t_sine += ms(t); t = Clock::now();
    linear(sine.data(), Q, 512, W1.data(), B1.data(), 256, hid.data(), true);
    linear(hid.data(), Q, 256, W2.data(), B2.data(), 256, qp.data());
    t_ref += ms(t); t = Clock::now();
    for (int axis = 0; axis < 2; ++axis) {
      float* out = axis ? bx.data() : by.data();
#pragma omp parallel for schedule(static)
      for (int q = 0; q < Q; ++q) {
        float h[256];
        for (int c = 0; c < G; ++c) {
          const float i0 = std::log2(std::fabs(c / 48.f - 0.3f) * 8 + 1) / 3, i1 = std::log2(std::fabs(c / 48.f - 0.6f) * 8 + 1) / 3;
          for (int j = 0; j < 256; ++j) h[j] = std::max(0.f, RB1[j] + R1[j * 2] * i0 + R1[j * 2 + 1] * i1);
          for (int hd = 0; hd < H; ++hd) {
            float s = 0;
#pragma omp simd reduction(+ : s)
            for (int j = 0; j < 256; ++j) s += R2[hd * 256 + j] * h[j];
            out[((size_t)q * G + c) * H + hd] = s + RB2[hd];
          }
        }
      }
    }
    t_mlp += ms(t); t = Clock::now();
#pragma omp parallel for collapse(2) schedule(static)
    for (int hd = 0; hd < H; ++hd)
      for (int q = 0; q <= Q; ++q) {
        uint16_t* row = rpb.data() + ((size_t)hd * (Q + 1) + q) * N;
        if (q == 0) { std::memset(row, 0, N * 2); continue; }
        for (int y = 0; y < G; ++y) {
          const float vy = by[((size_t)(q - 1) * G + y) * H + hd];
          for (int x = 0; x < G; ++x) row[y * G + x] = dino::to_bf16(vy + bx[((size_t)(q - 1) * G + x) * H + hd]);
        }
      }
    t_exp += ms(t); t = Clock::now();
    linear(qp.data(), Q, 256, H1.data(), HB.data(), 256, t1.data(), true);
    linear(t1.data(), Q, 256, H2.data(), HB.data(), 256, t2.data(), true);
    linear(t2.data(), Q, 256, H3.data(), HB.data(), 4, dl.data());
    t_box += ms(t);
  }
  const double n = 12;
  std::printf("per decoder layer (ms, %d threads): sine %.2f  ref-head MLP %.2f  box-bias MLPs %.2f  box-bias expand %.2f  box head %.2f  total %.2f\n",
              omp_get_max_threads(), t_sine / n, t_ref / n, t_mlp / n, t_exp / n, t_box / n, (t_sine + t_ref + t_mlp + t_exp + t_box) / n);
}
