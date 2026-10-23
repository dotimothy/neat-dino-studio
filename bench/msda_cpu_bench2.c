// Multi-scale deformable attention sampling on the A65 cores, optimised variants.
// value per level: (H, W, heads, 32) as fp32 or fp16 ; loc in pixel units (align_corners=False)
// Realistic locations: fine levels near the query's own position (sigma px), coarse levels anywhere.
// gcc -O3 -fopenmp -march=armv8.2-a+fp16 msda_cpu_bench2.c -o msda_cpu_bench2 -lm
#include <arm_neon.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NH 8
#define D 32
#define NL 4
#define NP 4

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static inline void axpy32_f32(float *acc, const float *src, float w) {
  float32x4_t wv = vdupq_n_f32(w);
  for (int d = 0; d < D; d += 4) vst1q_f32(acc + d, vfmaq_f32(vld1q_f32(acc + d), vld1q_f32(src + d), wv));
}

static inline void axpy32_f16(float *acc, const __fp16 *src, float w) {
  float32x4_t wv = vdupq_n_f32(w);
  for (int d = 0; d < D; d += 8) {
    float16x8_t s = vld1q_f16(src + d);
    vst1q_f32(acc + d, vfmaq_f32(vld1q_f32(acc + d), vcvt_f32_f16(vget_low_f16(s)), wv));
    vst1q_f32(acc + d + 4, vfmaq_f32(vld1q_f32(acc + d + 4), vcvt_f32_f16(vget_high_f16(s)), wv));
  }
}

// One query/head: 16 samples, 4 corners each.
#define BODY(AXPY, T)                                                                          \
  for (int l = 0; l < NL; l++) {                                                               \
    const int H = Hs[l], W = Ws[l];                                                            \
    const T *v = (const T *)val[l];                                                            \
    for (int p = 0; p < NP; p++) {                                                             \
      const size_t k = (((size_t)q * NH + h) * NL + l) * NP + p;                               \
      const float x = loc[2 * k] - 0.5f, y = loc[2 * k + 1] - 0.5f, a = aw[k];                 \
      const int x0 = (int)floorf(x), y0 = (int)floorf(y);                                      \
      const float fx = x - x0, fy = y - y0;                                                    \
      const int vx0 = x0 >= 0 && x0 < W, vx1 = x0 + 1 >= 0 && x0 + 1 < W;                      \
      const int vy0 = y0 >= 0 && y0 < H, vy1 = y0 + 1 >= 0 && y0 + 1 < H;                      \
      const size_t row0 = (size_t)y0 * W, row1 = row0 + W;                                     \
      if (vy0 && vx0) AXPY(acc, v + ((row0 + x0) * NH + h) * D, (1 - fx) * (1 - fy) * a);      \
      if (vy0 && vx1) AXPY(acc, v + ((row0 + x0 + 1) * NH + h) * D, fx * (1 - fy) * a);        \
      if (vy1 && vx0) AXPY(acc, v + ((row1 + x0) * NH + h) * D, (1 - fx) * fy * a);            \
      if (vy1 && vx1) AXPY(acc, v + ((row1 + x0 + 1) * NH + h) * D, fx * fy * a);              \
    }                                                                                          \
  }

static void msda_f32(int Q, const int *Hs, const int *Ws, void *const *val, const float *loc,
                     const float *aw, float *out) {
#pragma omp parallel for schedule(static)
  for (int q = 0; q < Q; q++)
    for (int h = 0; h < NH; h++) {
      float acc[D] = {0};
      BODY(axpy32_f32, float)
      memcpy(out + ((size_t)q * NH + h) * D, acc, sizeof acc);
    }
}

static void msda_f16(int Q, const int *Hs, const int *Ws, void *const *val, const float *loc,
                     const float *aw, float *out) {
#pragma omp parallel for schedule(static)
  for (int q = 0; q < Q; q++)
    for (int h = 0; h < NH; h++) {
      float acc[D] = {0};
      BODY(axpy32_f16, __fp16)
      memcpy(out + ((size_t)q * NH + h) * D, acc, sizeof acc);
    }
}

static float randn(void) {
  float u = (rand() + 1.0f) / (RAND_MAX + 2.0f), v = (rand() + 1.0f) / (RAND_MAX + 2.0f);
  return sqrtf(-2 * logf(u)) * cosf(6.2831853f * v);
}

int main(int argc, char **argv) {
  int S = argc > 1 ? atoi(argv[1]) : 768;
  int dec = argc > 2 ? atoi(argv[2]) : 0;  // 1: 900 decoder queries
  float sigma = argc > 3 ? atof(argv[3]) : 4.0f;
  int Hs[NL], Ws[NL], N = 0, starts[NL];
  for (int l = 0; l < NL; l++) {
    Hs[l] = Ws[l] = (S / 8 + (1 << l) - 1) >> l;
    starts[l] = N;
    N += Hs[l] * Ws[l];
  }
  int Q = dec ? 900 : N;
  void *v32[NL], *v16[NL];
  srand(1);
  for (int l = 0; l < NL; l++) {
    size_t n = (size_t)Hs[l] * Ws[l] * NH * D;
    float *a = malloc(n * sizeof(float));
    __fp16 *b = malloc(n * sizeof(__fp16));
    for (size_t i = 0; i < n; i++) b[i] = (__fp16)(a[i] = rand() / (float)RAND_MAX);
    v32[l] = a;
    v16[l] = b;
  }
  size_t ns = (size_t)Q * NH * NL * NP;
  float *loc = malloc(ns * 2 * sizeof(float)), *aw = malloc(ns * sizeof(float));
  float *out = malloc((size_t)Q * NH * D * sizeof(float));
  for (int q = 0; q < Q; q++) {
    // query's normalised position: encoder = its own token cell; decoder = random box centre
    float rx, ry;
    if (dec) {
      rx = rand() / (float)RAND_MAX;
      ry = rand() / (float)RAND_MAX;
    } else {
      int l = NL - 1;
      while (l > 0 && q < starts[l]) l--;
      int i = q - starts[l];
      rx = ((i % Ws[l]) + 0.5f) / Ws[l];
      ry = ((i / Ws[l]) + 0.5f) / Hs[l];
    }
    for (int h = 0; h < NH; h++)
      for (int l = 0; l < NL; l++)
        for (int p = 0; p < NP; p++) {
          size_t k = (((size_t)q * NH + h) * NL + l) * NP + p;
          float s = l < 2 ? sigma : Ws[l] * 0.5f;
          loc[2 * k] = rx * Ws[l] + randn() * s;
          loc[2 * k + 1] = ry * Hs[l] + randn() * s;
          aw[k] = 1.0f / (NL * NP);
        }
  }
  int threads[] = {8, 16};
  for (int t = 0; t < 2; t++) {
    omp_set_num_threads(threads[t]);
    for (int f16 = 0; f16 < 2; f16++) {
      void (*fn)(int, const int *, const int *, void *const *, const float *, const float *, float *) =
          f16 ? msda_f16 : msda_f32;
      void *const *val = f16 ? v16 : v32;
      fn(Q, Hs, Ws, val, loc, aw, out);
      double best = 1e9;
      for (int r = 0; r < 7; r++) {
        double t0 = now();
        fn(Q, Hs, Ws, val, loc, aw, out);
        double dt = now() - t0;
        if (dt < best) best = dt;
      }
      printf("S=%d Q=%5d sigma=%.0f threads=%2d %s  %.2f ms\n", S, Q, sigma, threads[t], f16 ? "fp16" : "fp32",
             best * 1e3);
    }
  }
  return 0;
}
