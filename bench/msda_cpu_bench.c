// Benchmark: multi-scale deformable attention sampling on the A65 cores.
// value: per level (H, W, heads, 32) fp32 ; loc: (Q, heads, L, P, 2) pixel coords
// (align_corners=False: pixel centre at i+0.5) ; aw: (Q, heads, L, P) ; out: (Q, heads, 32)
// gcc -O3 -fopenmp -march=armv8.2-a+fp16 msda_cpu_bench.c -o msda_cpu_bench
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

static void msda(int Q, const int *Hs, const int *Ws, float *const *val, const float *loc,
                 const float *aw, float *out) {
#pragma omp parallel for schedule(static)
  for (int q = 0; q < Q; q++) {
    for (int h = 0; h < NH; h++) {
      float acc[D];
      memset(acc, 0, sizeof acc);
      for (int l = 0; l < NL; l++) {
        const int H = Hs[l], W = Ws[l];
        const float *v = val[l];
        for (int p = 0; p < NP; p++) {
          const size_t k = (((size_t)q * NH + h) * NL + l) * NP + p;
          const float x = loc[2 * k] - 0.5f, y = loc[2 * k + 1] - 0.5f, a = aw[k];
          const int x0 = (int)floorf(x), y0 = (int)floorf(y);
          const float fx = x - x0, fy = y - y0;
          const float w[4] = {(1 - fx) * (1 - fy) * a, fx * (1 - fy) * a, (1 - fx) * fy * a, fx * fy * a};
          const int xs[4] = {x0, x0 + 1, x0, x0 + 1}, ys[4] = {y0, y0, y0 + 1, y0 + 1};
          for (int c = 0; c < 4; c++) {
            if (xs[c] < 0 || ys[c] < 0 || xs[c] >= W || ys[c] >= H) continue;
            const float *src = v + (((size_t)ys[c] * W + xs[c]) * NH + h) * D;
            const float wc = w[c];
            for (int d = 0; d < D; d++) acc[d] += wc * src[d];
          }
        }
      }
      memcpy(out + ((size_t)q * NH + h) * D, acc, sizeof acc);
    }
  }
}

int main(int argc, char **argv) {
  int S = argc > 1 ? atoi(argv[1]) : 768;
  int Q = argc > 2 ? atoi(argv[2]) : -1;  // -1: encoder (all tokens)
  int Hs[NL], Ws[NL], N = 0;
  for (int l = 0; l < NL; l++) {
    Hs[l] = Ws[l] = (S / 8 + (1 << l) - 1) >> l;
    N += Hs[l] * Ws[l];
  }
  if (Q < 0) Q = N;
  float *val[NL];
  srand(1);
  for (int l = 0; l < NL; l++) {
    size_t n = (size_t)Hs[l] * Ws[l] * NH * D;
    val[l] = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) val[l][i] = rand() / (float)RAND_MAX;
  }
  size_t ns = (size_t)Q * NH * NL * NP;
  float *loc = malloc(ns * 2 * sizeof(float)), *aw = malloc(ns * sizeof(float));
  float *out = malloc((size_t)Q * NH * D * sizeof(float));
  for (size_t k = 0; k < ns; k++) {
    int l = (k / NP) % NL;
    loc[2 * k] = rand() / (float)RAND_MAX * Ws[l];
    loc[2 * k + 1] = rand() / (float)RAND_MAX * Hs[l];
    aw[k] = 1.0f / (NL * NP);
  }
  int threads[] = {1, 4, 8, 16};
  for (int t = 0; t < 4; t++) {
    omp_set_num_threads(threads[t]);
    msda(Q, Hs, Ws, val, loc, aw, out);
    double best = 1e9;
    for (int r = 0; r < 5; r++) {
      double t0 = now();
      msda(Q, Hs, Ws, val, loc, aw, out);
      double dt = now() - t0;
      if (dt < best) best = dt;
    }
    printf("S=%d Q=%d tokens=%d threads=%2d  %.2f ms\n", S, Q, N, threads[t], best * 1e3);
  }
  return 0;
}
