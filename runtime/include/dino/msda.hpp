// Multi-scale deformable attention sampling for Grounding DINO, on the CPU.
//
// The MLA stages before and after this step keep every tensor as tokens x 256
// bf16 values (HWC layout of a (1, 256, 1, T) tensor). This kernel reads the
// value map, sampling offsets and attention logits of one deformable-attention
// call straight out of the MLA output buffer and writes the sampled features,
// as bf16, straight into the next stage's input buffer.
//
// It reproduces transformers' MultiScaleDeformableAttention (grid_sample,
// bilinear, zeros padding, align_corners=False):
//   encoder (2-d refs):  pixel = ref * size + offset - 0.5
//   decoder (4-d refs):  pixel = (ref_xy + offset / P * ref_wh * 0.5) * size - 0.5
// with the attention softmax over the 16 (level, point) pairs of each head.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "dino/bf16.hpp"

namespace dino {

constexpr int kHeads = 8, kHeadDim = 32, kLevels = 4, kPoints = 4, kD = 256;

struct Level {
  int h = 0, w = 0;
  size_t start = 0;  // first token of the level
};

struct MsdaArgs {
  const uint16_t* value = nullptr;   // [N][256] bf16
  const uint16_t* offsets = nullptr;  // [Q][256] bf16: ((h*L + l)*P + p)*2 + xy
  const uint16_t* logits = nullptr;   // [Q][256] bf16: first 128 = h*16 + l*4 + p
  uint16_t* out = nullptr;           // [Q][256] bf16
  const float* ref = nullptr;        // encoder: [Q][2] (x, y); decoder: [Q][4] (cx, cy, w, h)
  int ref_dims = 2;
  int queries = 0;
  int n_levels = kLevels;            // OmDet-Turbo: 3
  Level levels[kLevels];
};

namespace detail {

inline void axpy32(float* acc, const uint16_t* v, float w) {
#if defined(__aarch64__)
  const float32x4_t wv = vdupq_n_f32(w);
  for (int d = 0; d < kHeadDim; d += 8) {
    const uint16x8_t raw = vld1q_u16(v + d);
    const float32x4_t lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(raw), 16));
    const float32x4_t hi = vreinterpretq_f32_u32(vshll_high_n_u16(raw, 16));
    vst1q_f32(acc + d, vfmaq_f32(vld1q_f32(acc + d), lo, wv));
    vst1q_f32(acc + d + 4, vfmaq_f32(vld1q_f32(acc + d + 4), hi, wv));
  }
#else
  for (int d = 0; d < kHeadDim; ++d) acc[d] += w * from_bf16(v[d]);
#endif
}

// One (query, head): softmax of the 16 logits, then 16 bilinear samples.
inline void sample_head(const MsdaArgs& a, int q, int h, float* acc) {
  const int L = a.n_levels, LP = L * kPoints;
  const uint16_t* lg = a.logits + (size_t)q * kD + h * LP;
  const uint16_t* of = a.offsets + (size_t)q * kD + h * LP * 2;
  float w[kLevels * kPoints], m = -INFINITY;
  for (int i = 0; i < LP; ++i) {
    w[i] = from_bf16(lg[i]);
    m = w[i] > m ? w[i] : m;
  }
  float sum = 0;
  for (int i = 0; i < LP; ++i) sum += (w[i] = std::exp(w[i] - m));
  const float inv = 1.0f / sum;
  const float* r = a.ref + (size_t)q * a.ref_dims;
  for (int d = 0; d < kHeadDim; ++d) acc[d] = 0;
  for (int l = 0; l < L; ++l) {
    const Level& lv = a.levels[l];
    const float W = (float)lv.w, H = (float)lv.h;
    const uint16_t* base = a.value + lv.start * kD + h * kHeadDim;
    for (int p = 0; p < kPoints; ++p) {
      const int i = l * kPoints + p;
      const float ox = from_bf16(of[2 * i]), oy = from_bf16(of[2 * i + 1]);
      float x, y;
      if (a.ref_dims == 2) {
        x = r[0] * W + ox - 0.5f;
        y = r[1] * H + oy - 0.5f;
      } else {
        x = (r[0] + ox * (0.5f / kPoints) * r[2]) * W - 0.5f;
        y = (r[1] + oy * (0.5f / kPoints) * r[3]) * H - 0.5f;
      }
      const float aw = w[i] * inv;
      const float fx0 = std::floor(x), fy0 = std::floor(y);
      const int x0 = (int)fx0, y0 = (int)fy0;
      const float fx = x - fx0, fy = y - fy0;
      const bool vx0 = x0 >= 0 && x0 < lv.w, vx1 = x0 + 1 >= 0 && x0 + 1 < lv.w;
      const bool vy0 = y0 >= 0 && y0 < lv.h, vy1 = y0 + 1 >= 0 && y0 + 1 < lv.h;
      const size_t r0 = (size_t)y0 * lv.w, r1 = r0 + lv.w;
      if (vy0 && vx0) axpy32(acc, base + (r0 + x0) * kD, (1 - fx) * (1 - fy) * aw);
      if (vy0 && vx1) axpy32(acc, base + (r0 + x0 + 1) * kD, fx * (1 - fy) * aw);
      if (vy1 && vx0) axpy32(acc, base + (r1 + x0) * kD, (1 - fx) * fy * aw);
      if (vy1 && vx1) axpy32(acc, base + (r1 + x0 + 1) * kD, fx * fy * aw);
    }
  }
}

}  // namespace detail

// Runs on the caller's OpenMP team (or serially without OpenMP).
inline void msda(const MsdaArgs& a) {
#pragma omp parallel for schedule(dynamic, 64)
  for (int q = 0; q < a.queries; ++q) {
    float acc[kHeadDim];
    uint16_t* o = a.out + (size_t)q * kD;
    for (int h = 0; h < kHeads; ++h) {
      detail::sample_head(a, q, h, acc);
      to_bf16_span(acc, o + h * kHeadDim, kHeadDim);
    }
  }
}

}  // namespace dino
