// bf16 <-> float conversion (round to nearest even), scalar and NEON.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace dino {

inline uint16_t to_bf16(float v) {
  uint32_t b;
  std::memcpy(&b, &v, 4);
  if ((b & 0x7f800000u) == 0x7f800000u) return (uint16_t)((b >> 16) | ((b & 0xffff) ? 0x40 : 0));  // inf/nan
  return (uint16_t)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}

inline float from_bf16(uint16_t v) {
  uint32_t b = (uint32_t)v << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

inline void to_bf16_span(const float* src, uint16_t* dst, size_t n) {
  size_t i = 0;
#if defined(__aarch64__)
  const uint32x4_t bias = vdupq_n_u32(0x7fff), one = vdupq_n_u32(1);
  const size_t nv = n & ~(size_t)7;
  for (; i < nv; i += 8) {
    uint32x4_t a = vreinterpretq_u32_f32(vld1q_f32(src + i));
    uint32x4_t b = vreinterpretq_u32_f32(vld1q_f32(src + i + 4));
    a = vaddq_u32(a, vaddq_u32(bias, vandq_u32(vshrq_n_u32(a, 16), one)));
    b = vaddq_u32(b, vaddq_u32(bias, vandq_u32(vshrq_n_u32(b, 16), one)));
    vst1q_u16(dst + i, vcombine_u16(vshrn_n_u32(a, 16), vshrn_n_u32(b, 16)));
  }
#endif
  for (; i < n; ++i) dst[i] = to_bf16(src[i]);
}

inline void from_bf16_span(const uint16_t* src, float* dst, size_t n) {
  size_t i = 0;
#if defined(__aarch64__)
  const size_t nv = n & ~(size_t)7;
  for (; i < nv; i += 8) {
    const uint16x8_t a = vld1q_u16(src + i);
    vst1q_u32((uint32_t*)(dst + i), vshll_n_u16(vget_low_u16(a), 16));
    vst1q_u32((uint32_t*)(dst + i + 4), vshll_high_n_u16(a, 16));
  }
#endif
  for (; i < n; ++i) dst[i] = from_bf16(src[i]);
}

}  // namespace dino
