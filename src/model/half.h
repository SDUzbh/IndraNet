// IEEE half precision (fp16) and bfloat16 (bf16) <-> float32 conversion, scalar and in bulk. Internal.
#pragma once

#include <cstdint>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tfi {
namespace detail {

inline float half_to_float(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF, bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {  // subnormal
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                --exp;
            }
            bits = sign | (exp << 23) | ((mant & 0x3FF) << 13);
        }
    } else if (exp == 31) bits = sign | 0x7F800000 | (mant << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Round to nearest, ties to even (as the hardware conversion); beyond fp16's range gives infinity.
inline uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000, ax = x & 0x7FFFFFFF;
    if (ax >= 0x7F800000) return static_cast<uint16_t>(sign | 0x7C00 | (ax > 0x7F800000 ? 0x200 : 0));  // inf, NaN
    if (ax >= 0x477FF000) return static_cast<uint16_t>(sign | 0x7C00);  // >= 65520 rounds to infinity
    if (ax < 0x38800000) {                                               // below 2^-14: subnormal or zero
        if (ax < 0x33000000) return static_cast<uint16_t>(sign);         // below 2^-25 rounds to zero
        const uint32_t e = ax >> 23, m = (ax & 0x7FFFFF) | 0x800000, shift = 126 - e;
        uint32_t r = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (r & 1))) ++r;
        return static_cast<uint16_t>(sign | r);
    }
    uint32_t r = (ax - 0x38000000) >> 13;  // rebias the exponent from 127 to 15
    const uint32_t rem = ax & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) ++r;  // a carry into the exponent is still right
    return static_cast<uint16_t>(sign | r);
}

// Whether v survives the round trip through fp16 without overflowing: finite and |v| <= 65504.
inline bool fits_half(float v) { return v >= -65504.f && v <= 65504.f; }

// Whether every fp16 value is finite (no infinity or NaN: exponent bits not all set).
inline bool all_finite_halves(const uint16_t* h, int64_t n) {
    uint32_t bad = 0;
    for (int64_t i = 0; i < n; ++i) bad |= (h[i] & 0x7C00) == 0x7C00;
    return bad == 0;
}

inline void halves_to_floats(const uint16_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if defined(__aarch64__)
    for (; i + 8 <= n; i += 8) {
        const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(src + i));
        vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst + i + 4, vcvt_high_f32_f16(h));
    }
#endif
    for (; i < n; ++i) dst[i] = half_to_float(src[i]);
}

// Returns how many values did not fit fp16 (fits_half); those become infinity or NaN.
inline int64_t floats_to_halves(const float* src, uint16_t* dst, int64_t n) {
    int64_t bad = 0, i = 0;
#if defined(__aarch64__)
    for (; i + 8 <= n; i += 8) {
        const float32x4_t a = vld1q_f32(src + i), b = vld1q_f32(src + i + 4);
        const float16x8_t h = vcvt_high_f16_f32(vcvt_f16_f32(a), b);
        vst1q_u16(dst + i, vreinterpretq_u16_f16(h));
        const float32x4_t lim = vdupq_n_f32(65504.f);
        // |v| <= 65504 is false for NaN too
        const uint32x4_t oka = vcleq_f32(vabsq_f32(a), lim), okb = vcleq_f32(vabsq_f32(b), lim);
        bad += 8 - static_cast<int64_t>(vaddvq_u32(vshrq_n_u32(oka, 31)) + vaddvq_u32(vshrq_n_u32(okb, 31)));
    }
#endif
    for (; i < n; ++i) {
        bad += !fits_half(src[i]);
        dst[i] = float_to_half(src[i]);
    }
    return bad;
}

// bfloat16 (bf16): the top 16 bits of a float32 (sign, 8 exponent bits, 7 fraction bits), so it widens
// to float32 exactly by a shift; float32's range, fewer significant bits than fp16 (8 against 11).
inline float bf16_to_float(uint16_t h) {
    const uint32_t bits = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Round to nearest, ties to even; NaN stays NaN (quiet). A finite value of at least 0x7F7F8000 in magnitude
// (just under float32's largest) rounds to infinity.
inline uint16_t float_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    if ((x & 0x7FFFFFFF) > 0x7F800000) return static_cast<uint16_t>((x >> 16) | 0x40);
    return static_cast<uint16_t>((x + 0x7FFF + ((x >> 16) & 1)) >> 16);
}

// Whether v rounds to a finite bf16 value.
inline bool fits_bf16(float v) {
    uint32_t x;
    std::memcpy(&x, &v, 4);
    return (x & 0x7FFFFFFF) < 0x7F7F8000;
}

// Whether every bf16 value is finite (exponent bits not all set).
inline bool all_finite_bf16(const uint16_t* h, int64_t n) {
    uint32_t bad = 0;
    for (int64_t i = 0; i < n; ++i) bad |= (h[i] & 0x7F80) == 0x7F80;
    return bad == 0;
}

inline void bf16s_to_floats(const uint16_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if defined(__aarch64__)
    for (; i + 8 <= n; i += 8) {
        const uint16x8_t h = vld1q_u16(src + i);
        vst1q_f32(dst + i, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16)));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(vshll_high_n_u16(h, 16)));
    }
#endif
    for (; i < n; ++i) dst[i] = bf16_to_float(src[i]);
}

// Returns how many values did not fit bf16 (fits_bf16); those become infinity or NaN.
inline int64_t floats_to_bf16s(const float* src, uint16_t* dst, int64_t n) {
    int64_t bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        bad += !fits_bf16(src[i]);
        dst[i] = float_to_bf16(src[i]);
    }
    return bad;
}

}  // namespace detail
}  // namespace tfi
