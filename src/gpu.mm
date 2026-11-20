// GPU matrix multiplies through Metal Performance Shaders (see gpu.h).
#include "gpu.h"

#ifdef TFI_USE_ACCELERATE

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace tfi {
namespace detail {

namespace {
constexpr int64_t kDefaultMinRows = 256;      // tuned on an M5 Pro
constexpr double kDefaultMinWork = 6e10;       // tuned on an M5 Pro (models of 1-85M parameters)
std::atomic<int64_t> g_min_rows{0};
std::atomic<double> g_min_work{0};
}  // namespace

int64_t gpu_min_rows() {
    if (gpu_only()) return 1;  // GPU-only: no threshold
    const int64_t r = g_min_rows.load();
    return r > 0 ? r : kDefaultMinRows;
}
void set_gpu_min_rows(int64_t rows) { g_min_rows = rows; }
double gpu_min_work() {
    if (gpu_only()) return 0;  // GPU-only: no threshold
    const double w = g_min_work.load();
    return w > 0 ? w : kDefaultMinWork;
}
void set_gpu_min_work(double multiply_adds) { g_min_work = multiply_adds; }

// Kernels of Gpu::Seq, compiled when first needed. The math is src/vmath.h's (same polynomials),
// so GPU and CPU agree to float rounding.
const char* const kKernelSource = R"(
#include <metal_stdlib>
using namespace metal;

static float pow2i(int n) { return as_type<float>((n + 127) << 23); }

static float vexp(float x) {
    const float xc = x > -104.f ? (x < 88.8f ? x : 88.8f) : -104.f;
    const float n = (xc * 1.44269504088896341f + 12582912.f) - 12582912.f;
    const float r = (xc - n * 0.693359375f) - n * -2.12194440e-4f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * r * r + r + 1.f;
    const int ni = int(n), h = ni / 2;
    float y = p * pow2i(h) * pow2i(ni - h);
    y = x > 88.7228394f ? INFINITY : y;
    y = x < -103.98f ? 0.f : y;
    return x != x ? x : y;
}

static float verf(float x) {
    const float a = fabs(x);
    const float s = x * x;
    float p = -6.37348276e-4f;
    p = p * s + 5.06145647e-3f;
    p = p * s + -2.68091038e-2f;
    p = p * s + 1.12828925e-1f;
    p = p * s + -3.76125902e-1f;
    p = p * s + 1.12837911f;
    const float small = x * p;
    const float ac = a < 4.f ? a : 4.f;
    const float t = 1.f / (1.f + ac);
    float q = -0.514288127f;
    q = q * t + 1.09394825f;
    q = q * t + -0.410534561f;
    q = q * t + -0.74236536f;
    q = q * t + 0.463124543f;
    q = q * t + 0.528218627f;
    q = q * t + 0.567896247f;
    q = q * t + -0.000157813804f;
    const float big = 1.f - vexp(-ac * ac) * q;
    const float y = a < 0.84375f ? small : (x < 0.f ? -big : big);
    return x != x ? x : y;
}

static float vtanh(float x) {
    const float a = fabs(x);
    const float s = x * x;
    float p = -5.69752231e-3f;
    p = p * s + 2.06303075e-2f;
    p = p * s + -5.37358634e-2f;
    p = p * s + 1.33313656e-1f;
    p = p * s + -3.33332747e-1f;
    p = p * s + 1.f;
    const float small = x * p;
    const float ac = a < 10.f ? a : 10.f;
    const float big = 1.f - 2.f / (vexp(2.f * ac) + 1.f);
    const float y = a < 0.625f ? small : (x < 0.f ? -big : big);
    return x != x ? x : y;
}

// GpuAct order: relu, gelu, gelu_tanh, silu, tanh, relu2.
static float act(float v, int kind, float k) {
    switch (kind) {
        case 0: return v > 0 ? v : 0.f;
        case 1: return 0.5f * v * (1.f + verf(v * 0.70710678118654752440f));
        case 2: return 0.5f * v * (1.f + vtanh(k * (v + 0.044715f * v * v * v)));
        case 3: return v / (1.f + vexp(-v));
        case 4: return vtanh(v);
        default: return v > 0 ? v * v : 0.f;
    }
}

// d act(v) / dv, as detail::activation_grad_.
static float act_grad(float v, int kind, float k) {
    switch (kind) {
        case 0: return v > 0 ? 1.f : 0.f;
        case 1: return 0.5f * (1.f + verf(v * 0.70710678118654752440f)) + v * 0.398942280401432678f * vexp(-0.5f * v * v);
        case 2: {
            const float th = vtanh(k * (v + 0.044715f * v * v * v));
            return 0.5f * (1.f + th) + 0.5f * v * (1.f - th * th) * k * (1.f + 3.f * 0.044715f * v * v);
        }
        case 3: {
            const float s = 1.f / (1.f + vexp(-v));
            return s * (1.f + v * (1.f - s));
        }
        case 4: {
            const float th = vtanh(v);
            return 1.f - th * th;
        }
        default: return v > 0 ? 2.f * v : 0.f;
    }
}

struct ActArgs { uint n; int kind; float k; };

kernel void act_bwd_k(device const float* dpost [[buffer(0)]], device const float* pre [[buffer(1)]],
                      device float* dpre [[buffer(2)]], constant ActArgs& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) dpre[i] = dpost[i] * act_grad(pre[i], a.kind, a.k);
}

kernel void glu_bwd_k(device const float* dpost [[buffer(0)]], device const float* pre [[buffer(1)]],
                      device const float* up [[buffer(2)]], device float* dpre [[buffer(3)]], device float* dup [[buffer(4)]],
                      constant ActArgs& a [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    const float p = pre[i], g = dpost[i];
    dpre[i] = (g * up[i]) * act_grad(p, a.kind, a.k);
    dup[i] = g * act(p, a.kind, a.k);
}

kernel void act_k(device float* x [[buffer(0)]], constant ActArgs& a [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) x[i] = act(x[i], a.kind, a.k);
}

kernel void glu_k(device float* g [[buffer(0)]], device const float* u [[buffer(1)]], constant ActArgs& a [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    if (i < a.n) g[i] = act(g[i], a.kind, a.k) * u[i];
}

kernel void add_k(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]],
                  constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = a[i] + b[i];
}

kernel void copy_k(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint& n [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = x[i];
}

// rows of keys or values (rows, heads * D) into a cache (heads, cap, D) at positions start, start + 1, ...
struct KvArgs { uint rows, heads, D, cap, start; };
kernel void kv_append_k(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant KvArgs& a [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= a.rows * a.heads * a.D) return;
    const uint d = i % a.D, t = i / a.D, g = t % a.heads, r = t / a.heads;
    dst[(ulong(g) * a.cap + a.start + r) * a.D + d] = src[i];
}

struct BiasArgs { uint n, cols; };

kernel void bias_k(device float* y [[buffer(0)]], device const float* b [[buffer(1)]], constant BiasArgs& a [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < a.n) y[i] += b[i % a.cols];
}

// fp16 copy of x for a half-precision multiply; flags any value outside fp16's range (or NaN).
kernel void half_k(device const float* x [[buffer(0)]], device half* h [[buffer(1)]], device atomic_uint* flag [[buffer(2)]],
                   constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float v = x[i];
    h[i] = half(v);
    if (!(fabs(v) <= 65504.f)) atomic_store_explicit(flag, 1u, memory_order_relaxed);
}

// Sum over the threadgroup; every thread gets the same value (summed in the same order).
static float tg_sum(float v, threadgroup float* part, uint lane, uint sg, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.f;
    for (uint i = 0; i < nsg; ++i) t += part[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

struct NormArgs { uint d; float eps; uint layernorm; uint has_b; };

static float tg_max(float v, threadgroup float* part, uint lane, uint sg, uint nsg) {
    v = simd_max(v);
    if (lane == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = -INFINITY;
    for (uint i = 0; i < nsg; ++i) t = max(t, part[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

// RoPE, rotate-half, out of place (inverse: the transpose, RoPE's backward pass), with cos and sin times
// `scale` (RoPE scaling's attention factor; 1 otherwise): one thread per (row, head, i < half).
struct RopeArgs { uint heads, D, nhalf, n, inverse; float scale; };
kernel void rope_k(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], device const float* inv [[buffer(2)]],
                   device const uint* meta [[buffer(3)]], constant RopeArgs& a [[buffer(4)]], uint t [[thread_position_in_grid]]) {
    if (t >= a.n) return;
    const uint i = t % a.nhalf, rh = t / a.nhalf;  // rh = row * heads + head
    const uint row = rh / a.heads;
    const float ang = float(meta[4 * row]) * inv[i];
    const float c = precise::cos(ang) * a.scale, s = (a.inverse != 0 ? -precise::sin(ang) : precise::sin(ang)) * a.scale;
    device const float* v = x + ulong(rh) * a.D;
    device float* o = y + ulong(rh) * a.D;
    const float x1 = v[i], x2 = v[i + a.nhalf];
    o[i] = x1 * c - x2 * s;
    o[i + a.nhalf] = x2 * c + x1 * s;
}

// The values of each head that RoPE leaves alone (rot <= j < D): copied, one thread each.
struct PassArgs { uint D, rot, n; };
kernel void rope_pass_k(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant PassArgs& a [[buffer(2)]],
                        uint t [[thread_position_in_grid]]) {
    if (t >= a.n) return;
    const uint w = a.D - a.rot, rh = t / w, j = a.rot + t % w;
    y[ulong(rh) * a.D + j] = x[ulong(rh) * a.D + j];
}

// Causal attention, one threadgroup per (row, head): the row's allowed keys in tiles of 1024 with a
// running softmax (max m, sum l), the weighted values accumulated per dimension (at most 4 per thread).
struct AttnArgs { uint H, Hkv, D, Tp, window, has_slopes, has_lse; float scale, cap; };
kernel void attn_k(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]],
                   device float* z [[buffer(3)]], device const float* pk [[buffer(4)]], device const float* pv [[buffer(5)]],
                   device const uint* meta [[buffer(6)]], device const float* slopes [[buffer(7)]], constant AttnArgs& a [[buffer(8)]],
                   device float* lse [[buffer(9)]], uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint tg [[threads_per_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float qs[512];
    threadgroup float ss[1024];
    threadgroup float part[32];
    const uint D = a.D, r = tgid / a.H, h = tgid % a.H, g = h / (a.H / a.Hkv);
    const uint pos = meta[4 * r], start = meta[4 * r + 1], off = meta[4 * r + 2];
    device const float* qr = q + (ulong(r) * a.H + h) * D;
    for (uint d = tid; d < D; d += tg) qs[d] = qr[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint lo = (a.window != 0 && pos + 1 > a.window) ? pos + 1 - a.window : 0;
    const float slope = a.has_slopes != 0 ? slopes[h] : 0.f;
    float m = -INFINITY, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint c0 = lo; c0 <= pos; c0 += 1024) {
        const uint n = min(1024u, pos + 1 - c0);
        float cmax = -INFINITY;
        for (uint jj = tid; jj < n; jj += tg) {
            const uint j = c0 + jj;
            device const float* kr = j < start ? pk + (ulong(g) * a.Tp + j) * D : k + (ulong(off + j - start) * a.Hkv + g) * D;
            float s = 0.f;
            for (uint d = 0; d < D; ++d) s += qs[d] * kr[d];
            s *= a.scale;
            if (a.cap != 0.f) s = a.cap * vtanh(s / a.cap);
            if (slope != 0.f) s += -slope * float(pos - j);
            ss[jj] = s;
            cmax = max(cmax, s);
        }
        cmax = tg_max(cmax, part, lane, sg, nsg);
        const float mnew = max(m, cmax), keep = m == -INFINITY ? 0.f : vexp(m - mnew);
        float lsum = 0.f;
        for (uint jj = tid; jj < n; jj += tg) {
            const float p = vexp(ss[jj] - mnew);
            ss[jj] = p;
            lsum += p;
        }
        lsum = tg_sum(lsum, part, lane, sg, nsg);
        l = l * keep + lsum;
        uint t = 0;
        for (uint d = tid; d < D; d += tg, ++t) {
            float s = acc[t] * keep;
            for (uint jj = 0; jj < n; ++jj) {
                const uint j = c0 + jj;
                device const float* vr = j < start ? pv + (ulong(g) * a.Tp + j) * D : v + (ulong(off + j - start) * a.Hkv + g) * D;
                s += ss[jj] * vr[d];
            }
            acc[t] = s;
        }
        m = mnew;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* zr = z + (ulong(r) * a.H + h) * D;
    uint t = 0;
    for (uint d = tid; d < D; d += tg, ++t) zr[d] = acc[t] / l;
    if (a.has_lse != 0 && tid == 0) lse[ulong(r) * a.H + h] = m + log(l);
}

// Split-key attention for few rows over many keys (a generated token's pass): the row's allowed keys
// in chunks of kSplitKeys, one threadgroup per (row, head, chunk), each writing its chunk's max m, sum
// l and weighted values (D) to `part`; attn_combine_k merges each (row, head)'s chunks into z. The same
// scores, masks and softmax as attn_k, so many threadgroups share a long context instead of one.
constant uint kSplitKeys = 256;
kernel void attn_split_k(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]],
                         device float* part [[buffer(3)]], device const float* pk [[buffer(4)]], device const float* pv [[buffer(5)]],
                         device const uint* meta [[buffer(6)]], device const float* slopes [[buffer(7)]], constant AttnArgs& a [[buffer(8)]],
                         constant uint& nsplit [[buffer(9)]], uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint tg [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float qs[512];
    threadgroup float ss[kSplitKeys];
    threadgroup float red[32];
    const uint D = a.D, split = tgid % nsplit, rh = tgid / nsplit, r = rh / a.H, h = rh % a.H, g = h / (a.H / a.Hkv);
    const uint pos = meta[4 * r], start = meta[4 * r + 1], off = meta[4 * r + 2];
    const uint lo = (a.window != 0 && pos + 1 > a.window) ? pos + 1 - a.window : 0;
    const uint c0 = lo + split * kSplitKeys;
    device float* out = part + ulong(tgid) * (D + 2);
    if (c0 > pos) {  // past this row's keys: an empty chunk
        if (tid == 0) {
            out[0] = -INFINITY;
            out[1] = 0.f;
        }
        return;
    }
    device const float* qr = q + (ulong(r) * a.H + h) * D;
    for (uint d = tid; d < D; d += tg) qs[d] = qr[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float slope = a.has_slopes != 0 ? slopes[h] : 0.f;
    const uint n = min(kSplitKeys, pos + 1 - c0);
    float cmax = -INFINITY;
    for (uint jj = tid; jj < n; jj += tg) {
        const uint j = c0 + jj;
        device const float* kr = j < start ? pk + (ulong(g) * a.Tp + j) * D : k + (ulong(off + j - start) * a.Hkv + g) * D;
        float sc = 0.f;
        for (uint d = 0; d < D; ++d) sc += qs[d] * kr[d];
        sc *= a.scale;
        if (a.cap != 0.f) sc = a.cap * vtanh(sc / a.cap);
        if (slope != 0.f) sc += -slope * float(pos - j);
        ss[jj] = sc;
        cmax = max(cmax, sc);
    }
    cmax = tg_max(cmax, red, lane, sg, nsg);
    float lsum = 0.f;
    for (uint jj = tid; jj < n; jj += tg) {
        const float p = vexp(ss[jj] - cmax);
        ss[jj] = p;
        lsum += p;
    }
    lsum = tg_sum(lsum, red, lane, sg, nsg);  // (its barriers also make every ss visible)
    for (uint d = tid; d < D; d += tg) {
        float acc = 0.f;
        for (uint jj = 0; jj < n; ++jj) {
            const uint j = c0 + jj;
            device const float* vr = j < start ? pv + (ulong(g) * a.Tp + j) * D : v + (ulong(off + j - start) * a.Hkv + g) * D;
            acc += ss[jj] * vr[d];
        }
        out[2 + d] = acc;
    }
    if (tid == 0) {
        out[0] = cmax;
        out[1] = lsum;
    }
}

// attn_split_k's chunks of one (row, head) per threadgroup: rescaled to the largest max, summed, into z.
kernel void attn_combine_k(device const float* part [[buffer(0)]], device float* z [[buffer(1)]], constant AttnArgs& a [[buffer(2)]],
                           constant uint& nsplit [[buffer(3)]], uint rh [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], uint tg [[threads_per_threadgroup]]) {
    const uint D = a.D, stride = D + 2;
    device const float* p = part + ulong(rh) * nsplit * stride;
    float m = -INFINITY;
    for (uint c = 0; c < nsplit; ++c) m = max(m, p[c * stride]);
    float l = 0.f;
    for (uint c = 0; c < nsplit; ++c)
        if (p[c * stride] != -INFINITY) l += p[c * stride + 1] * vexp(p[c * stride] - m);
    device float* zr = z + ulong(rh) * D;
    for (uint d = tid; d < D; d += tg) {
        float acc = 0.f;
        for (uint c = 0; c < nsplit; ++c)
            if (p[c * stride] != -INFINITY) acc += p[c * stride + 2 + d] * vexp(p[c * stride] - m);
        zr[d] = acc / l;
    }
}

// Attention's backward pass, from dz and what the forward pass kept: q, k, v, z and each (row,
// head)'s log-sum-exp of its scores (lse), so that the pattern is recomputed instead of stored. With
// p = exp(score - lse), dp = dz . v and delta = dz . z (= sum of p dp):
// dscore = p (dp - delta) x the softcap's derivative x scale; dq = sum over keys of dscore k,
// dk = sum over queries of dscore q, dv = sum over queries of p dz. No earlier positions (Tp = 0).
// meta has 4 uint32 per row: position, segment start (0), segment's first row, segment's rows.

// delta of each (row, head): one thread each.
kernel void attn_delta_k(device const float* dz [[buffer(0)]], device const float* z [[buffer(1)]], device float* delta [[buffer(2)]],
                         constant uint2& a [[buffer(3)]], uint t [[thread_position_in_grid]]) {  // a: D, rows * H
    if (t >= a.y) return;
    device const float* x = dz + ulong(t) * a.x;
    device const float* y = z + ulong(t) * a.x;
    float s = 0.f;
    for (uint d = 0; d < a.x; ++d) s += x[d] * y[d];
    delta[t] = s;
}

// The score of query row i (position pos_i, head h) and key j (position pos_j) as the forward pass
// computed it from raw = q . k, and the derivative of the softcap (1 when there is none).
static float attn_score(float raw, constant AttnArgs& a, float slope, uint pos_i, uint pos_j, thread float& dcap) {
    float s = raw * a.scale;
    dcap = 1.f;
    if (a.cap != 0.f) {
        const float th = vtanh(s / a.cap);
        s = a.cap * th;
        dcap = 1.f - th * th;
    }
    if (slope != 0.f) s += -slope * float(pos_i - pos_j);
    return s;
}

// One whole sequence (rows at positions 0..T-1): the attention pattern P (H, T, T) from q, k and the
// forward pass's lse, as Model::attend's hook_pattern (entries a row may not attend to are 0), one
// thread per entry; and z (T, H D) = P v, one thread per value.
kernel void attn_pattern_k(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* lse [[buffer(2)]],
                           device const float* slopes [[buffer(3)]], constant AttnArgs& a [[buffer(4)]], device float* P [[buffer(5)]],
                           constant uint& T [[buffer(6)]], uint t [[thread_position_in_grid]]) {
    if (t >= a.H * T * T) return;
    const uint j = t % T, i = (t / T) % T, h = t / (T * T), g = h / (a.H / a.Hkv);
    const uint lo = (a.window != 0 && i + 1 > a.window) ? i + 1 - a.window : 0;
    float p = 0.f;
    if (j <= i && j >= lo) {
        device const float* qr = q + (ulong(i) * a.H + h) * a.D;
        device const float* kr = k + (ulong(j) * a.Hkv + g) * a.D;
        float raw = 0.f;
        for (uint d = 0; d < a.D; ++d) raw += qr[d] * kr[d];
        float dcap;
        p = vexp(attn_score(raw, a, a.has_slopes != 0 ? slopes[h] : 0.f, i, j, dcap) - lse[ulong(i) * a.H + h]);
    }
    P[t] = p;
}
kernel void attn_apply_k(device const float* P [[buffer(0)]], device const float* v [[buffer(1)]], device float* z [[buffer(2)]],
                         constant AttnArgs& a [[buffer(3)]], constant uint& T [[buffer(4)]], uint t [[thread_position_in_grid]]) {
    if (t >= T * a.H * a.D) return;
    const uint d = t % a.D, h = (t / a.D) % a.H, i = t / (a.D * a.H), g = h / (a.H / a.Hkv);
    device const float* pr = P + (ulong(h) * T + i) * T;
    float s = 0.f;
    for (uint j = 0; j < T; ++j) s += pr[j] * v[(ulong(j) * a.Hkv + g) * a.D + d];
    z[t] = s;
}

// dq: one threadgroup per (row, head), its keys in tiles of 1024 (as attn_k).
kernel void attn_bwd_dq_k(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]],
                          device const float* dz [[buffer(3)]], device const float* lse [[buffer(4)]], device const float* delta [[buffer(5)]],
                          device const uint* meta [[buffer(6)]], device const float* slopes [[buffer(7)]], constant AttnArgs& a [[buffer(8)]],
                          device float* dq [[buffer(9)]],
                          uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint tg [[threads_per_threadgroup]]) {
    threadgroup float qs[512];
    threadgroup float gs[512];
    threadgroup float ds[1024];
    const uint D = a.D, r = tgid / a.H, h = tgid % a.H, g = h / (a.H / a.Hkv);
    const uint pos = meta[4 * r], off = meta[4 * r + 2];
    for (uint d = tid; d < D; d += tg) {
        qs[d] = q[(ulong(r) * a.H + h) * D + d];
        gs[d] = dz[(ulong(r) * a.H + h) * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint lo = (a.window != 0 && pos + 1 > a.window) ? pos + 1 - a.window : 0;
    const float slope = a.has_slopes != 0 ? slopes[h] : 0.f;
    const float L = lse[ulong(r) * a.H + h], dl = delta[ulong(r) * a.H + h];
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint c0 = lo; c0 <= pos; c0 += 1024) {
        const uint n = min(1024u, pos + 1 - c0);
        for (uint jj = tid; jj < n; jj += tg) {
            const uint j = c0 + jj;
            device const float* kr = k + (ulong(off + j) * a.Hkv + g) * D;
            device const float* vr = v + (ulong(off + j) * a.Hkv + g) * D;
            float raw = 0.f, dp = 0.f;
            for (uint d = 0; d < D; ++d) {
                raw += qs[d] * kr[d];
                dp += gs[d] * vr[d];
            }
            float dcap;
            const float p = vexp(attn_score(raw, a, slope, pos, j, dcap) - L);
            ds[jj] = p * (dp - dl) * dcap * a.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint t = 0;
        for (uint d = tid; d < D; d += tg, ++t) {
            float s = acc[t];
            for (uint jj = 0; jj < n; ++jj) s += ds[jj] * k[(ulong(off + c0 + jj) * a.Hkv + g) * D + d];
            acc[t] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint t = 0;
    for (uint d = tid; d < D; d += tg, ++t) dq[(ulong(r) * a.H + h) * D + d] = acc[t];
}

// dk, dv: one threadgroup per (key row, kv head), over the query heads that share it and the
// segment's later rows that attend to it, in tiles of 1024.
kernel void attn_bwd_dkv_k(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]],
                           device const float* dz [[buffer(3)]], device const float* lse [[buffer(4)]], device const float* delta [[buffer(5)]],
                           device const uint* meta [[buffer(6)]], device const float* slopes [[buffer(7)]], constant AttnArgs& a [[buffer(8)]],
                           device float* dk [[buffer(9)]], device float* dv [[buffer(10)]],
                           uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint tg [[threads_per_threadgroup]]) {
    threadgroup float ks[512];
    threadgroup float vs[512];
    threadgroup float ps[1024];
    threadgroup float ds[1024];
    const uint D = a.D, r = tgid / a.Hkv, g = tgid % a.Hkv, rep = a.H / a.Hkv;
    const uint pos = meta[4 * r], off = meta[4 * r + 2], len = meta[4 * r + 3];
    for (uint d = tid; d < D; d += tg) {
        ks[d] = k[(ulong(r) * a.Hkv + g) * D + d];
        vs[d] = v[(ulong(r) * a.Hkv + g) * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint hi = len;  // query positions pos..hi-1 attend to this key
    if (a.window != 0) hi = min(hi, pos + a.window);
    float ak[4] = {0.f, 0.f, 0.f, 0.f}, av[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint h = g * rep; h < (g + 1) * rep; ++h) {
        const float slope = a.has_slopes != 0 ? slopes[h] : 0.f;
        for (uint c0 = pos; c0 < hi; c0 += 1024) {
            const uint n = min(1024u, hi - c0);
            for (uint ii = tid; ii < n; ii += tg) {
                const ulong qh = ulong(off + c0 + ii) * a.H + h;
                device const float* qr = q + qh * D;
                device const float* gr = dz + qh * D;
                float raw = 0.f, dp = 0.f;
                for (uint d = 0; d < D; ++d) {
                    raw += qr[d] * ks[d];
                    dp += gr[d] * vs[d];
                }
                float dcap;
                const float p = vexp(attn_score(raw, a, slope, c0 + ii, pos, dcap) - lse[qh]);
                ps[ii] = p;
                ds[ii] = p * (dp - delta[qh]) * dcap * a.scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint t = 0;
            for (uint d = tid; d < D; d += tg, ++t) {
                float sk = ak[t], sv = av[t];
                for (uint ii = 0; ii < n; ++ii) {
                    const ulong qh = ulong(off + c0 + ii) * a.H + h;
                    sk += ds[ii] * q[qh * D + d];
                    sv += ps[ii] * dz[qh * D + d];
                }
                ak[t] = sk;
                av[t] = sv;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    uint t = 0;
    for (uint d = tid; d < D; d += tg, ++t) {
        dk[(ulong(r) * a.Hkv + g) * D + d] = ak[t];
        dv[(ulong(r) * a.Hkv + g) * D + d] = av[t];
    }
}


// Tiled causal attention on the matrix units, for head sizes that are multiples of 8 up to 128: one
// threadgroup of 4 simdgroups per block of up to 32 consecutive query rows of one segment, for one
// head; simdgroup i takes rows 8i..8i+7. Each tile of 16 keys and values is loaded once into
// threadgroup memory for all 32 rows, and the scores (Q K^T) and outputs (P V) are 8x8 matrix
// products, with the softmax kept online (a running max and sum per row). blk: 4 uint32 per block
// (first row, rows, segment start position, segment's first row).
//
// In an 8x8 simdgroup matrix, lane l holds the elements at row 4*b4 + 2*b2 + b1 and columns
// 4*b3 + 2*b0 and the next one (bi: bit i of l), so a row's 8 values are spread over the 4 lanes
// differing in bits 0 and 3: row reductions are shuffles by 1 and 8.
#define AQ 32
#define AK 16
#define AD 128
kernel void attn_mma_k(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]],
                       device float* z [[buffer(3)]], device const float* pk [[buffer(4)]], device const float* pv [[buffer(5)]],
                       device const uint* blk [[buffer(6)]], device const float* slopes [[buffer(7)]], constant AttnArgs& a [[buffer(8)]],
                       device float* lse [[buffer(9)]], uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[AQ * AD];          // the block's queries, then each tile's keys and values
    threadgroup float* Ks = sh;
    threadgroup float* Vs = sh + AK * AD;
    const uint D = a.D, b = tgid / a.H, h = tgid % a.H, g = h / (a.H / a.Hkv);
    const uint off = blk[4 * b], nq = blk[4 * b + 1], start = blk[4 * b + 2], seg = blk[4 * b + 3];
    const uint pos0 = start + (off - seg);   // position of the block's first row
    const uint last = pos0 + nq - 1;
    const uint lo = (a.window != 0 && pos0 + 1 > a.window) ? pos0 + 1 - a.window : 0;
    const uint fm = 4 * ((lane >> 4) & 1) + 2 * ((lane >> 2) & 1) + ((lane >> 1) & 1);  // this lane's row
    const uint fn = 4 * ((lane >> 3) & 1) + 2 * (lane & 1);                              // and first column
    const uint qi = 8 * sg + fm;             // this lane's query row in the block
    const uint pos = pos0 + qi;
    const bool live = qi < nq;
    const uint row_lo = (a.window != 0 && pos + 1 > a.window) ? pos + 1 - a.window : 0;
    const float slope = a.has_slopes != 0 ? slopes[h] : 0.f;
    const uint nd = D / 8;
    for (uint e = tid; e < AQ * D; e += 128) {
        const uint r = e / D, d = e % D;
        sh[e] = r < nq ? q[(ulong(off + r) * a.H + h) * D + d] : 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 Q[AD / 8], O[AD / 8];
    for (uint t = 0; t < nd; ++t) {
        simdgroup_load(Q[t], sh, D, ulong2(8 * t, 8 * sg));
        O[t] = simdgroup_float8x8(0.f);
    }
    float m = -INFINITY, l = 0.f;
    for (uint j0 = lo; j0 <= last; j0 += AK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);  // the queries or the previous tile are no longer read
        for (uint e = tid; e < AK * D; e += 128) {
            const uint jj = e / D, d = e % D, j = j0 + jj;
            float kv = 0.f, vv = 0.f;
            if (j <= last) {
                const ulong src = j < start ? (ulong(g) * a.Tp + j) * D + d : (ulong(seg + j - start) * a.Hkv + g) * D + d;
                kv = j < start ? pk[src] : k[src];
                vv = j < start ? pv[src] : v[src];
            }
            Ks[e] = kv;
            Vs[e] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 S[AK / 8];
        for (uint c = 0; c < AK / 8; ++c) {
            S[c] = simdgroup_float8x8(0.f);
            for (uint t = 0; t < nd; ++t) {
                simdgroup_float8x8 Kt;
                simdgroup_load(Kt, Ks, D, ulong2(8 * t, 8 * c), true);  // keys 8c.. as columns, dims 8t.. as rows
                simdgroup_multiply_accumulate(S[c], Q[t], Kt, S[c]);
            }
        }
        // Scale, softcap, ALiBi and mask this lane's 4 scores, then the online softmax for its row.
        float sc[AK / 4];
        float tmax = -INFINITY;
        for (uint c = 0; c < AK / 8; ++c)
            for (uint u = 0; u < 2; ++u) {
                const uint j = j0 + 8 * c + fn + u;
                float x = -INFINITY;
                if (live && j <= pos && j >= row_lo) {
                    x = S[c].thread_elements()[u] * a.scale;
                    if (a.cap != 0.f) x = a.cap * vtanh(x / a.cap);
                    if (slope != 0.f) x += -slope * float(pos - j);
                }
                sc[2 * c + u] = x;
                tmax = max(tmax, x);
            }
        tmax = max(tmax, simd_shuffle_xor(tmax, 1));
        tmax = max(tmax, simd_shuffle_xor(tmax, 8));
        const float mnew = max(m, tmax);
        const float keep = m == -INFINITY ? 0.f : vexp(m - mnew);
        float psum = 0.f;
        for (uint c = 0; c < AK / 8; ++c)
            for (uint u = 0; u < 2; ++u) {
                const float p = sc[2 * c + u] == -INFINITY ? 0.f : vexp(sc[2 * c + u] - mnew);
                S[c].thread_elements()[u] = p;
                psum += p;
            }
        psum += simd_shuffle_xor(psum, 1);
        psum += simd_shuffle_xor(psum, 8);
        l = l * keep + psum;
        m = mnew;
        for (uint t = 0; t < nd; ++t) {
            O[t].thread_elements() *= keep;  // both of this lane's elements are in its row
            for (uint c = 0; c < AK / 8; ++c) {
                simdgroup_float8x8 Vb;
                simdgroup_load(Vb, Vs, D, ulong2(8 * t, 8 * c));
                simdgroup_multiply_accumulate(O[t], S[c], Vb, O[t]);
            }
        }
    }
    if (!live) return;
    device float* zr = z + (ulong(off + qi) * a.H + h) * D;
    for (uint t = 0; t < nd; ++t) {
        zr[8 * t + fn] = O[t].thread_elements()[0] / l;
        zr[8 * t + fn + 1] = O[t].thread_elements()[1] / l;
    }
    if (a.has_lse != 0 && fn == 0) lse[ulong(off + qi) * a.H + h] = m + log(l);
}

// One threadgroup per block: the sum of squares of a bm x bn block of a matrix with row stride ld.
struct BlockArgs { uint ld, bm, bn, nbc; };
kernel void block_sumsq_k(device const float* c [[buffer(0)]], device float* out [[buffer(1)]],
                          constant BlockArgs& a [[buffer(2)]], uint blk [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint tg [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32];
    const uint bi = blk / a.nbc, bj = blk % a.nbc;
    device const float* base = c + ulong(bi) * a.bm * a.ld + ulong(bj) * a.bn;
    float s = 0.f;
    for (uint r = 0; r < a.bm; ++r)
        for (uint j = tid; j < a.bn; j += tg) {
            const float v = base[ulong(r) * a.ld + j];
            s += v * v;
        }
    s = tg_sum(s, part, lane, sg, nsg);
    if (tid == 0) out[blk] = s;
}

// One threadgroup per row: the index of the row's largest value, the first on ties.
struct ArgmaxArgs { uint n, ld; };
kernel void row_argmax_k(device const float* c [[buffer(0)]], device uint* out [[buffer(1)]],
                         constant ArgmaxArgs& a [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint tg [[threads_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float pv[32];
    threadgroup uint pi[32];
    device const float* r = c + ulong(row) * a.ld;
    float best = -INFINITY;
    uint idx = 0xFFFFFFFFu;
    for (uint j = tid; j < a.n; j += tg) {
        const float v = r[j];
        if (idx == 0xFFFFFFFFu || v > best) {
            best = v;
            idx = j;
        }
    }
    const float m = simd_max(best);
    const uint i = simd_min(best == m ? idx : 0xFFFFFFFFu);
    if (lane == 0) {
        pv[sg] = m;
        pi[sg] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float bv = pv[0];
        uint bix = pi[0];
        for (uint s = 1; s < nsg; ++s)
            if (pv[s] > bv || (pv[s] == bv && pi[s] < bix)) {
                bv = pv[s];
                bix = pi[s];
            }
        out[row] = bix;
    }
}

// One threadgroup per row.
kernel void norm_k(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], device const float* w [[buffer(2)]],
                   device const float* b [[buffer(3)]], constant NormArgs& a [[buffer(4)]],
                   uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint tg [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32];
    const uint d = a.d;
    device const float* xi = x + ulong(row) * d;
    device float* yi = y + ulong(row) * d;
    float mean = 0.f;
    if (a.layernorm) {
        float s = 0.f;
        for (uint j = tid; j < d; j += tg) s += xi[j];
        mean = tg_sum(s, part, lane, sg, nsg) / float(d);
    }
    float q = 0.f;
    for (uint j = tid; j < d; j += tg) {
        const float t = xi[j] - mean;
        q += t * t;
    }
    const float inv = 1.0f / precise::sqrt(tg_sum(q, part, lane, sg, nsg) / float(d) + a.eps);
    for (uint j = tid; j < d; j += tg) {
        const float v = (xi[j] - mean) * inv * w[j];
        yi[j] = a.has_b ? v + b[j] : v;
    }
}

// Backward of norm_k (without its bias): dx from x and dy, one threadgroup per row.
kernel void norm_bwd_k(device const float* x [[buffer(0)]], device const float* dy [[buffer(1)]],
                       device const float* w [[buffer(2)]], device float* dx [[buffer(3)]], constant NormArgs& a [[buffer(4)]],
                       uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint tg [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32];
    const uint d = a.d;
    device const float* xi = x + ulong(row) * d;
    device const float* dyi = dy + ulong(row) * d;
    device float* dxi = dx + ulong(row) * d;
    if (a.layernorm) {  // dx = inv * (g - mean(g) - xhat * mean(g * xhat)), g = dy * w
        float s = 0.f;
        for (uint j = tid; j < d; j += tg) s += xi[j];
        const float mean = tg_sum(s, part, lane, sg, nsg) / float(d);
        float q = 0.f;
        for (uint j = tid; j < d; j += tg) {
            const float t = xi[j] - mean;
            q += t * t;
        }
        const float inv = 1.0f / precise::sqrt(tg_sum(q, part, lane, sg, nsg) / float(d) + a.eps);
        float gs = 0.f, gx = 0.f;
        for (uint j = tid; j < d; j += tg) {
            const float g = dyi[j] * w[j];
            gs += g;
            gx += g * (xi[j] - mean) * inv;
        }
        const float gm = tg_sum(gs, part, lane, sg, nsg) / float(d);
        const float gxm = tg_sum(gx, part, lane, sg, nsg) / float(d);
        for (uint j = tid; j < d; j += tg) dxi[j] = inv * (dyi[j] * w[j] - gm - (xi[j] - mean) * inv * gxm);
    } else {  // rmsnorm: dx = inv * g - x * inv^3 * mean(g * x)
        float ms = 0.f, gx = 0.f;
        for (uint j = tid; j < d; j += tg) {
            ms += xi[j] * xi[j];
            gx += dyi[j] * w[j] * xi[j];
        }
        const float inv = 1.0f / precise::sqrt(tg_sum(ms, part, lane, sg, nsg) / float(d) + a.eps);
        const float gxm = tg_sum(gx, part, lane, sg, nsg) / float(d);
        for (uint j = tid; j < d; j += tg) dxi[j] = inv * dyi[j] * w[j] - xi[j] * inv * inv * inv * gxm;
    }
}
)";

struct ActArgs { uint32_t n; int32_t kind; float k; };
struct BiasArgs { uint32_t n, cols; };
struct NormArgs { uint32_t d; float eps; uint32_t layernorm, has_b; };
struct RopeArgs { uint32_t heads, D, nhalf, n, inverse; float scale; };
struct PassArgs { uint32_t D, rot, n; };
struct KvArgs { uint32_t rows, heads, D, cap, start; };
constexpr int64_t kSplitKeys = 256;  // as the kernel's
struct AttnArgs { uint32_t H, Hkv, D, Tp, window, has_slopes, has_lse; float scale, cap; };

// Write src as fp16; false if a value is outside fp16's range (or not finite).
bool to_half(const float* src, __fp16* dst, int64_t n) {
    int64_t out_of_range = 0;
    for (int64_t i = 0; i < n; ++i) {
        out_of_range += !(std::fabs(src[i]) <= 65504.f);
        dst[i] = static_cast<__fp16>(src[i]);
    }
    return out_of_range == 0;
}

struct Gpu::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    std::mutex mutex;  // guards the weight caches and pool
    // By host address. Vectors (norm scales, biases): GPU copies. float32 matrices in fp16 runs: fp16
    // copies (nil for one outside fp16's range). Matrices (float32 or fp16): buffers over their own memory.
    std::unordered_map<const float*, id<MTLBuffer>> weights, weights_fp16;
    std::unordered_map<const void*, id<MTLBuffer>> shared;
    std::vector<id<MTLBuffer>> pool;  // free scratch buffers

    // Where a multiply reads a weight from, and whether those values are fp16.
    struct WBuf {
        id<MTLBuffer> buf = nil;
        bool half = false;
    };
    // A matrix is read in place, float32 or fp16. In an fp16 run, a float32 matrix gives an fp16
    // copy instead when every value fits.
    WBuf weight_for(const Weight& W, bool fp16) {
        if (W.dim() != 2) throw std::invalid_argument("linear weight must be 2-D");
        if (fp16 && !W.is_fp16())
            if (id<MTLBuffer> b = copy_buffer(W.f32_data(), W.numel(), true)) return {b, true};
        return {shared_buffer(W), W.is_fp16()};
    }

    id<MTLBuffer> shared_buffer(const Weight& W) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = shared.find(W.data());
        if (it != shared.end()) return it->second;
        // The buffer keeps the weight's memory alive (Weight allocates it page-aligned, in whole pages).
        std::shared_ptr<const void> keep = W.owner();
        id<MTLBuffer> b = [device newBufferWithBytesNoCopy:const_cast<void*>(W.data())
                                                    length:W.capacity()
                                                   options:MTLResourceStorageModeShared
                                               deallocator:^(void*, NSUInteger) { (void)keep; }];
        if (!b) b = [device newBufferWithBytes:W.data() length:W.capacity() options:MTLResourceStorageModeShared];
        if (!b) throw std::runtime_error("GPU: could not allocate a weight buffer");
        shared.emplace(W.data(), b);
        return b;
    }

    id<MTLBuffer> weight_buffer(const Tensor& W, bool fp16) { return copy_buffer(W.ptr(), W.numel(), fp16); }

    // A GPU copy of n floats at p (fp16 when asked: nil if a value doesn't fit), cached by address.
    id<MTLBuffer> copy_buffer(const float* p, int64_t n, bool fp16) {
        std::lock_guard<std::mutex> lock(mutex);
        auto& cache = fp16 ? weights_fp16 : weights;
        auto it = cache.find(p);
        if (it != cache.end()) return it->second;
        const NSUInteger bytes = static_cast<NSUInteger>(std::max<int64_t>(n, 1)) * (fp16 ? sizeof(__fp16) : sizeof(float));
        id<MTLBuffer> b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!b) throw std::runtime_error("GPU: could not allocate a weight buffer");
        if (!fp16) std::memcpy(b.contents, p, static_cast<size_t>(n) * sizeof(float));
        else if (!to_half(p, static_cast<__fp16*>(b.contents), n)) b = nil;
        cache.emplace(p, b);
        return b;
    }

    id<MTLBuffer> acquire(size_t bytes) {
        {
            // The smallest free buffer that fits, so small requests don't take large buffers.
            std::lock_guard<std::mutex> lock(mutex);
            size_t best = pool.size();
            for (size_t i = 0; i < pool.size(); ++i)
                if (pool[i].length >= bytes && (best == pool.size() || pool[i].length < pool[best].length)) best = i;
            if (best < pool.size()) {
                id<MTLBuffer> b = pool[best];
                pool.erase(pool.begin() + static_cast<std::ptrdiff_t>(best));
                return b;
            }
        }
        id<MTLBuffer> b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!b) throw std::runtime_error("GPU: could not allocate a buffer");
        return b;
    }

    void release(id<MTLBuffer> b) {
        std::lock_guard<std::mutex> lock(mutex);
        pool.push_back(b);
    }

    // Gpu::Seq's kernels, compiled on first use.
    std::once_flag kernels_once;
    id<MTLComputePipelineState> kv_append_k, attn_split_k, attn_combine_k;
    id<MTLComputePipelineState> act_k, glu_k, add_k, copy_k, bias_k, half_k, norm_k, norm_bwd_k, act_bwd_k, glu_bwd_k,
        block_sumsq_k, row_argmax_k, rope_k, rope_pass_k, attn_k, attn_mma_k, attn_delta_k, attn_bwd_dq_k, attn_bwd_dkv_k, attn_pattern_k, attn_apply_k;

    void load_kernels() {
        std::call_once(kernels_once, [this] {
            @autoreleasepool {
                MTLCompileOptions* opt = [MTLCompileOptions new];
                opt.mathMode = MTLMathModeSafe;  // no fast-math rewrites: keep the CPU's arithmetic (macOS 15+)
                NSError* err = nil;
                id<MTLLibrary> lib = [device newLibraryWithSource:@(kKernelSource) options:opt error:&err];
                if (!lib)
                    throw std::runtime_error(std::string("GPU: kernels did not compile: ") +
                                             (err ? err.localizedDescription.UTF8String : "unknown error"));
                auto make = [&](const char* name) {
                    id<MTLFunction> f = [lib newFunctionWithName:@(name)];
                    id<MTLComputePipelineState> p = f ? [device newComputePipelineStateWithFunction:f error:&err] : nil;
                    if (!p) throw std::runtime_error(std::string("GPU: could not load kernel ") + name);
                    return p;
                };
                act_k = make("act_k");
                glu_k = make("glu_k");
                add_k = make("add_k");
                copy_k = make("copy_k");
                kv_append_k = make("kv_append_k");
                bias_k = make("bias_k");
                half_k = make("half_k");
                norm_k = make("norm_k");
                norm_bwd_k = make("norm_bwd_k");
                act_bwd_k = make("act_bwd_k");
                glu_bwd_k = make("glu_bwd_k");
                block_sumsq_k = make("block_sumsq_k");
                row_argmax_k = make("row_argmax_k");
                rope_k = make("rope_k");
                rope_pass_k = make("rope_pass_k");
                attn_k = make("attn_k");
                attn_mma_k = make("attn_mma_k");
                attn_split_k = make("attn_split_k");
                attn_combine_k = make("attn_combine_k");
                attn_delta_k = make("attn_delta_k");
                attn_bwd_dq_k = make("attn_bwd_dq_k");
                attn_bwd_dkv_k = make("attn_bwd_dkv_k");
                attn_pattern_k = make("attn_pattern_k");
                attn_apply_k = make("attn_apply_k");
            }
        });
    }
};

Gpu::Gpu() : impl_(new Impl) {}
Gpu::~Gpu() = default;

std::unique_ptr<Gpu> Gpu::create() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !MPSSupportsMTLDevice(device)) return nullptr;
        std::unique_ptr<Gpu> gpu(new Gpu);
        gpu->impl_->device = device;
        gpu->impl_->queue = [device newCommandQueue];
        if (!gpu->impl_->queue) return nullptr;
        return gpu;
    }
}

Tensor Gpu::linear(const Tensor& x, const Weight& W, const Weight* b, bool fp16) {
    return std::move(linear_many(x, {&W}, {b}, fp16)[0]);
}

std::vector<Tensor> Gpu::linear_many(const Tensor& x, const std::vector<const Weight*>& Ws,
                                     const std::vector<const Weight*>& bs, bool fp16) {
    if (!bs.empty() && bs.size() != Ws.size()) throw std::invalid_argument("linear_many: one bias per weight");
    const int64_t k = x.size(-1), m = k ? x.numel() / k : 0;
    std::vector<Tensor> ys;
    ys.reserve(Ws.size());
    for (const Weight* W : Ws) {
        if (W->dim() != 2) throw std::invalid_argument("linear weight must be 2-D");
        if (W->shape[1] != k)
            throw std::invalid_argument("linear: input " + shape_str(x.shape) + " vs weight " + shape_str(W->shape));
        Shape shape(x.shape.begin(), x.shape.end() - 1);
        shape.push_back(W->shape[0]);
        ys.emplace_back(std::move(shape));
    }
    if (m == 0) return ys;
    Impl& g = *impl_;
    const size_t nw = Ws.size();
    @autoreleasepool {
        // Each weight takes fp16 inputs if asked and both it and x fit, else fp32 inputs (with a
        // half-precision weight, fp32 x times its fp16 values: as exact as linear()).
        std::vector<Impl::WBuf> wb(nw);
        std::vector<bool> half(nw, false);  // fp16 inputs
        id<MTLBuffer> xh = nil, xf = nil;
        if (fp16) {
            bool any = false;
            for (size_t i = 0; i < nw; ++i)
                if (Ws[i]->numel()) any |= (wb[i] = g.weight_for(*Ws[i], true)).half;
            if (any) {
                xh = g.acquire(static_cast<size_t>(m * k) * sizeof(__fp16));
                if (!to_half(x.ptr(), static_cast<__fp16*>(xh.contents), m * k)) {
                    g.release(xh);
                    xh = nil;
                }
            }
            for (size_t i = 0; i < nw; ++i) half[i] = xh && wb[i].half;
        }
        for (size_t i = 0; i < nw; ++i) {
            if (half[i] || Ws[i]->numel() == 0) continue;
            wb[i] = g.weight_for(*Ws[i], false);
            if (!xf) {
                xf = g.acquire(static_cast<size_t>(m * k) * sizeof(float));
                std::memcpy(xf.contents, x.ptr(), static_cast<size_t>(m * k) * sizeof(float));
            }
        }
        std::vector<id<MTLBuffer>> yb(nw, nil);
        id<MTLCommandBuffer> cmd = [g.queue commandBuffer];
        for (size_t i = 0; i < nw; ++i) {
            const int64_t n = Ws[i]->shape[0];
            if (n == 0) continue;
            yb[i] = g.acquire(static_cast<size_t>(m * n) * sizeof(float));
            const MPSDataType in_type = half[i] ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
            const NSUInteger in_size = half[i] ? sizeof(__fp16) : sizeof(float);
            const MPSDataType w_type = wb[i].half ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
            const NSUInteger w_size = wb[i].half ? sizeof(__fp16) : sizeof(float);
            MPSMatrixDescriptor* dx = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k * in_size
                                                                           dataType:in_type];
            MPSMatrixDescriptor* dw = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:k * w_size
                                                                           dataType:w_type];
            MPSMatrixDescriptor* dy = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n
                                                                           rowBytes:n * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];
            MPSMatrix* mx = [[MPSMatrix alloc] initWithBuffer:(half[i] ? xh : xf) descriptor:dx];
            MPSMatrix* mw = [[MPSMatrix alloc] initWithBuffer:wb[i].buf descriptor:dw];
            MPSMatrix* my = [[MPSMatrix alloc] initWithBuffer:yb[i] descriptor:dy];
            // A kernel object per call: MPS kernels must not be shared between threads.
            MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:g.device
                                                                            transposeLeft:NO
                                                                           transposeRight:YES
                                                                               resultRows:m
                                                                            resultColumns:n
                                                                          interiorColumns:k
                                                                                    alpha:1.0
                                                                                     beta:0.0];
            [mm encodeToCommandBuffer:cmd leftMatrix:mx rightMatrix:mw resultMatrix:my];
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        const bool ok = cmd.status == MTLCommandBufferStatusCompleted;
        for (size_t i = 0; i < nw; ++i) {
            if (!yb[i]) continue;
            if (ok) std::memcpy(ys[i].ptr(), yb[i].contents, static_cast<size_t>(ys[i].numel()) * sizeof(float));
            g.release(yb[i]);
        }
        if (xh) g.release(xh);
        if (xf) g.release(xf);
        if (!ok) throw std::runtime_error("GPU: matrix multiply failed");
    }
    for (size_t i = 0; i < nw; ++i) {
        const Weight* b = bs.empty() ? nullptr : bs[i];
        if (!b) continue;
        const int64_t n = Ws[i]->shape[0];
        const Tensor& bias = b->f32();
        if (bias.numel() != n) throw std::invalid_argument("linear: bias does not match the weight");
        float* y = ys[i].ptr();
        for (int64_t r = 0; r < m; ++r)
            for (int64_t j = 0; j < n; ++j) y[r * n + j] += bias.data[j];
    }
    return ys;
}

// --- Gpu::Seq ---------------------------------------------------------------------------

struct Gpu::Seq::Impl {
    Gpu::Impl& g;
    struct Slot {
        id<MTLBuffer> buf = nil, half = nil;  // half: fp16 copy for half-precision multiplies
        int64_t floats = 0;
    };
    struct Op {
        enum Kind { Linear, LinearT, Norm, NormBwd, Act, Glu, Add, Copy, ActBwd, GluBwd, Rope, Attention, AttentionBwd, Pattern, Apply, KvAppend } kind;
        int a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0, i = 0, j = -1;
        int more[4] = {0, 0, 0, 0};  // AttentionBwd: delta, dq, dk, dv
        Gpu::Seq::AttnShape attn{};      // Attention
        int64_t heads = 0, D = 0, rot = 0;  // Rope
        float scale = 1.f;                  // Rope
        int64_t rows = 0, n = 0;
        const Tensor* W = nullptr;     // Norm, NormBwd
        const Tensor* bias = nullptr;  // Norm
        const Weight* lw = nullptr;    // Linear, LinearT
        const Weight* lbias = nullptr; // Linear
        GpuAct act = GpuAct::Relu;
        float eps = 0.f;
        bool layernorm = false;
        bool inverse = false;  // Rope
    };
    std::vector<Slot> slots;
    std::vector<Op> ops;
    id<MTLBuffer> flag = nil;  // set by half_k when an input does not fit fp16
    id<MTLBuffer> split = nil;  // split-key attention's partial results (attn_split_k)
    size_t split_bytes = 0;
    std::map<std::tuple<int64_t, int64_t, int64_t, bool>, MPSMatrixMultiplication*> mms;  // by (m, n, k, W^T)

    // Encoding state of one attempt.
    id<MTLCommandBuffer> cmd = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    std::vector<bool> half_valid;

    explicit Impl(Gpu::Impl& gpu) : g(gpu) {}

    Slot& slot(int i) {
        if (i < 0 || static_cast<size_t>(i) >= slots.size() || !slots[i].buf)
            throw std::logic_error("Gpu::Seq: slot " + std::to_string(i) + " was not reserved");
        return slots[i];
    }
    void need(int i, int64_t floats) {
        if (slot(i).floats < floats) throw std::logic_error("Gpu::Seq: slot " + std::to_string(i) + " is too small");
    }
    id<MTLComputeCommandEncoder> compute() {
        if (!enc) enc = [cmd computeCommandEncoder];
        return enc;
    }
    void end_compute() {
        if (enc) [enc endEncoding];
        enc = nil;
    }
    void dispatch(id<MTLComputePipelineState> p, int64_t n) {
        const NSUInteger tg = std::min<NSUInteger>(256, p.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
    static uint32_t u32(int64_t n) {
        if (n < 0 || n > static_cast<int64_t>(UINT32_MAX)) throw std::length_error("Gpu::Seq: operation too large");
        return static_cast<uint32_t>(n);
    }

    // y (m, n) = x (m, k) W^T, W (n, k); or without trans_w, y = x W, W (k, n).
    MPSMatrixMultiplication* mm(int64_t m, int64_t n, int64_t k, bool trans_w = true) {
        const auto key = std::make_tuple(m, n, k, trans_w);
        auto it = mms.find(key);
        if (it != mms.end()) return it->second;
        MPSMatrixMultiplication* p = [[MPSMatrixMultiplication alloc] initWithDevice:g.device
                                                                       transposeLeft:NO
                                                                      transposeRight:trans_w ? YES : NO
                                                                          resultRows:m
                                                                       resultColumns:n
                                                                     interiorColumns:k
                                                                               alpha:1.0
                                                                                beta:0.0];
        mms.emplace(key, p);
        return p;
    }

    void encode_linear(const Op& op, bool fp16) {
        const Weight& W = *op.lw;
        const int64_t m = op.rows, n = W.shape[0], k = W.shape[1];
        need(op.a, m * k);
        need(op.c, m * n);
        if (m == 0 || n == 0) return;
        if (k == 0) throw std::logic_error("Gpu::Seq: linear with an empty input");
        Gpu::Impl::WBuf wb = fp16 ? g.weight_for(W, true) : Gpu::Impl::WBuf{};
        const bool half = wb.buf && wb.half;  // fp16 inputs
        id<MTLBuffer> in = slot(op.a).buf;
        if (half) {
            Slot& s = slot(op.a);
            if (!half_valid[op.a]) {
                const size_t bytes = static_cast<size_t>(m * k) * sizeof(__fp16);
                if (s.half && s.half.length < bytes) {
                    g.release(s.half);
                    s.half = nil;
                }
                if (!s.half) s.half = g.acquire(bytes);
                const uint32_t cnt = u32(m * k);
                id<MTLComputeCommandEncoder> e = compute();
                [e setComputePipelineState:g.half_k];
                [e setBuffer:s.buf offset:0 atIndex:0];
                [e setBuffer:s.half offset:0 atIndex:1];
                [e setBuffer:flag offset:0 atIndex:2];
                [e setBytes:&cnt length:sizeof cnt atIndex:3];
                dispatch(g.half_k, m * k);
                half_valid[op.a] = true;
            }
            in = s.half;
        } else {
            wb = g.weight_for(W, false);
        }
        end_compute();
        const MPSDataType type = half ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
        const NSUInteger size = half ? sizeof(__fp16) : sizeof(float);
        const MPSDataType w_type = wb.half ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
        const NSUInteger w_size = wb.half ? sizeof(__fp16) : sizeof(float);
        MPSMatrixDescriptor* dx = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k * size dataType:type];
        MPSMatrixDescriptor* dw = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:k * w_size dataType:w_type];
        MPSMatrixDescriptor* dy = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n * sizeof(float)
                                                                       dataType:MPSDataTypeFloat32];
        MPSMatrix* mx = [[MPSMatrix alloc] initWithBuffer:in descriptor:dx];
        MPSMatrix* mw = [[MPSMatrix alloc] initWithBuffer:wb.buf descriptor:dw];
        MPSMatrix* my = [[MPSMatrix alloc] initWithBuffer:slot(op.c).buf descriptor:dy];
        [mm(m, n, k) encodeToCommandBuffer:cmd leftMatrix:mx rightMatrix:mw resultMatrix:my];
        if (op.lbias) {
            if (op.lbias->numel() != n) throw std::invalid_argument("Gpu::Seq: bias does not match the weight");
            const BiasArgs args{u32(m * n), u32(n)};
            id<MTLComputeCommandEncoder> e = compute();
            [e setComputePipelineState:g.bias_k];
            [e setBuffer:slot(op.c).buf offset:0 atIndex:0];
            [e setBuffer:g.weight_buffer(op.lbias->f32(), false) offset:0 atIndex:1];
            [e setBytes:&args length:sizeof args atIndex:2];
            dispatch(g.bias_k, m * n);
        }
    }

    // dx (m, k) = dy (m, n) W, W (n, k).
    void encode_linear_t(const Op& op) {
        const Weight& W = *op.lw;
        const int64_t m = op.rows, n = W.shape[0], k = W.shape[1];
        need(op.a, m * n);
        need(op.c, m * k);
        if (m == 0 || k == 0) return;
        if (n == 0) throw std::logic_error("Gpu::Seq: linear_t with an empty input");
        end_compute();
        MPSMatrixDescriptor* ddy = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n * sizeof(float)
                                                                        dataType:MPSDataTypeFloat32];
        const Gpu::Impl::WBuf wb = g.weight_for(W, false);
        MPSMatrixDescriptor* dw =
            [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                  columns:k
                                                 rowBytes:k * (wb.half ? sizeof(__fp16) : sizeof(float))
                                                 dataType:wb.half ? MPSDataTypeFloat16 : MPSDataTypeFloat32];
        MPSMatrixDescriptor* ddx = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k * sizeof(float)
                                                                        dataType:MPSDataTypeFloat32];
        MPSMatrix* mdy = [[MPSMatrix alloc] initWithBuffer:slot(op.a).buf descriptor:ddy];
        MPSMatrix* mw = [[MPSMatrix alloc] initWithBuffer:wb.buf descriptor:dw];
        MPSMatrix* mdx = [[MPSMatrix alloc] initWithBuffer:slot(op.c).buf descriptor:ddx];
        [mm(m, k, n, false) encodeToCommandBuffer:cmd leftMatrix:mdy rightMatrix:mw resultMatrix:mdx];
    }

    void encode(const Op& op, bool fp16) {
        switch (op.kind) {
            case Op::Linear:
                encode_linear(op, fp16);
                break;
            case Op::LinearT:
                encode_linear_t(op);
                break;
            case Op::NormBwd: {
                const int64_t d = op.W->numel();
                need(op.a, op.rows * d);
                need(op.b, op.rows * d);
                need(op.c, op.rows * d);
                if (op.rows == 0 || d == 0) break;
                const NormArgs args{u32(d), op.eps, op.layernorm ? 1u : 0u, 0u};
                id<MTLBuffer> wb = g.weight_buffer(*op.W, false);
                id<MTLComputeCommandEncoder> e = compute();
                [e setComputePipelineState:g.norm_bwd_k];
                [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                [e setBuffer:slot(op.b).buf offset:0 atIndex:1];
                [e setBuffer:wb offset:0 atIndex:2];
                [e setBuffer:slot(op.c).buf offset:0 atIndex:3];
                [e setBytes:&args length:sizeof args atIndex:4];
                const NSUInteger tg = std::min<NSUInteger>(256, g.norm_bwd_k.maxTotalThreadsPerThreadgroup);
                [e dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(op.rows), 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                break;
            }
            case Op::ActBwd:
            case Op::GluBwd: {
                const bool glu = op.kind == Op::GluBwd;
                need(op.a, op.n);
                need(op.b, op.n);
                need(op.c, op.n);
                if (glu) {
                    need(op.d, op.n);
                    need(op.e, op.n);
                }
                if (op.n == 0) break;
                const ActArgs args{u32(op.n), static_cast<int32_t>(op.act), std::sqrt(2.f / 3.14159265358979323846f)};
                id<MTLComputeCommandEncoder> e = compute();
                [e setComputePipelineState:glu ? g.glu_bwd_k : g.act_bwd_k];
                [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                [e setBuffer:slot(op.b).buf offset:0 atIndex:1];
                [e setBuffer:slot(op.c).buf offset:0 atIndex:2];
                if (glu) {
                    [e setBuffer:slot(op.d).buf offset:0 atIndex:3];
                    [e setBuffer:slot(op.e).buf offset:0 atIndex:4];
                }
                [e setBytes:&args length:sizeof args atIndex:glu ? 5 : 3];
                dispatch(glu ? g.glu_bwd_k : g.act_bwd_k, op.n);
                break;
            }
            case Op::Norm: {
                const int64_t d = op.W->numel();
                need(op.a, op.rows * d);
                need(op.c, op.rows * d);
                if (op.rows == 0 || d == 0) break;
                if (op.bias && op.bias->numel() != d) throw std::invalid_argument("Gpu::Seq: norm bias does not match");
                const NormArgs args{u32(d), op.eps, op.layernorm ? 1u : 0u, op.bias ? 1u : 0u};
                id<MTLBuffer> wb = g.weight_buffer(*op.W, false);
                id<MTLComputeCommandEncoder> e = compute();
                [e setComputePipelineState:g.norm_k];
                [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                [e setBuffer:slot(op.c).buf offset:0 atIndex:1];
                [e setBuffer:wb offset:0 atIndex:2];
                [e setBuffer:op.bias ? g.weight_buffer(*op.bias, false) : wb offset:0 atIndex:3];
                [e setBytes:&args length:sizeof args atIndex:4];
                const NSUInteger tg = std::min<NSUInteger>(256, g.norm_k.maxTotalThreadsPerThreadgroup);
                [e dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(op.rows), 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                break;
            }
            case Op::Act:
            case Op::Glu: {
                need(op.a, op.n);
                if (op.kind == Op::Glu) need(op.b, op.n);
                if (op.n == 0) break;
                const ActArgs args{u32(op.n), static_cast<int32_t>(op.act), std::sqrt(2.f / 3.14159265358979323846f)};
                id<MTLComputeCommandEncoder> e = compute();
                [e setComputePipelineState:op.kind == Op::Act ? g.act_k : g.glu_k];
                [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                if (op.kind == Op::Glu) [e setBuffer:slot(op.b).buf offset:0 atIndex:1];
                [e setBytes:&args length:sizeof args atIndex:op.kind == Op::Act ? 1 : 2];
                dispatch(op.kind == Op::Act ? g.act_k : g.glu_k, op.n);
                break;
            }
            case Op::KvAppend: {  // rows = rows, heads, D, n = cap, rot = start
                const int64_t n = op.rows * op.heads * op.D;
                need(op.a, n);
                need(op.c, op.heads * op.n * op.D);
                if (op.rot + op.rows > op.n) throw std::logic_error("Gpu::Seq: kv_append past the cache's capacity");
                if (n == 0) break;
                const KvArgs args{u32(op.rows), u32(op.heads), u32(op.D), u32(op.n), u32(op.rot)};
                id<MTLComputeCommandEncoder> e = compute();
                [e setComputePipelineState:g.kv_append_k];
                [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                [e setBuffer:slot(op.c).buf offset:0 atIndex:1];
                [e setBytes:&args length:sizeof args atIndex:2];
                dispatch(g.kv_append_k, n);
                break;
            }
            case Op::Rope: {
                const int64_t half = op.rot / 2, rows = op.rows * op.heads;
                need(op.a, rows * op.D);
                need(op.c, rows * op.D);
                need(op.d, std::max<int64_t>(half, 1));
                need(op.e, 4 * op.rows);
                const int64_t n_rot = rows * half, n_pass = rows * (op.D - op.rot);
                if (n_rot) {
                    const RopeArgs args{u32(op.heads), u32(op.D), u32(half), u32(n_rot), op.inverse ? 1u : 0u, op.scale};
                    id<MTLComputeCommandEncoder> e = compute();
                    [e setComputePipelineState:g.rope_k];
                    [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                    [e setBuffer:slot(op.c).buf offset:0 atIndex:1];
                    [e setBuffer:slot(op.d).buf offset:0 atIndex:2];
                    [e setBuffer:slot(op.e).buf offset:0 atIndex:3];
                    [e setBytes:&args length:sizeof args atIndex:4];
                    dispatch(g.rope_k, n_rot);
                }
                if (n_pass) {
                    const PassArgs args{u32(op.D), u32(op.rot), u32(n_pass)};
                    id<MTLComputeCommandEncoder> e = compute();
                    [e setComputePipelineState:g.rope_pass_k];
                    [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                    [e setBuffer:slot(op.c).buf offset:0 atIndex:1];
                    [e setBytes:&args length:sizeof args atIndex:2];
                    dispatch(g.rope_pass_k, n_pass);
                }
                break;
            }
            case Op::Attention: {
                const Gpu::Seq::AttnShape& A = op.attn;
                if (A.D <= 0 || A.D > kMaxHeadDim)
                    throw std::invalid_argument("Gpu::Seq: attention needs a head size of 1 to " + std::to_string(kMaxHeadDim));
                if (A.Hkv <= 0 || A.H % A.Hkv) throw std::invalid_argument("Gpu::Seq: attention heads must group evenly");
                need(op.a, A.rows * A.H * A.D);
                need(op.b, A.rows * A.Hkv * A.D);
                need(op.d, A.rows * A.Hkv * A.D);
                need(op.c, A.rows * A.H * A.D);
                need(op.e, std::max<int64_t>(A.Hkv * A.Tp * A.D, 1));
                need(op.f, std::max<int64_t>(A.Hkv * A.Tp * A.D, 1));
                need(op.g, 4 * A.rows);
                if (A.has_slopes) need(op.h, A.H);
                if (op.j >= 0) need(op.j, A.rows * A.H);
                if (A.rows == 0) break;
                const AttnArgs args{u32(A.H), u32(A.Hkv), u32(A.D), u32(A.Tp), u32(A.window), A.has_slopes ? 1u : 0u, op.j >= 0 ? 1u : 0u, A.scale, A.cap};
                // Few rows over more keys than one chunk, without lse (only gradient passes ask for it):
                // split the keys across threadgroups (flash-decoding), then merge.
                const int64_t max_keys = A.Tp + A.rows;  // no row attends to more
                const int64_t nsplit = (max_keys + kSplitKeys - 1) / kSplitKeys;
                if (op.j < 0 && A.rows <= 8 && nsplit > 1) {
                    const size_t bytes = static_cast<size_t>(A.rows * A.H * nsplit * (A.D + 2)) * sizeof(float);
                    if (split_bytes < bytes) {
                        if (split) g.release(split);
                        split = g.acquire(bytes);
                        split_bytes = bytes;
                    }
                    const uint32_t ns = u32(nsplit);
                    id<MTLComputeCommandEncoder> e = compute();
                    [e setComputePipelineState:g.attn_split_k];
                    [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                    [e setBuffer:slot(op.b).buf offset:0 atIndex:1];
                    [e setBuffer:slot(op.d).buf offset:0 atIndex:2];
                    [e setBuffer:split offset:0 atIndex:3];
                    [e setBuffer:slot(op.e).buf offset:0 atIndex:4];
                    [e setBuffer:slot(op.f).buf offset:0 atIndex:5];
                    [e setBuffer:slot(op.g).buf offset:0 atIndex:6];
                    [e setBuffer:slot(A.has_slopes ? op.h : op.g).buf offset:0 atIndex:7];
                    [e setBytes:&args length:sizeof args atIndex:8];
                    [e setBytes:&ns length:sizeof ns atIndex:9];
                    const NSUInteger tg = std::min<NSUInteger>(256, g.attn_split_k.maxTotalThreadsPerThreadgroup);
                    [e dispatchThreadgroups:MTLSizeMake(NSUInteger(u32(A.rows * A.H * nsplit)), 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    id<MTLComputeCommandEncoder> e2 = compute();
                    [e2 setComputePipelineState:g.attn_combine_k];
                    [e2 setBuffer:split offset:0 atIndex:0];
                    [e2 setBuffer:slot(op.c).buf offset:0 atIndex:1];
                    [e2 setBytes:&args length:sizeof args atIndex:2];
                    [e2 setBytes:&ns length:sizeof ns atIndex:3];
                    const NSUInteger tg2 = std::min<NSUInteger>(128, g.attn_combine_k.maxTotalThreadsPerThreadgroup);
                    [e2 dispatchThreadgroups:MTLSizeMake(NSUInteger(u32(A.rows * A.H)), 1, 1) threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];
                    break;
                }
                const bool tiled = A.blocks > 0 && A.D <= 128 && A.D % 8 == 0;
                if (tiled) need(op.i, 4 * A.blocks);
                id<MTLComputeCommandEncoder> e = compute();
                [e setComputePipelineState:tiled ? g.attn_mma_k : g.attn_k];
                [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                [e setBuffer:slot(op.b).buf offset:0 atIndex:1];
                [e setBuffer:slot(op.d).buf offset:0 atIndex:2];
                [e setBuffer:slot(op.c).buf offset:0 atIndex:3];
                [e setBuffer:slot(op.e).buf offset:0 atIndex:4];
                [e setBuffer:slot(op.f).buf offset:0 atIndex:5];
                [e setBuffer:slot(tiled ? op.i : op.g).buf offset:0 atIndex:6];
                [e setBuffer:slot(A.has_slopes ? op.h : op.g).buf offset:0 atIndex:7];
                [e setBytes:&args length:sizeof args atIndex:8];
                [e setBuffer:slot(op.j >= 0 ? op.j : op.c).buf offset:0 atIndex:9];
                if (tiled) {
                    if (g.attn_mma_k.maxTotalThreadsPerThreadgroup < 128) throw std::runtime_error("GPU: the attention kernel needs 128 threads");
                    [e dispatchThreadgroups:MTLSizeMake(NSUInteger(u32(A.blocks * A.H)), 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    break;
                }
                const NSUInteger tg = std::min<NSUInteger>(128, g.attn_k.maxTotalThreadsPerThreadgroup);
                [e dispatchThreadgroups:MTLSizeMake(NSUInteger(u32(A.rows * A.H)), 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                break;
            }
            case Op::Pattern:
            case Op::Apply: {
                const Gpu::Seq::AttnShape& A = op.attn;
                if (A.Hkv <= 0 || A.H % A.Hkv) throw std::invalid_argument("Gpu::Seq: attention heads must group evenly");
                const int64_t T = A.rows;
                const AttnArgs args{u32(A.H), u32(A.Hkv), u32(A.D), 0u, u32(A.window), A.has_slopes ? 1u : 0u, 0u, A.scale, A.cap};
                const uint32_t t32 = u32(T);
                id<MTLComputeCommandEncoder> e = compute();
                if (op.kind == Op::Pattern) {  // q a, k b, lse f, slopes h, P c
                    need(op.a, T * A.H * A.D);
                    need(op.b, T * A.Hkv * A.D);
                    need(op.f, T * A.H);
                    if (A.has_slopes) need(op.h, A.H);
                    need(op.c, A.H * T * T);
                    if (T == 0) break;
                    [e setComputePipelineState:g.attn_pattern_k];
                    [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                    [e setBuffer:slot(op.b).buf offset:0 atIndex:1];
                    [e setBuffer:slot(op.f).buf offset:0 atIndex:2];
                    [e setBuffer:slot(A.has_slopes ? op.h : op.f).buf offset:0 atIndex:3];
                    [e setBytes:&args length:sizeof args atIndex:4];
                    [e setBuffer:slot(op.c).buf offset:0 atIndex:5];
                    [e setBytes:&t32 length:sizeof t32 atIndex:6];
                    dispatch(g.attn_pattern_k, A.H * T * T);
                } else {  // P a, v d, z c
                    need(op.a, A.H * T * T);
                    need(op.d, T * A.Hkv * A.D);
                    need(op.c, T * A.H * A.D);
                    if (T == 0) break;
                    [e setComputePipelineState:g.attn_apply_k];
                    [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                    [e setBuffer:slot(op.d).buf offset:0 atIndex:1];
                    [e setBuffer:slot(op.c).buf offset:0 atIndex:2];
                    [e setBytes:&args length:sizeof args atIndex:3];
                    [e setBytes:&t32 length:sizeof t32 atIndex:4];
                    dispatch(g.attn_apply_k, T * A.H * A.D);
                }
                break;
            }
            case Op::AttentionBwd: {
                const Gpu::Seq::AttnShape& A = op.attn;
                if (A.D <= 0 || A.D > kMaxHeadDim)
                    throw std::invalid_argument("Gpu::Seq: attention needs a head size of 1 to " + std::to_string(kMaxHeadDim));
                if (A.Hkv <= 0 || A.H % A.Hkv) throw std::invalid_argument("Gpu::Seq: attention heads must group evenly");
                const int64_t nq = A.rows * A.H * A.D, nkv = A.rows * A.Hkv * A.D;
                need(op.a, nq);
                need(op.b, nkv);
                need(op.d, nkv);
                need(op.c, nq);
                need(op.e, nq);
                need(op.f, A.rows * A.H);
                need(op.g, 4 * A.rows);
                if (A.has_slopes) need(op.h, A.H);
                need(op.more[0], A.rows * A.H);
                need(op.more[1], nq);
                need(op.more[2], nkv);
                need(op.more[3], nkv);
                if (A.rows == 0) break;
                {
                    const uint32_t args[2] = {u32(A.D), u32(A.rows * A.H)};
                    id<MTLComputeCommandEncoder> e = compute();
                    [e setComputePipelineState:g.attn_delta_k];
                    [e setBuffer:slot(op.e).buf offset:0 atIndex:0];
                    [e setBuffer:slot(op.c).buf offset:0 atIndex:1];
                    [e setBuffer:slot(op.more[0]).buf offset:0 atIndex:2];
                    [e setBytes:args length:sizeof args atIndex:3];
                    dispatch(g.attn_delta_k, A.rows * A.H);
                }
                const AttnArgs args{u32(A.H), u32(A.Hkv), u32(A.D), 0u, u32(A.window), A.has_slopes ? 1u : 0u, 0u, A.scale, A.cap};
                auto common = [&](id<MTLComputeCommandEncoder> e) {
                    [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                    [e setBuffer:slot(op.b).buf offset:0 atIndex:1];
                    [e setBuffer:slot(op.d).buf offset:0 atIndex:2];
                    [e setBuffer:slot(op.e).buf offset:0 atIndex:3];
                    [e setBuffer:slot(op.f).buf offset:0 atIndex:4];
                    [e setBuffer:slot(op.more[0]).buf offset:0 atIndex:5];
                    [e setBuffer:slot(op.g).buf offset:0 atIndex:6];
                    [e setBuffer:slot(A.has_slopes ? op.h : op.g).buf offset:0 atIndex:7];
                    [e setBytes:&args length:sizeof args atIndex:8];
                };
                {
                    id<MTLComputeCommandEncoder> e = compute();
                    [e setComputePipelineState:g.attn_bwd_dq_k];
                    common(e);
                    [e setBuffer:slot(op.more[1]).buf offset:0 atIndex:9];
                    const NSUInteger tg = std::min<NSUInteger>(128, g.attn_bwd_dq_k.maxTotalThreadsPerThreadgroup);
                    [e dispatchThreadgroups:MTLSizeMake(NSUInteger(u32(A.rows * A.H)), 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }
                {
                    id<MTLComputeCommandEncoder> e = compute();
                    [e setComputePipelineState:g.attn_bwd_dkv_k];
                    common(e);
                    [e setBuffer:slot(op.more[2]).buf offset:0 atIndex:9];
                    [e setBuffer:slot(op.more[3]).buf offset:0 atIndex:10];
                    const NSUInteger tg = std::min<NSUInteger>(128, g.attn_bwd_dkv_k.maxTotalThreadsPerThreadgroup);
                    [e dispatchThreadgroups:MTLSizeMake(NSUInteger(u32(A.rows * A.Hkv)), 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }
                break;
            }
            case Op::Add:
            case Op::Copy: {
                need(op.a, op.n);
                need(op.c, op.n);
                if (op.kind == Op::Add) need(op.b, op.n);
                if (op.n == 0) break;
                const uint32_t cnt = u32(op.n);
                id<MTLComputeCommandEncoder> e = compute();
                if (op.kind == Op::Add) {
                    [e setComputePipelineState:g.add_k];
                    [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                    [e setBuffer:slot(op.b).buf offset:0 atIndex:1];
                    [e setBuffer:slot(op.c).buf offset:0 atIndex:2];
                    [e setBytes:&cnt length:sizeof cnt atIndex:3];
                    dispatch(g.add_k, op.n);
                } else {
                    [e setComputePipelineState:g.copy_k];
                    [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                    [e setBuffer:slot(op.c).buf offset:0 atIndex:1];
                    [e setBytes:&cnt length:sizeof cnt atIndex:2];
                    dispatch(g.copy_k, op.n);
                }
                break;
            }
        }
        // Whatever the operation wrote no longer matches its fp16 copy.
        if (op.kind == Op::GluBwd) {
            half_valid[op.d] = false;
            half_valid[op.e] = false;
        } else {
            half_valid[op.kind == Op::Act || op.kind == Op::Glu ? op.a : op.c] = false;
        }
    }

    // One attempt; true if fp16 was asked for and an input did not fit.
    bool attempt(bool fp16) {
        @autoreleasepool {
            *static_cast<uint32_t*>(flag.contents) = 0;
            cmd = [g.queue commandBuffer];
            half_valid.assign(slots.size(), false);
            try {
                for (const Op& op : ops) encode(op, fp16);
                end_compute();
            } catch (...) {
                end_compute();
                cmd = nil;
                throw;
            }
            [cmd commit];
            [cmd waitUntilCompleted];
            const bool ok = cmd.status == MTLCommandBufferStatusCompleted;
            cmd = nil;
            if (!ok) throw std::runtime_error("GPU: a batched layer failed");
            return fp16 && *static_cast<uint32_t*>(flag.contents) != 0;
        }
    }
};

Gpu::Seq::Seq(Gpu& gpu) : impl_(new Impl(*gpu.impl_)) {
    gpu.impl_->load_kernels();
    impl_->flag = gpu.impl_->acquire(sizeof(uint32_t));
}

Gpu::Seq::~Seq() {
    Impl& s = *impl_;
    for (Impl::Slot& slot : s.slots) {
        if (slot.buf) s.g.release(slot.buf);
        if (slot.half) s.g.release(slot.half);
    }
    if (s.flag) s.g.release(s.flag);
    if (s.split) s.g.release(s.split);
}

float* Gpu::Seq::reserve(int slot, int64_t floats) {
    if (slot < 0) throw std::invalid_argument("Gpu::Seq: negative slot");
    Impl& s = *impl_;
    if (static_cast<size_t>(slot) >= s.slots.size()) s.slots.resize(static_cast<size_t>(slot) + 1);
    Impl::Slot& sl = s.slots[slot];
    if (!sl.buf || sl.floats < floats) {
        if (sl.buf) s.g.release(sl.buf);
        sl.buf = s.g.acquire(static_cast<size_t>(std::max<int64_t>(floats, 1)) * sizeof(float));
        sl.floats = floats;
    }
    return static_cast<float*>(sl.buf.contents);
}

float* Gpu::Seq::data(int slot) { return static_cast<float*>(impl_->slot(slot).buf.contents); }

void Gpu::Seq::linear(int x, int y, int64_t rows, const Weight& W, const Weight* b) {
    if (W.dim() != 2) throw std::invalid_argument("linear weight must be 2-D");
    if (b && b->is_fp16()) throw std::invalid_argument("Gpu::Seq: linear biases must be float32");
    Impl::Op op{Impl::Op::Linear};
    op.a = x;
    op.c = y;
    op.rows = rows;
    op.lw = &W;
    op.lbias = b;
    impl_->ops.push_back(op);
}

void Gpu::Seq::norm(int x, int y, int64_t rows, const Tensor& w, const Tensor* b, bool layernorm, float eps) {
    if (b && !layernorm) throw std::invalid_argument("Gpu::Seq: rmsnorm has no bias");
    Impl::Op op{Impl::Op::Norm};
    op.a = x;
    op.c = y;
    op.rows = rows;
    op.W = &w;
    op.bias = b;
    op.layernorm = layernorm;
    op.eps = eps;
    impl_->ops.push_back(op);
}

void Gpu::Seq::activate(int x, int64_t n, GpuAct act) {
    Impl::Op op{Impl::Op::Act};
    op.a = x;
    op.n = n;
    op.act = act;
    impl_->ops.push_back(op);
}

void Gpu::Seq::glu(int gate, int up, int64_t n, GpuAct act) {
    Impl::Op op{Impl::Op::Glu};
    op.a = gate;
    op.b = up;
    op.n = n;
    op.act = act;
    impl_->ops.push_back(op);
}

void Gpu::Seq::add(int a, int b, int out, int64_t n) {
    Impl::Op op{Impl::Op::Add};
    op.a = a;
    op.b = b;
    op.c = out;
    op.n = n;
    impl_->ops.push_back(op);
}

void Gpu::Seq::copy(int x, int y, int64_t n) {
    Impl::Op op{Impl::Op::Copy};
    op.a = x;
    op.c = y;
    op.n = n;
    impl_->ops.push_back(op);
}

void Gpu::Seq::linear_t(int dy, int dx, int64_t rows, const Weight& W) {
    if (W.dim() != 2) throw std::invalid_argument("linear weight must be 2-D");
    Impl::Op op{Impl::Op::LinearT};
    op.a = dy;
    op.c = dx;
    op.rows = rows;
    op.lw = &W;
    impl_->ops.push_back(op);
}

void Gpu::Seq::norm_backward(int x, int dy, int dx, int64_t rows, const Tensor& w, bool layernorm, float eps) {
    Impl::Op op{Impl::Op::NormBwd};
    op.a = x;
    op.b = dy;
    op.c = dx;
    op.rows = rows;
    op.W = &w;
    op.layernorm = layernorm;
    op.eps = eps;
    impl_->ops.push_back(op);
}

void Gpu::Seq::kv_append(int src, int dst, int64_t rows, int64_t heads, int64_t D, int64_t cap, int64_t start) {
    if (src == dst) throw std::invalid_argument("Gpu::Seq: kv_append into the slot it reads");
    Impl::Op op{Impl::Op::KvAppend};
    op.a = src;
    op.c = dst;
    op.rows = rows;
    op.heads = heads;
    op.D = D;
    op.n = cap;
    op.rot = start;
    impl_->ops.push_back(op);
}

void Gpu::Seq::rope(int x, int y, int64_t rows, int64_t heads, int64_t D, int64_t rot, int inv_freq, int meta, bool inverse, float scale) {
    if (rot < 0 || rot % 2 || rot > D) throw std::invalid_argument("Gpu::Seq: RoPE rotates an even number of values, at most D");
    if (x == y) throw std::invalid_argument("Gpu::Seq: RoPE works out of place");
    Impl::Op op{Impl::Op::Rope};
    op.a = x;
    op.c = y;
    op.d = inv_freq;
    op.e = meta;
    op.rows = rows;
    op.heads = heads;
    op.D = D;
    op.rot = rot;
    op.inverse = inverse;
    op.scale = scale;
    impl_->ops.push_back(op);
}

void Gpu::Seq::attention(int q, int k, int v, int z, int pk, int pv, int meta, int slopes, const AttnShape& shape, int blk, int lse) {
    if (z == q || z == k || z == v) throw std::invalid_argument("Gpu::Seq: attention writes z to a slot of its own");
    if (lse >= 0 && (lse == q || lse == k || lse == v || lse == z)) throw std::invalid_argument("Gpu::Seq: attention writes lse to a slot of its own");
    Impl::Op op{Impl::Op::Attention};
    op.a = q;
    op.b = k;
    op.c = z;
    op.d = v;
    op.e = pk;
    op.f = pv;
    op.g = meta;
    op.h = slopes;
    op.i = blk;
    op.j = lse;
    op.attn = shape;
    impl_->ops.push_back(op);
}

void Gpu::Seq::attention_pattern(int q, int k, int lse, int slopes, int P, const AttnShape& shape) {
    if (P == q || P == k || P == lse) throw std::invalid_argument("Gpu::Seq: the pattern goes to a slot of its own");
    Impl::Op op{Impl::Op::Pattern};
    op.a = q;
    op.b = k;
    op.f = lse;
    op.h = slopes;
    op.c = P;
    op.attn = shape;
    impl_->ops.push_back(op);
}

void Gpu::Seq::attention_apply(int P, int v, int z, const AttnShape& shape) {
    if (z == P || z == v) throw std::invalid_argument("Gpu::Seq: attention writes z to a slot of its own");
    Impl::Op op{Impl::Op::Apply};
    op.a = P;
    op.d = v;
    op.c = z;
    op.attn = shape;
    impl_->ops.push_back(op);
}

void Gpu::Seq::attention_backward(const AttnSlots& in, int delta, int dq, int dk, int dv, const AttnShape& shape) {
    for (int o : {delta, dq, dk, dv})
        for (int i : {in.q, in.k, in.v, in.z, in.dz, in.lse, in.meta, in.slopes})
            if (o == i) throw std::invalid_argument("Gpu::Seq: attention's backward pass writes to slots of its own");
    if (delta == dq || delta == dk || delta == dv || dq == dk || dq == dv || dk == dv)
        throw std::invalid_argument("Gpu::Seq: attention's backward pass writes four different slots");
    if (shape.Tp != 0) throw std::invalid_argument("Gpu::Seq: attention's backward pass takes no earlier positions");
    Impl::Op op{Impl::Op::AttentionBwd};
    op.a = in.q;
    op.b = in.k;
    op.d = in.v;
    op.c = in.z;
    op.e = in.dz;
    op.f = in.lse;
    op.g = in.meta;
    op.h = in.slopes;
    op.more[0] = delta;
    op.more[1] = dq;
    op.more[2] = dk;
    op.more[3] = dv;
    op.attn = shape;
    impl_->ops.push_back(op);
}

void Gpu::Seq::act_backward(int dpost, int pre, int dpre, int64_t n, GpuAct act) {
    Impl::Op op{Impl::Op::ActBwd};
    op.a = dpost;
    op.b = pre;
    op.c = dpre;
    op.n = n;
    op.act = act;
    impl_->ops.push_back(op);
}

void Gpu::Seq::glu_backward(int dpost, int pre, int up, int dpre, int dup, int64_t n, GpuAct act) {
    Impl::Op op{Impl::Op::GluBwd};
    op.a = dpost;
    op.b = pre;
    op.c = up;
    op.d = dpre;
    op.e = dup;
    op.n = n;
    op.act = act;
    impl_->ops.push_back(op);
}

void Gpu::Seq::run(bool fp16) {
    Impl& s = *impl_;
    if (s.ops.empty()) return;
    struct Clear {
        std::vector<Impl::Op>& ops;
        ~Clear() { ops.clear(); }
    } clear{s.ops};
    if (s.attempt(fp16)) s.attempt(false);
}

// --- Gpu::Batch -------------------------------------------------------------------------

struct Gpu::Batch::Impl {
    Gpu::Impl& g;
    struct Buf {
        id<MTLBuffer> buf = nil;
        bool half = false;   // fp16 values (a weight)
        bool owned = false;  // from the pool (released at the end), not a weight's own memory
    };
    struct Op {
        enum Kind { MatMul, BlockSumsq, Argmax, ToHalf } kind;
        Mat a{0, 0, 0}, b{0, 0, 0}, c{0, 0, 0};
        int64_t m = 0, n = 0, k = 0, bm = 0, bn = 0;
        bool trans_b = false;
        int out = 0;
        int64_t out_offset = 0;
        double flops = 0;
    };
    std::vector<Buf> bufs;
    id<MTLBuffer> flag = nil;  // set by half_k when a value doesn't fit fp16
    std::vector<std::vector<Op>> groups;
    std::vector<Op> current;
    std::map<std::tuple<int64_t, int64_t, int64_t, bool>, MPSMatrixMultiplication*> mms;

    explicit Impl(Gpu::Impl& gpu) : g(gpu) {}

    Buf& at(int i) {
        if (i < 0 || static_cast<size_t>(i) >= bufs.size() || !bufs[i].buf)
            throw std::logic_error("Gpu::Batch: no buffer " + std::to_string(i));
        return bufs[i];
    }
    static uint32_t u32(int64_t n) {
        if (n < 0 || n > static_cast<int64_t>(UINT32_MAX)) throw std::length_error("Gpu::Batch: operation too large");
        return static_cast<uint32_t>(n);
    }
    // Results go in the batch's own buffers, never in a model's weights (which the GPU uses in place).
    void writable(int i) {
        if (!at(i).owned) throw std::invalid_argument("Gpu::Batch: results go in a buffer(), not in a weight");
    }
    // Checks that rows x cols at m.offset with stride m.ld fit the buffer.
    void check(const Mat& m, int64_t rows, int64_t cols) {
        Buf& b = at(m.buf);
        const int64_t size = b.half ? 2 : 4;
        if (rows < 0 || cols < 0 || m.offset < 0 || (rows && m.ld < cols))
            throw std::invalid_argument("Gpu::Batch: bad matrix layout");
        if (rows && cols && static_cast<uint64_t>((m.offset + (rows - 1) * m.ld + cols) * size) > b.buf.length)
            throw std::out_of_range("Gpu::Batch: matrix runs past its buffer");
    }
    MPSMatrix* matrix(const Mat& m, int64_t rows, int64_t cols) {
        Buf& b = at(m.buf);
        const NSUInteger size = b.half ? sizeof(__fp16) : sizeof(float);
        MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(rows)
                                                                       columns:static_cast<NSUInteger>(cols)
                                                                      rowBytes:static_cast<NSUInteger>(m.ld) * size
                                                                      dataType:b.half ? MPSDataTypeFloat16 : MPSDataTypeFloat32];
        return [[MPSMatrix alloc] initWithBuffer:b.buf offset:static_cast<NSUInteger>(m.offset) * size descriptor:d];
    }
    MPSMatrixMultiplication* mm(int64_t m, int64_t n, int64_t k, bool trans_b) {
        const auto key = std::make_tuple(m, n, k, trans_b);
        auto it = mms.find(key);
        if (it != mms.end()) return it->second;
        MPSMatrixMultiplication* p = [[MPSMatrixMultiplication alloc] initWithDevice:g.device
                                                                       transposeLeft:NO
                                                                      transposeRight:trans_b ? YES : NO
                                                                          resultRows:static_cast<NSUInteger>(m)
                                                                       resultColumns:static_cast<NSUInteger>(n)
                                                                     interiorColumns:static_cast<NSUInteger>(k)
                                                                               alpha:1.0
                                                                                beta:0.0];
        mms.emplace(key, p);
        return p;
    }

    void encode(id<MTLCommandBuffer> cmd, const Op& op) {
        if (op.kind == Op::ToHalf) {
            if (op.n == 0) return;
            id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            const uint32_t cnt = u32(op.n);
            [e setComputePipelineState:g.half_k];
            [e setBuffer:at(op.c.buf).buf offset:0 atIndex:0];
            [e setBuffer:at(op.out).buf offset:0 atIndex:1];
            [e setBuffer:flag offset:0 atIndex:2];
            [e setBytes:&cnt length:sizeof cnt atIndex:3];
            const NSUInteger tg = std::min<NSUInteger>(256, g.half_k.maxTotalThreadsPerThreadgroup);
            [e dispatchThreads:MTLSizeMake(NSUInteger(cnt), 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [e endEncoding];
            return;
        }
        if (op.kind == Op::MatMul) {
            if (op.m == 0 || op.n == 0) return;
            MPSMatrix* A = matrix(op.a, op.m, op.k);
            MPSMatrix* B = op.trans_b ? matrix(op.b, op.n, op.k) : matrix(op.b, op.k, op.n);
            MPSMatrix* C = matrix(op.c, op.m, op.n);
            [mm(op.m, op.n, op.k, op.trans_b) encodeToCommandBuffer:cmd leftMatrix:A rightMatrix:B resultMatrix:C];
            return;
        }
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        Buf& c = at(op.c.buf);
        Buf& out = at(op.out);
        if (op.kind == Op::BlockSumsq) {
            const uint32_t nbr = u32(op.m / op.bm), nbc = u32(op.n / op.bn);
            struct { uint32_t ld, bm, bn, nbc; } args{u32(op.c.ld), u32(op.bm), u32(op.bn), nbc};
            [e setComputePipelineState:g.block_sumsq_k];
            [e setBuffer:c.buf offset:static_cast<NSUInteger>(op.c.offset) * sizeof(float) atIndex:0];
            [e setBuffer:out.buf offset:static_cast<NSUInteger>(op.out_offset) * sizeof(float) atIndex:1];
            [e setBytes:&args length:sizeof args atIndex:2];
            const NSUInteger tg = std::min<NSUInteger>(256, g.block_sumsq_k.maxTotalThreadsPerThreadgroup);
            if (nbr && nbc)
                [e dispatchThreadgroups:MTLSizeMake(NSUInteger(nbr) * nbc, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        } else {
            struct { uint32_t n, ld; } args{u32(op.n), u32(op.c.ld)};
            [e setComputePipelineState:g.row_argmax_k];
            [e setBuffer:c.buf offset:static_cast<NSUInteger>(op.c.offset) * sizeof(float) atIndex:0];
            [e setBuffer:out.buf offset:static_cast<NSUInteger>(op.out_offset) * sizeof(uint32_t) atIndex:1];
            [e setBytes:&args length:sizeof args atIndex:2];
            const NSUInteger tg = std::min<NSUInteger>(256, g.row_argmax_k.maxTotalThreadsPerThreadgroup);
            if (op.m && op.n)
                [e dispatchThreadgroups:MTLSizeMake(NSUInteger(op.m), 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }
        [e endEncoding];
    }

    // run_chunks' Metal side: a chunk is one command buffer.
    struct Backend : ChunkBackend {
        Impl& s;
        std::vector<id<MTLCommandBuffer>> cmds;
        explicit Backend(Impl& impl) : s(impl) {}
        size_t submit(const Chunk& c) override {
            id<MTLCommandBuffer> cmd = [s.g.queue commandBuffer];
            for (size_t i = c.first; i < c.last; ++i)
                for (const Op& op : s.groups[i]) s.encode(cmd, op);
            [cmd commit];
            cmds.push_back(cmd);
            return cmds.size() - 1;
        }
        bool wait(size_t ticket) override {
            [cmds[ticket] waitUntilCompleted];
            return cmds[ticket].status == MTLCommandBufferStatusCompleted;
        }
    };
};

Gpu::Batch::Batch(Gpu& gpu) : impl_(new Impl(*gpu.impl_)) {
    gpu.impl_->load_kernels();
    impl_->flag = gpu.impl_->acquire(sizeof(uint32_t));
    *static_cast<uint32_t*>(impl_->flag.contents) = 0;
}

Gpu::Batch::~Batch() {
    for (Impl::Buf& b : impl_->bufs)
        if (b.buf && b.owned) impl_->g.release(b.buf);
    if (impl_->flag) impl_->g.release(impl_->flag);
}

int Gpu::Batch::buffer(int64_t count, bool fp16) {
    if (count < 0) throw std::invalid_argument("Gpu::Batch: negative buffer size");
    Impl::Buf b;
    b.buf = impl_->g.acquire(static_cast<size_t>(std::max<int64_t>(count, 1)) * (fp16 ? sizeof(__fp16) : sizeof(float)));
    b.half = fp16;
    b.owned = true;
    impl_->bufs.push_back(b);
    return static_cast<int>(impl_->bufs.size() - 1);
}

float* Gpu::Batch::floats(int buf) {
    Impl::Buf& b = impl_->at(buf);
    if (b.half) throw std::logic_error("Gpu::Batch: an fp16 buffer has no float32 values");
    return static_cast<float*>(b.buf.contents);
}

uint16_t* Gpu::Batch::halves(int buf) {
    Impl::Buf& b = impl_->at(buf);
    if (!b.half || !b.owned) throw std::logic_error("Gpu::Batch: halves() is for the batch's own fp16 buffers");
    return static_cast<uint16_t*>(b.buf.contents);
}

bool Gpu::Batch::is_fp16(int buf) { return impl_->at(buf).half; }

bool Gpu::Batch::overflowed() { return *static_cast<uint32_t*>(impl_->flag.contents) != 0; }

void Gpu::Batch::to_half(int src, int dst, int64_t n) {
    Impl& s = *impl_;
    if (s.at(src).half || !s.at(dst).half) throw std::invalid_argument("Gpu::Batch: to_half goes from float32 to fp16");
    s.writable(dst);
    s.check(Mat{src, 0, n}, n ? 1 : 0, n);
    s.check(Mat{dst, 0, n}, n ? 1 : 0, n);
    Impl::Op op{Impl::Op::ToHalf};
    op.c = Mat{src, 0, n};
    op.out = dst;
    op.n = n;
    op.flops = double(n);
    s.current.push_back(op);
}

int Gpu::Batch::weight(const Weight& W, bool fp16) {
    const Gpu::Impl::WBuf w = impl_->g.weight_for(W, fp16);
    Impl::Buf b;
    b.buf = w.buf;
    b.half = w.half;
    impl_->bufs.push_back(b);
    return static_cast<int>(impl_->bufs.size() - 1);
}

void Gpu::Batch::release(int buf) {
    Impl::Buf& b = impl_->at(buf);
    if (b.owned) impl_->g.release(b.buf);
    b = Impl::Buf();
}

void Gpu::Batch::matmul(Mat A, Mat B, Mat C, int64_t m, int64_t n, int64_t k, bool trans_b) {
    Impl& s = *impl_;
    s.check(A, m, k);
    if (trans_b) s.check(B, n, k);
    else s.check(B, k, n);
    s.check(C, m, n);
    s.writable(C.buf);
    if (s.at(C.buf).half) throw std::invalid_argument("Gpu::Batch: multiplies write float32 (to_half converts)");
    Impl::Op op{Impl::Op::MatMul};
    op.a = A;
    op.b = B;
    op.c = C;
    op.m = m;
    op.n = n;
    op.k = k;
    op.trans_b = trans_b;
    op.flops = 2.0 * double(m) * double(n) * double(k);
    s.current.push_back(op);
}

void Gpu::Batch::block_sumsq(Mat C, int64_t rows, int64_t cols, int64_t bm, int64_t bn, int out, int64_t out_offset) {
    Impl& s = *impl_;
    if (bm <= 0 || bn <= 0 || rows % bm || cols % bn) throw std::invalid_argument("Gpu::Batch: blocks must tile the matrix");
    s.check(C, rows, cols);
    s.check(Mat{out, out_offset, (rows / bm) * (cols / bn)}, rows ? 1 : 0, (rows / bm) * (cols / bn));
    if (s.at(C.buf).half) throw std::invalid_argument("Gpu::Batch: block sums read float32 values");
    s.writable(out);
    Impl::Op op{Impl::Op::BlockSumsq};
    op.c = C;
    op.m = rows;
    op.n = cols;
    op.bm = bm;
    op.bn = bn;
    op.out = out;
    op.out_offset = out_offset;
    op.flops = 2.0 * double(rows) * double(cols);
    s.current.push_back(op);
}

void Gpu::Batch::row_argmax(Mat C, int64_t rows, int64_t n, int out, int64_t out_offset) {
    Impl& s = *impl_;
    s.check(C, rows, n);
    s.check(Mat{out, out_offset, rows}, rows ? 1 : 0, rows);
    if (s.at(C.buf).half) throw std::invalid_argument("Gpu::Batch: argmaxes read float32 values");
    s.writable(out);
    Impl::Op op{Impl::Op::Argmax};
    op.c = C;
    op.m = rows;
    op.n = n;
    op.out = out;
    op.out_offset = out_offset;
    op.flops = double(rows) * double(n);
    s.current.push_back(op);
}

void Gpu::Batch::end_group() {
    Impl& s = *impl_;
    if (!s.current.empty()) s.groups.push_back(std::move(s.current));
    s.current.clear();
}

void Gpu::Batch::run() {
    Impl& s = *impl_;
    end_group();
    struct Clear {
        std::vector<std::vector<Impl::Op>>& groups;
        ~Clear() { groups.clear(); }
    } clear{s.groups};
    if (s.groups.empty()) return;
    std::vector<double> flops;
    for (const std::vector<Impl::Op>& group : s.groups) {
        double f = 0;
        for (const Impl::Op& op : group) f += op.flops;
        flops.push_back(f);
    }
    *static_cast<uint32_t*>(s.flag.contents) = 0;
    @autoreleasepool {
        Impl::Backend backend(s);
        run_chunks(backend, pack_chunks(flops, gpu_chunk_flops()));
    }
}

}  // namespace detail
}  // namespace tfi

#endif
