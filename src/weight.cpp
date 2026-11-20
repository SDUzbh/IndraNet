#include "tfi/weight.h"

#include <algorithm>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <unistd.h>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

#include "half.h"
#include "ops.h"
#include "tfi/analysis.h"  // num_threads

namespace tfi {

namespace {

// Converted half-precision blocks hold about this many floats (1 MB), per thread.
constexpr int64_t kBlockFloats = int64_t(1) << 18;

std::vector<float>& block_buffer(int64_t floats) {
    thread_local std::vector<float> buf;
    if (static_cast<int64_t>(buf.size()) < floats) buf.resize(static_cast<size_t>(floats));
    return buf;
}

// Up to this many rows, a linear layer multiplies straight from the weights (direct_rows): for one row
// (a generated token) converting fp16 weights to float32 first would cost more than the multiply, and
// the library's matrix multiply gains nothing. More rows use the library (fp16: converted blocks).
constexpr int64_t kDirectRows = 4;

// One weight value as float32: as stored, or converted from fp16 (exactly).
inline float weight_value(float w) { return w; }
inline float weight_value(uint16_t w) { return detail::half_to_float(w); }

// y[r * ldy + j] = x[r] . W[j] for R rows of x (stride in) and weight rows j0..j1-1 of W (float32 or
// fp16, `in` columns), in float32, each weight row read once. fp16 weights take the same steps on
// their (exact) float32 values, so an fp16 model gives bit for bit what float32 weights holding the
// same rounded values give.
template <int R, typename WT>
void direct_rows(const float* x, const WT* W, int64_t in, int64_t j0, int64_t j1, float* y, int64_t ldy) {
    for (int64_t j = j0; j < j1; ++j) {
        const WT* w = W + j * in;
        float sum[R];
        int64_t k = 0;
#if defined(__aarch64__)
        float32x4_t acc[R][2];
        for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = vdupq_n_f32(0.f);
        for (; k + 8 <= in; k += 8) {
            float32x4_t lo, hi;
            if constexpr (std::is_same_v<WT, uint16_t>) {
                const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(w + k));
                lo = vcvt_f32_f16(vget_low_f16(h));
                hi = vcvt_high_f32_f16(h);
            } else {
                lo = vld1q_f32(w + k);
                hi = vld1q_f32(w + k + 4);
            }
            for (int r = 0; r < R; ++r) {
                acc[r][0] = vfmaq_f32(acc[r][0], vld1q_f32(x + r * in + k), lo);
                acc[r][1] = vfmaq_f32(acc[r][1], vld1q_f32(x + r * in + k + 4), hi);
            }
        }
        for (int r = 0; r < R; ++r) sum[r] = vaddvq_f32(vaddq_f32(acc[r][0], acc[r][1]));
#else
        for (int r = 0; r < R; ++r) sum[r] = 0.f;
#endif
        for (; k < in; ++k) {
            const float wk = weight_value(w[k]);
            for (int r = 0; r < R; ++r) {
                // Two statements: the compiler may fuse a multiply-add within one, and might do so for
                // one weight type and not the other, which would break fp16 = rounded float32.
                const float p = x[r * in + k] * wk;
                sum[r] += p;
            }
        }
        for (int r = 0; r < R; ++r) y[r * ldy + j] = sum[r];
    }
}

// direct_rows over every weight row, split into chunks of rows run side by side (macOS: libdispatch's
// threads, which stay alive between calls, so a call costs microseconds, not a thread start each).
template <typename WT>
void linear_direct(const float* x, int64_t n_rows, const WT* W, int64_t in, int64_t out, float* y) {
    auto rows = [&](int64_t j0, int64_t j1) {
        switch (n_rows) {
            case 1: return direct_rows<1>(x, W, in, j0, j1, y, out);
            case 2: return direct_rows<2>(x, W, in, j0, j1, y, out);
            case 3: return direct_rows<3>(x, W, in, j0, j1, y, out);
            default: return direct_rows<4>(x, W, in, j0, j1, y, out);
        }
    };
    // At least 64K weights per chunk; with several threads allowed, 4 chunks per thread, so that faster
    // cores take more of them instead of waiting for the slower ones. One thread allowed: no others.
    const int64_t min_rows = std::max<int64_t>(1, (int64_t(1) << 16) / std::max<int64_t>(in, 1));
    const int64_t threads = num_threads();
    const int64_t chunks = threads <= 1 ? 1 : std::max<int64_t>(1, std::min<int64_t>(4 * threads, out / min_rows));
#if defined(__APPLE__)
    if (chunks > 1) {
        struct Job {
            decltype(rows)* fn;
            int64_t out, chunks;
        } job{&rows, out, chunks};
        dispatch_apply_f(static_cast<size_t>(chunks), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &job,
                         [](void* p, size_t c) {
                             const Job& jb = *static_cast<const Job*>(p);
                             const int64_t i = static_cast<int64_t>(c);
                             (*jb.fn)(jb.out * i / jb.chunks, jb.out * (i + 1) / jb.chunks);
                         });
        return;
    }
#else
    (void)chunks;  // elsewhere one thread: the chunks run in order, as one
#endif
    rows(0, out);
}

}  // namespace

namespace {

// `bytes` (at least 1) in whole pages, starting on a page boundary; capacity receives the size.
std::shared_ptr<void> page_alloc(size_t bytes, size_t& capacity) {
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    capacity = (std::max<size_t>(bytes, 1) + page - 1) / page * page;
    void* p = nullptr;
    if (posix_memalign(&p, page, capacity) != 0) throw std::bad_alloc();
    std::fill(static_cast<char*>(p) + bytes, static_cast<char*>(p) + capacity, char(0));
    return std::shared_ptr<void>(p, [](void* q) { std::free(q); });
}

}  // namespace

Weight::Weight(Tensor t) : shape(t.shape) {
    if (t.dim() != 2) {
        vec_ = std::move(t);
        return;
    }
    const size_t bytes = static_cast<size_t>(t.numel()) * sizeof(float);
    std::shared_ptr<void> mem = page_alloc(bytes, capacity_);
    std::copy(t.data.begin(), t.data.end(), static_cast<float*>(mem.get()));
    mat_ = std::move(mem);
}

int64_t Weight::numel() const { return shape_numel(shape); }

bool Weight::fits_fp16(const Tensor& t) {
    for (float v : t.data)
        if (!detail::fits_half(v)) return false;
    return true;
}

Weight Weight::fp16(const Tensor& t) {
    if (t.dim() != 2) throw std::invalid_argument("only 2-D weights are stored in fp16, got " + shape_str(t.shape));
    const int64_t n = t.numel();
    Weight w;
    std::shared_ptr<void> mem = page_alloc(static_cast<size_t>(n) * sizeof(uint16_t), w.capacity_);
    if (detail::floats_to_halves(t.ptr(), static_cast<uint16_t*>(mem.get()), n) != 0)
        throw std::invalid_argument("weight has values outside fp16's range (|v| > 65504 or not finite)");
    w.shape = t.shape;
    w.mat_ = std::move(mem);
    w.half_ = true;
    return w;
}

Weight Weight::fp16_bits(Shape shape, const uint16_t* bits) {
    const int64_t n = shape_numel(shape);
    if (shape.size() == 2 && !detail::all_finite_halves(bits, n))
        throw std::invalid_argument("fp16 weight has infinite or NaN values");
    return filled(std::move(shape), true, [&](void* dst) { std::copy(bits, bits + n, static_cast<uint16_t*>(dst)); });
}

Weight Weight::filled(Shape shape, bool fp16, const std::function<void(void* dst)>& fill) {
    if (shape.size() != 2) throw std::invalid_argument("filled() makes 2-D weights, got " + shape_str(shape));
    for (int64_t d : shape)
        if (d < 0) throw std::invalid_argument("negative dimension in " + shape_str(shape));
    Weight w;
    std::shared_ptr<void> mem =
        page_alloc(static_cast<size_t>(shape_numel(shape)) * (fp16 ? sizeof(uint16_t) : sizeof(float)), w.capacity_);
    fill(mem.get());
    w.shape = std::move(shape);
    w.mat_ = std::move(mem);
    w.half_ = fp16;
    return w;
}

Weight Weight::borrowed(Shape shape, bool fp16, const void* data, size_t capacity, std::shared_ptr<const void> owner) {
    if (shape.size() != 2) throw std::invalid_argument("borrowed() makes 2-D weights, got " + shape_str(shape));
    for (int64_t d : shape)
        if (d < 0) throw std::invalid_argument("negative dimension in " + shape_str(shape));
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t bytes = static_cast<size_t>(shape_numel(shape)) * (fp16 ? sizeof(uint16_t) : sizeof(float));
    if (reinterpret_cast<uintptr_t>(data) % page != 0 || capacity % page != 0 || capacity < std::max<size_t>(bytes, 1))
        throw std::invalid_argument("borrowed(): the values must start on a page boundary, in whole pages");
    Weight w;
    w.shape = std::move(shape);
    w.mat_ = std::shared_ptr<const void>(std::move(owner), data);  // shares the owner's lifetime
    w.capacity_ = capacity;
    w.half_ = fp16;
    w.borrowed_ = true;
    return w;
}

const Tensor& Weight::f32() const {
    if (dim() == 2)
        throw std::logic_error("f32() is for weights that are not 2-D; read a 2-D weight with rows(), cols(), "
                               "to_tensor() or linear()");
    return vec_;
}

void Weight::read_rows(int64_t begin, int64_t end, float* dst) const {
    if (dim() != 2) throw std::invalid_argument("read_rows needs a 2-D weight, got " + shape_str(shape));
    if (begin < 0 || end < begin || end > shape[0])
        throw std::out_of_range("rows [" + std::to_string(begin) + ", " + std::to_string(end) + ") of " + shape_str(shape));
    const int64_t c = shape[1];
    if (is_fp16()) detail::halves_to_floats(f16() + begin * c, dst, (end - begin) * c);
    else std::copy(f32_data() + begin * c, f32_data() + end * c, dst);
}

Tensor Weight::rows(int64_t begin, int64_t end) const {
    if (dim() != 2) throw std::invalid_argument("rows needs a 2-D weight, got " + shape_str(shape));
    if (begin < 0 || end < begin || end > shape[0])
        throw std::out_of_range("rows [" + std::to_string(begin) + ", " + std::to_string(end) + ") of " + shape_str(shape));
    Tensor out({end - begin, shape[1]});
    read_rows(begin, end, out.ptr());
    return out;
}

Tensor Weight::row(int64_t i) const {
    Tensor r = rows(i, i + 1);
    return std::move(r).reshape({shape[1]});
}

Tensor Weight::cols(int64_t begin, int64_t end) const {
    if (dim() != 2) throw std::invalid_argument("cols needs a 2-D weight, got " + shape_str(shape));
    if (begin < 0 || end < begin || end > shape[1])
        throw std::out_of_range("cols [" + std::to_string(begin) + ", " + std::to_string(end) + ") of " + shape_str(shape));
    const int64_t R = shape[0], C = shape[1], n = end - begin;
    Tensor out({R, n});
    for (int64_t r = 0; r < R; ++r) {
        if (is_fp16()) detail::halves_to_floats(f16() + r * C + begin, out.ptr() + r * n, n);
        else std::copy(f32_data() + r * C + begin, f32_data() + r * C + end, out.ptr() + r * n);
    }
    return out;
}

Tensor Weight::to_tensor() const {
    if (dim() != 2) return vec_;
    Tensor out(shape);
    read_rows(0, shape[0], out.ptr());
    return out;
}

Tensor linear(const Tensor& x, const Weight& W, const Weight* b) {
    if (W.dim() != 2) throw std::invalid_argument("linear weight must be 2-D");
    const int64_t in = W.shape[1], out = W.shape[0];
    if (x.size(-1) != in) throw std::invalid_argument("linear: input " + shape_str(x.shape) + " vs weight " + shape_str(W.shape));
    Shape s(x.shape.begin(), x.shape.end() - 1);
    s.push_back(out);
    Tensor y(std::move(s));
    const int64_t n_rows = in ? x.numel() / in : 0;
    if (n_rows && out) {
        if (n_rows <= kDirectRows) {
            if (W.is_fp16()) linear_direct(x.ptr(), n_rows, W.f16(), in, out, y.ptr());
            else linear_direct(x.ptr(), n_rows, W.f32_data(), in, out, y.ptr());
        } else if (!W.is_fp16()) {
            detail::gemm(/*trans_b=*/true, n_rows, out, in, x.ptr(), in, W.f32_data(), in, y.ptr(), out);
        } else {
            const int64_t block = std::min(out, std::max<int64_t>(1, kBlockFloats / in));
            std::vector<float>& buf = block_buffer(block * in);
            for (int64_t j = 0; j < out; j += block) {
                const int64_t nb = std::min(block, out - j);
                W.read_rows(j, j + nb, buf.data());
                detail::gemm(/*trans_b=*/true, n_rows, nb, in, x.ptr(), in, buf.data(), in, y.ptr() + j, out);
            }
        }
    }
    if (b) {
        const Tensor& bias = b->f32();
        if (bias.numel() != out) throw std::invalid_argument("linear: bias does not match the weight");
        for (int64_t r = 0; r < n_rows; ++r)
            for (int64_t j = 0; j < out; ++j) y.data[r * out + j] += bias.data[j];
    }
    return y;
}

Tensor matmul(const Tensor& A, const Weight& W) {
    if (A.dim() != 2 || W.dim() != 2) throw std::invalid_argument("matmul expects 2-D tensors");
    const int64_t m = A.shape[0], n = A.shape[1], k = W.shape[1];
    if (n != W.shape[0]) throw std::invalid_argument("matmul shape mismatch " + shape_str(A.shape) + " @ " + shape_str(W.shape));
    Tensor C({m, k});
    if (!m || !k || !n) return C;
    if (!W.is_fp16()) {
        detail::gemm(/*trans_b=*/false, m, k, n, A.ptr(), n, W.f32_data(), k, C.ptr(), k);
        return C;
    }
    const int64_t block = std::min(n, std::max<int64_t>(1, kBlockFloats / k));
    std::vector<float>& buf = block_buffer(block * k);
    for (int64_t j = 0; j < n; j += block) {
        const int64_t nb = std::min(block, n - j);
        W.read_rows(j, j + nb, buf.data());
        detail::gemm(/*trans_b=*/false, m, k, nb, A.ptr() + j, n, buf.data(), k, C.ptr(), k, /*accumulate=*/j > 0);
    }
    return C;
}

}  // namespace tfi
