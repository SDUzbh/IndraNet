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
// ... and up to this many rows (a speculative step, a short prompt), groups of kDirectRows rows each read
// the weights in turn, chunk by chunk, so that a chunk read from memory for the first group is still in
// the cache for the others. Each row's sums are the same instructions whatever the group, so a row's
// output doesn't depend on how many rows the call has. Measured on an M5 Pro (8192x2048, 16 rows): 0.98 ms
// against the library's 1.44 on float32 weights, 0.69 against 2.4 on fp16 (converted blocks).
constexpr int64_t kDirectMaxRows = 16;

// sum[r] += x[r][k] * w[k] for k in [k0, in), one statement each for the multiply and the add (a
// compiler may fuse a multiply-add within one). Both weight types' scalar steps go through this one
// float32 function (fp16 values converted exactly first), so they are the same instructions whatever
// the compiler makes of them: an fp16 model gives bit for bit what float32 weights holding the same
// rounded values give.
template <int R>
void scalar_dot(const float* x, const float* w, int64_t in, int64_t k0, int64_t k1, float* sum) {
    for (int64_t k = k0; k < k1; ++k)
        for (int r = 0; r < R; ++r) {
            const float p = x[r * in + k] * w[k - k0];
            sum[r] += p;
        }
}

// y[r * ldy + j] = x[r] . W[j] for R rows of x (stride in) and weight rows j0..j1-1 of W (float32 or
// fp16, `in` columns), in float32, each weight row read once. On ARM 16 (then 8) columns at a time
// with vector multiply-adds, fp16 converted in registers; the rest (and everything elsewhere) through
// scalar_dot, fp16 converted 64 values at a time.
template <int R, typename WT>
void direct_rows(const float* x, const WT* W, int64_t in, int64_t j0, int64_t j1, float* y, int64_t ldy) {
    for (int64_t j = j0; j < j1; ++j) {
        const WT* w = W + j * in;
        float sum[R];
        int64_t k = 0;
#if defined(__aarch64__)
        // Four accumulators per row, 16 columns a step: enough independent multiply-adds to hide
        // their latency, so a core streams weights as fast as memory gives them; then 8 at a time.
        float32x4_t acc[R][4];
        for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = acc[r][2] = acc[r][3] = vdupq_n_f32(0.f);
        auto load8 = [&](int64_t at, float32x4_t& lo, float32x4_t& hi) {
            if constexpr (std::is_same_v<WT, uint16_t>) {
                const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(w + at));
                lo = vcvt_f32_f16(vget_low_f16(h));
                hi = vcvt_high_f32_f16(h);
            } else {
                lo = vld1q_f32(w + at);
                hi = vld1q_f32(w + at + 4);
            }
        };
        for (; k + 16 <= in; k += 16) {
            float32x4_t w0, w1, w2, w3;
            load8(k, w0, w1);
            load8(k + 8, w2, w3);
            for (int r = 0; r < R; ++r) {
                const float* xr = x + r * in + k;
                acc[r][0] = vfmaq_f32(acc[r][0], vld1q_f32(xr), w0);
                acc[r][1] = vfmaq_f32(acc[r][1], vld1q_f32(xr + 4), w1);
                acc[r][2] = vfmaq_f32(acc[r][2], vld1q_f32(xr + 8), w2);
                acc[r][3] = vfmaq_f32(acc[r][3], vld1q_f32(xr + 12), w3);
            }
        }
        for (; k + 8 <= in; k += 8) {
            float32x4_t w0, w1;
            load8(k, w0, w1);
            for (int r = 0; r < R; ++r) {
                acc[r][0] = vfmaq_f32(acc[r][0], vld1q_f32(x + r * in + k), w0);
                acc[r][1] = vfmaq_f32(acc[r][1], vld1q_f32(x + r * in + k + 4), w1);
            }
        }
        for (int r = 0; r < R; ++r) sum[r] = vaddvq_f32(vaddq_f32(vaddq_f32(acc[r][0], acc[r][1]), vaddq_f32(acc[r][2], acc[r][3])));
#else
        for (int r = 0; r < R; ++r) sum[r] = 0.f;
#endif
        if constexpr (std::is_same_v<WT, uint16_t>) {
            float buf[64];
            for (; k < in; k += 64) {
                const int64_t n = std::min<int64_t>(64, in - k);
                detail::halves_to_floats(w + k, buf, n);
                scalar_dot<R>(x, buf, in, k, k + n, sum);
            }
        } else {
            scalar_dot<R>(x, w + k, in, k, in, sum);
        }
        for (int r = 0; r < R; ++r) y[r * ldy + j] = sum[r];
    }
}

// direct_rows of weight W (fp16 or float32), its rows j0..j1-1, for n_rows (1..kDirectRows) rows of x.
void direct_range(const float* x, int64_t n_rows, const Weight& W, int64_t j0, int64_t j1, float* y, int64_t ldy) {
    const int64_t in = W.shape[1];
    auto run = [&](auto* w) {
        switch (n_rows) {
            case 1: return direct_rows<1>(x, w, in, j0, j1, y, ldy);
            case 2: return direct_rows<2>(x, w, in, j0, j1, y, ldy);
            case 3: return direct_rows<3>(x, w, in, j0, j1, y, ldy);
            default: return direct_rows<4>(x, w, in, j0, j1, y, ldy);
        }
    };
    if (W.is_fp16()) run(W.f16());
    else run(W.f32_data());
}

// rows(j0, j1) over [0, total), split into chunks of rows run side by side (macOS: libdispatch's
// threads, which stay alive between calls, so a call costs microseconds, not a thread start each).
// At least 512K weights (of `in` columns) per chunk: smaller ones cost more in waking and joining
// threads than they save (measured on an M5 Pro); with several threads allowed, 4 chunks per thread,
// so that faster cores take more of them instead of waiting for the slower ones. One thread allowed:
// no others.
template <typename F>
void run_row_chunks(int64_t total, int64_t in, F&& rows) {
    const int64_t min_rows = std::max<int64_t>(1, (int64_t(1) << 19) / std::max<int64_t>(in, 1));
    const int64_t threads = num_threads();
    const int64_t chunks = threads <= 1 ? 1 : std::max<int64_t>(1, std::min<int64_t>(4 * threads, total / min_rows));
#if defined(__APPLE__)
    if (chunks > 1) {
        struct Job {
            F* fn;
            int64_t total, chunks;
        } job{&rows, total, chunks};
        dispatch_apply_f(static_cast<size_t>(chunks), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &job,
                         [](void* p, size_t c) {
                             const Job& jb = *static_cast<const Job*>(p);
                             const int64_t i = static_cast<int64_t>(c);
                             (*jb.fn)(jb.total * i / jb.chunks, jb.total * (i + 1) / jb.chunks);
                         });
        return;
    }
#else
    (void)chunks;  // elsewhere one thread: the chunks run in order, as one
#endif
    rows(0, total);
}

// Several weights reading the same x rows (1..kDirectMaxRows): their rows taken as one range, so that a
// single run_row_chunks (one wake of the threads) covers them all. y[i] receives W[i]'s output.
void linear_direct_many(const float* x, int64_t n_rows, const std::vector<const Weight*>& Ws, const std::vector<float*>& y) {
    std::vector<int64_t> first(Ws.size() + 1, 0);
    for (size_t i = 0; i < Ws.size(); ++i) first[i + 1] = first[i] + Ws[i]->shape[0];
    const int64_t in = Ws.empty() ? 1 : Ws[0]->shape[1];
    run_row_chunks(first.back(), in, [&](int64_t a, int64_t b) {
        for (size_t i = 0; i < Ws.size(); ++i) {
            const int64_t lo = std::max(a, first[i]), hi = std::min(b, first[i + 1]), out = Ws[i]->shape[0];
            for (int64_t r = 0; lo < hi && r < n_rows; r += kDirectRows)
                direct_range(x + r * in, std::min(kDirectRows, n_rows - r), *Ws[i], lo - first[i], hi - first[i], y[i] + r * out, out);
        }
    });
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
        if (n_rows <= kDirectMaxRows) {
            linear_direct_many(x.ptr(), n_rows, {&W}, {y.ptr()});
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

std::vector<Tensor> linear_many(const Tensor& x, const std::vector<const Weight*>& Ws, const std::vector<const Weight*>& bs) {
    if (!bs.empty() && bs.size() != Ws.size()) throw std::invalid_argument("linear_many: one bias (or null) per weight");
    const int64_t in = x.size(-1), n_rows = in ? x.numel() / in : 0;
    bool direct = n_rows >= 1 && n_rows <= kDirectMaxRows && Ws.size() > 1;
    for (const Weight* W : Ws) direct = direct && W->dim() == 2 && W->shape[1] == in && W->shape[0] > 0;
    std::vector<Tensor> out;
    if (!direct) {  // separate calls (they check the shapes)
        for (size_t i = 0; i < Ws.size(); ++i) out.push_back(linear(x, *Ws[i], bs.empty() ? nullptr : bs[i]));
        return out;
    }
    std::vector<float*> ys;
    for (const Weight* W : Ws) {
        Shape s(x.shape.begin(), x.shape.end() - 1);
        s.push_back(W->shape[0]);
        out.emplace_back(std::move(s));
        ys.push_back(out.back().ptr());
    }
    linear_direct_many(x.ptr(), n_rows, Ws, ys);
    for (size_t i = 0; i < bs.size(); ++i) {
        if (!bs[i]) continue;
        const Tensor& bias = bs[i]->f32();
        const int64_t o = Ws[i]->shape[0];
        if (bias.numel() != o) throw std::invalid_argument("linear: bias does not match the weight");
        for (int64_t r = 0; r < n_rows; ++r)
            for (int64_t j = 0; j < o; ++j) out[i].data[r * o + j] += bias.data[j];
    }
    return out;
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
