#include "tfi/weight.h"

#include <algorithm>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

#include <unistd.h>

#include "half.h"
#include "ops.h"

namespace tfi {

namespace {

// Converted half-precision blocks hold about this many floats (1 MB), per thread.
constexpr int64_t kBlockFloats = int64_t(1) << 18;

std::vector<float>& block_buffer(int64_t floats) {
    thread_local std::vector<float> buf;
    if (static_cast<int64_t>(buf.size()) < floats) buf.resize(static_cast<size_t>(floats));
    return buf;
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
        if (!W.is_fp16()) {
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
