// Model weights, kept in float32 or, to halve a model's memory, in IEEE half precision (fp16).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "tfi/tensor.h"

namespace tfi {

// How a model keeps its weights in memory (Model::load). fp16 stores every 2-D weight (embeddings,
// attention and MLP projections, unembedding) in half precision, halving the model's memory. Each
// value is rounded to the nearest fp16 value, a relative error of at most 2^-11 (about 5e-4); a
// weight with a value outside fp16's range (|v| > 65504, or not finite) stays float32. 1-D weights
// (norm scales, biases) always stay float32. The CPU's arithmetic is float32 either way. On the GPU an fp16
// model's matrix multiplies take fp16 inputs (the activations rounded to fp16; sums and outputs float32),
// several times faster, with errors around 1e-3 relative; so do its passes' attention products (queries, keys,
// values and attention weights rounded to fp16), except where the pattern is captured or gradients are taken. A
// run with an input outside fp16's range is repeated with float32 inputs. A float32 model's GPU multiplies are
// float32.
enum class WeightFormat { fp32, fp16 };

// One weight. A 2-D weight (a matrix: embeddings, projections, unembedding) keeps its values, float32
// or half precision, in page-aligned memory that the GPU can use in place, without a copy of its own:
// an allocation of its own, or a memory-mapped model file (borrowed()); copies of a Weight share it. Read matrices
// through rows(), cols(), to_tensor() and the linear() / matmul() overloads below, which give float32 either way. Other
// weights (norm scales, biases) are float32 Tensors, read with f32().
class Weight {
  public:
    Shape shape;

    Weight() = default;
    explicit Weight(Tensor t);  // float32
    // A 2-D weight in half precision, rounded to nearest even. Throws if a value does not fit (fits_fp16).
    static Weight fp16(const Tensor& t);
    // A 2-D weight from fp16 values given as bit patterns (e.g. an F16 tensor in a file), as they are.
    // Throws if a value is infinite or NaN.
    static Weight fp16_bits(Shape shape, const uint16_t* bits);
    // A 2-D weight whose values `fill` writes straight into its storage: numel() float32 values, or
    // with fp16, numel() fp16 bit patterns, taken as they are (e.g. read or converted from a file).
    static Weight filled(Shape shape, bool fp16, const std::function<void(void* dst)>& fill);
    static bool fits_fp16(const Tensor& t);  // every value finite and within +-65504
    // A 2-D weight whose values stay in memory owned elsewhere (a memory-mapped model file): nothing is
    // copied. data must start on a page boundary and capacity bytes from it (whole pages, at least the
    // values' size) must be readable, so that the GPU can use it in place too; owner keeps it alive.
    static Weight borrowed(Shape shape, bool fp16, const void* data, size_t capacity,
                           std::shared_ptr<const void> owner);
    bool is_borrowed() const { return borrowed_; }  // values in memory owned elsewhere (see borrowed())

    int64_t numel() const;
    int64_t dim() const { return std::ssize(shape); }
    bool is_fp16() const { return half_; }
    int64_t bytes() const { return numel() * (is_fp16() ? 2 : 4); }  // memory the values take

    // A weight that is not 2-D (norm scale, bias), as a float32 Tensor. Throws std::logic_error for a
    // 2-D weight: read those with rows(), cols(), to_tensor() or linear() / matmul().
    const Tensor& f32() const;
    // A 2-D weight's values, row-major: float32, or fp16 bit patterns when is_fp16(); nullptr for
    // other weights. The allocation starts on a page boundary and is capacity() bytes, a whole number
    // of pages, so that the GPU can use it in place; owner() keeps it alive.
    const void* data() const { return mat_.get(); }
    const float* f32_data() const { return half_ ? nullptr : static_cast<const float*>(data()); }
    const uint16_t* f16() const { return half_ ? static_cast<const uint16_t*>(data()) : nullptr; }
    size_t capacity() const { return capacity_; }
    std::shared_ptr<const void> owner() const { return mat_; }

    Tensor to_tensor() const;                       // every value, as float32
    Tensor rows(int64_t begin, int64_t end) const;  // W[begin:end] of a 2-D weight, as float32
    Tensor cols(int64_t begin, int64_t end) const;  // W[:, begin:end] of a 2-D weight, as float32
    Tensor row(int64_t i) const;                    // W[i] of a 2-D weight, as a 1-D float32 tensor
    // W[begin:end] of a 2-D weight into dst ((end - begin) * columns floats), as float32.
    void read_rows(int64_t begin, int64_t end, float* dst) const;

  private:
    Tensor vec_;                       // a weight that is not 2-D
    std::shared_ptr<const void> mat_;  // a 2-D weight's values (page-aligned)
    size_t capacity_ = 0;
    bool half_ = false;
    bool borrowed_ = false;
};

// y = x @ W^T + b (W (out, in), as linear() above) with a model weight. A half-precision W is
// converted to float32 in blocks of rows as the multiply goes, so the full float32 copy never exists.
// With 1-16 rows of x (a generated token, a speculative step), it multiplies straight from the weights
// instead, split across threads; float32 and fp16 weights take the same steps, so fp16 gives what float32
// weights holding the same rounded values give, bit for bit, and each row's output is the same whatever
// the other rows.
Tensor linear(const Tensor& x, const Weight& W, const Weight* b = nullptr);
// linear(x, Ws[i], bs[i]) for every i (bs may be empty, or hold nulls), the same values as separate
// calls. With 1-16 rows of x the weights' rows are split across the threads together, so that one wake
// of the threads covers them all (e.g. q, k and v).
std::vector<Tensor> linear_many(const Tensor& x, const std::vector<const Weight*>& Ws,
                                const std::vector<const Weight*>& bs = {});
// A(m, n) @ W(n, k) for a 2-D weight W: e.g. linear()'s input gradient dy @ W.
Tensor matmul(const Tensor& A, const Weight& W);

}  // namespace tfi
