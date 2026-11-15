// The GPU side of batched patching passes: matrix multiplies through Metal Performance Shaders, and
// (Gpu::Seq) whole layers but for attention, with the residual stream kept on the GPU.
// macOS only (TFI_USE_ACCELERATE); internal, not part of the public API.
#pragma once

#ifdef TFI_USE_ACCELERATE

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "tfi/tensor.h"
#include "tfi/weight.h"
#include "gpu_chunks.h"

namespace tfi {
namespace detail {

// Activations of Gpu::Seq, the same functions as detail::activate_.
enum class GpuAct : int32_t { Relu, Gelu, GeluTanh, Silu, Tanh, Relu2 };

class Gpu {
  public:
    class Seq;
    class Batch;

    // nullptr when there is no Metal device (e.g. some virtual machines): callers then use the CPU.
    static std::unique_ptr<Gpu> create();
    ~Gpu();

    // linear(x, W, b) on the GPU: x (..., k), W (n, k). A float32 W is copied to the GPU the first
    // time it is used; a half-precision W (WeightFormat::fp16) is used where it is, without a copy.
    // Weights are identified by their address, so a Gpu must only see weights that outlive it and
    // never change (Model::gpu() keeps one per model). Thread-safe.
    // With fp16, x and W go in half precision (accumulation and output stay fp32): the multiply runs
    // several times faster, with relative errors around 1e-3. An x or a float32 W with a value outside
    // fp16's range (|v| > 65504) is multiplied in fp32 instead. Without fp16, x stays fp32; a
    // half-precision W is then multiplied as it is, exactly (each fp16 value is an fp32 value).
    Tensor linear(const Tensor& x, const Weight& W, const Weight* b, bool fp16 = false);

    // linear(x, Ws[i], bs[i]) for every i in one GPU submission: x is copied to the GPU once and
    // the thread waits once, instead of once per weight. Results are identical to separate
    // linear() calls. bs[i] may be null; bs may be empty (no biases).
    std::vector<Tensor> linear_many(const Tensor& x, const std::vector<const Weight*>& Ws,
                                    const std::vector<const Weight*>& bs, bool fp16 = false);

  private:
    Gpu();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Operations on GPU-resident fp32 buffers ("slots"), recorded and then run as one GPU submission:
// how Model::forward_rows keeps a batched pass's residual stream on the GPU, with one round trip per
// layer (q, k, v come back for attention on the CPU, z goes in). Slots are numbered by the caller;
// the CPU reads and writes them through data() between runs. One per thread; the Gpu must outlive
// it. Weights are cached as in Gpu::linear.
class Gpu::Seq {
  public:
    explicit Seq(Gpu& gpu);
    ~Seq();
    Seq(const Seq&) = delete;
    Seq& operator=(const Seq&) = delete;

    // Make `slot` hold at least `floats` values and return it. It only grows; growing loses the contents.
    float* reserve(int slot, int64_t floats);
    float* data(int slot);  // a reserved slot's values

    // Record operations; sizes are checked against the slots when run() encodes them.
    void linear(int x, int y, int64_t rows, const Weight& W, const Weight* b);  // y = x W^T (+ b)
    // Over rows of w.numel() values: (x - mean) / sqrt(var + eps) * w (+ b), as Model::norm.
    // Without `layernorm` (rmsnorm), mean is 0 and b must be null.
    void norm(int x, int y, int64_t rows, const Tensor& w, const Tensor* b, bool layernorm, float eps);
    void activate(int x, int64_t n, GpuAct act);         // in place
    void glu(int gate, int up, int64_t n, GpuAct act);   // gate = act(gate) * up
    void add(int a, int b, int out, int64_t n);          // out = a + b; out may be a or b
    void copy(int x, int y, int64_t n);
    // Backward passes (fp32 inputs, whatever run() is given): dx = dy W, linear()'s input gradient for W (out, in);
    // norm()'s input gradient given its input x and output gradient dy (as grad.cpp's norm_backward);
    // dpre = dpost * act'(pre); and for a GLU, post = act(pre) * up: dpre = dpost * up * act'(pre),
    // dup = dpost * act(pre).
    void linear_t(int dy, int dx, int64_t rows, const Weight& W);
    void norm_backward(int x, int dy, int dx, int64_t rows, const Tensor& w, bool layernorm, float eps);
    void act_backward(int dpost, int pre, int dpre, int64_t n, GpuAct act);
    // RoPE ("rotate half") on the first `rot` values of each of `heads` heads of D per row, y = rope(x):
    // row r is at position meta[4 r] (uint32), and inv_freq holds the rot / 2 frequencies. inverse:
    // rotates back (RoPE's backward pass).
    void rope(int x, int y, int64_t rows, int64_t heads, int64_t D, int64_t rot, int inv_freq, int meta, bool inverse = false);
    // Causal attention of every row's query, as Model::attend: z (rows, H D) from q (rows, H D) and
    // k, v (rows, Hkv D). meta has 4 uint32 per row: its position, its segment's start position, its
    // segment's first row and its segment's number of rows. Keys and values before the segment's start come from pk, pv (Hkv, Tp, D),
    // the rest from the segment's own rows. Scores are scaled, then softcapped (cap > 0), then ALiBi'd
    // (slopes: one per head, if has_slopes), within `window` keys if window > 0. Float32.
    // With `blocks` > 0 and D a multiple of 8 up to 128, slot `blk` divides the rows into that many
    // blocks of up to kAttnBlockRows consecutive rows of one segment (4 uint32 each: first row, rows,
    // segment start position, segment's first row), and each block's keys are read once for all its
    // rows, on the matrix units (faster).
    struct AttnShape {
        int64_t rows, H, Hkv, D, Tp, window;
        float scale, cap;
        bool has_slopes;
        int64_t blocks = 0;
    };
    // lse >= 0: also each (row, head)'s log of the sum of exp(score) into that slot (rows, H), for
    // attention_backward.
    void attention(int q, int k, int v, int z, int pk, int pv, int meta, int slopes, const AttnShape& shape, int blk = 0,
                   int lse = -1);
    // attention()'s backward pass with no earlier positions (shape.Tp = 0): from dz (rows, H D) and
    // the forward pass's q, k, v, z and lse, dq into `dq` (rows, H D), dk and dv into `dk`, `dv`
    // (rows, Hkv D); `delta` is scratch (rows, H).
    struct AttnSlots {
        int q, k, v, z, dz, lse, meta, slopes;
    };
    void attention_backward(const AttnSlots& in, int delta, int dq, int dk, int dv, const AttnShape& shape);
    // One whole sequence (shape.rows rows at positions 0..rows-1, Tp = 0): the attention pattern P
    // (H, rows, rows) from q, k and attention()'s lse, as Model::attend's hook_pattern; and z = P v.
    void attention_pattern(int q, int k, int lse, int slopes, int P, const AttnShape& shape);
    void attention_apply(int P, int v, int z, const AttnShape& shape);
    // Rows per attention block (AttnShape::blocks).
    static constexpr int64_t kAttnBlockRows = 32;
    void glu_backward(int dpost, int pre, int up, int dpre, int dup, int64_t n, GpuAct act);

    // Run everything recorded since the last run in one submission and wait. With fp16, linear()
    // inputs and weights go in half precision, as in Gpu::linear; if any input is outside fp16's
    // range the whole run is repeated in fp32. So the recorded operations must not overwrite a slot
    // that they read before writing it (anything the CPU or an earlier run left there).
    void run(bool fp16);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Independent GPU work whose results only come back to the CPU at the end (circuits). Operations
// are recorded in groups; run() hands them to run_chunks (gpu_chunks.h): whole groups per command
// buffer under gpu_chunk_flops(), all committed before one wait, a failed one run again, and
// GpuFailure if it keeps failing (the caller then computes the result on the CPU). So a group must
// never read a scratch value that another group wrote.
// One per thread; the Gpu must outlive it. Buffers stay until release() or the Batch's end.
class Gpu::Batch {
  public:
    explicit Batch(Gpu& gpu);
    ~Batch();
    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;

    // A buffer of `count` float32 values (or with fp16, fp16 values) in memory shared with the
    // CPU: inputs the CPU writes through floats() or halves(), scratch, and outputs the CPU reads
    // after run().
    int buffer(int64_t count, bool fp16 = false);
    float* floats(int buf);
    uint16_t* halves(int buf);  // fp16 bit patterns
    const uint32_t* uints(int buf) { return reinterpret_cast<const uint32_t*>(floats(buf)); }
    // A 2-D model weight (float32 or fp16), used in place. With fp16, a float32 weight's fp16 copy
    // instead (kept by the Gpu), unless a value doesn't fit fp16: then the float32 weight.
    // The weight must outlive the Batch.
    int weight(const Weight& W, bool fp16 = false);
    bool is_fp16(int buf);
    void release(int buf);  // back to the GPU's pool; the number is not reused

    // A matrix inside a buffer: its first element and row stride, in elements.
    struct Mat {
        int buf;
        int64_t offset;
        int64_t ld;
    };
    // C (m, n) = A (m, k) B (k, n), or with trans_b A B^T for B (n, k). A and B are float32 or
    // (weights) fp16; C is float32.
    void matmul(Mat A, Mat B, Mat C, int64_t m, int64_t n, int64_t k, bool trans_b);
    // out[out_offset + i * (cols / bn) + j] = the sum of squares of block (i, j), bm x bn, of the
    // rows x cols matrix C (rows and cols multiples of bm and bn).
    void block_sumsq(Mat C, int64_t rows, int64_t cols, int64_t bm, int64_t bn, int out, int64_t out_offset);
    // uints(out)[out_offset + r] = the index of the largest of row r's n values (the first on ties).
    void row_argmax(Mat C, int64_t rows, int64_t n, int out, int64_t out_offset);
    // dst = src as fp16, n contiguous values from the buffers' starts. A value outside fp16's range
    // sets overflowed() (the caller then does the work again in float32).
    void to_half(int src, int dst, int64_t n);
    bool overflowed();  // since the last run() began
    void end_group();  // the operations since the previous end_group() form one group
    // Run every group recorded since the last run() and wait for all of them.
    void run();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Batched passes and linear layers with at least this many rows use the GPU (smaller ones are
// faster on the CPU). GPU-only mode (Device::Gpu): 1.
int64_t gpu_min_rows();
// For tests: 0 restores the default; a very large value turns the GPU off.
void set_gpu_min_rows(int64_t rows);
// A command uses the GPU only if the multiply-adds its GPU-sized batches would run add up to at
// least this many (rows x layers x each layer's weights; a gradient run counts 3 passes). Below it,
// the CPU finishes before the GPU would repay its start-up (about 35 ms in a fresh process).
// GPU-only mode: 0.
double gpu_min_work();
// For tests: 0 restores the default.
void set_gpu_min_work(double multiply_adds);

}  // namespace detail
}  // namespace tfi

#endif
