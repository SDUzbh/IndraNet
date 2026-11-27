// Internal building blocks shared by the forward pass (model.cpp) and the backward pass (grad.cpp).
// Not part of the public API.
#pragma once

#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "tfi/config.h"
#include "tfi/tensor.h"

namespace tfi {

class Hooks;

class Model;

namespace detail {

class Gpu;     // gpu.h (macOS)
struct GpuKV;  // gpu.h (macOS): a KVCache's copy on the GPU

void activate_(Tensor& t, const std::string& act);
// d act(x) / dx, elementwise, for the pre-activation x.
void activation_grad_(Tensor& x, const std::string& act);
Tensor split_heads(const Tensor& x, int64_t H, int64_t D);            // (T, H*D) -> (H, T, D)
Tensor split_heads(const float* x, int64_t T, int64_t H, int64_t D);  // the same from a row-major (T, H*D) array
Tensor merge_heads(const Tensor& x);                                  // (H, T, D) -> (T, H*D)
// A layer's RoPE: the first `rot` values of each head rotate at frequencies inv_freq (rot / 2), and
// with rope_scaling cos and sin are multiplied by `factor`. Sequences longer than max_len (0: no
// limit) are refused: there "dynamic" and "longrope" scaling change their frequencies.
struct RopeTable {
    int64_t rot = 0;
    std::vector<float> inv_freq;
    float factor = 1.f;
    int64_t max_len = 0;
};
RopeTable rope_table(const ModelConfig& c);
void check_rope_length(const ModelConfig& c, const RopeTable& t, int64_t n_positions);
// Rotate (H, T, D) in place at positions 0..T-1 (indranet's "rotate half" RoPE), or at positions[t]
// for row t when given. `inverse` applies the transpose (the backward pass): with no cos/sin factor
// that is also the inverse rotation.
void apply_rope_(Tensor& x, const ModelConfig& c, bool inverse = false,
                 const std::vector<int64_t>* positions = nullptr);
std::vector<float> alibi_slopes(int64_t n);
// C(m,n) = A(m,k) @ op(B) with row strides lda/ldb/ldc; op(B) = B (k,n) or, with trans_b, B(n,k)^T.
// With accumulate, C += A @ op(B).
void gemm(bool trans_b, int64_t m, int64_t n, int64_t k, const float* A, int64_t lda, const float* B, int64_t ldb,
          float* C, int64_t ldc, bool accumulate = false);

// Backward pass of Model::norm (prefix's weight, config c) over the last dimension: dL/dx given its
// input x and dL/dy (grad.cpp).
Tensor norm_backward(const Model& m, const std::string& prefix, const ModelConfig& c, const Tensor& x,
                     const Tensor& dy);

// The part of attention's backward pass between o_proj and the q/k/v projections, for one sequence
// at positions 0..T-1 (grad.cpp). From dz (T, H*D) and the forward pass's hook_q, hook_k, hook_v
// ((H or Hkv, T, D), after qk_norm and RoPE) and hook_pattern (H, T, T): the gradients at those hook
// points, and at the outputs of the q, k and v projections, (T, H*D) and (T, Hkv*D). q0 and k0 are
// the projections split into heads before qk_norm, needed only with qk_norm.
struct AttentionGrads {
    Tensor pattern, q, k, v;        // at the hook points
    Tensor q_proj, k_proj, v_proj;  // at the projections' outputs
};
AttentionGrads attention_backward(const Model& m, int64_t layer, const Tensor& dz, const Tensor& q, const Tensor& k,
                                  const Tensor& v, const Tensor& P, const Tensor* q0, const Tensor* k0);

// Rows of several sequences stacked into one (R, d) activation, so that many patched passes run
// through the blocks together and every linear layer is one large matrix multiply
// (Model::forward_rows). Segment s is rows [offset[s], offset[s] + len[s]) of the stack: positions
// [start[s], start[s] + len[s]) of its sequence. Positions before start[s] are not recomputed:
// attention takes their keys and values from prefix_k / prefix_v[layer] ((Hkv, T, D), e.g. a cached
// run's hook_k / hook_v), which must hold what those positions would have produced.
// For tests: rows per batched pass (residual-stream patching; head patching, ablation and
// reference means); 0 restores the defaults.
void set_patch_batch_rows(int64_t residual, int64_t heads);
// For tests: check every KV cache after each forward_cached step (KVCache::check: the up-to-date prefixes
// of its CPU and GPU copies, and their values where both hold them), as TFI_CHECK_KV=1 does.
void set_check_kv(bool on);
// For tests: cached keys (of one layer) from which an fp16 cache's key / value heads run side by side on the CPU;
// 0 restores the default.
void set_kv_side_by_side(int64_t values);
// For tests (macOS): false runs a batched pass's GPU work one linear layer at a time, the residual
// stream returning to the CPU after each, instead of keeping it on the GPU (Model::run_rows).
void set_gpu_resident(bool on);
// For tests (macOS): true computes a batched pass's attention (and, in gradient passes, its backward
// pass) on the GPU outside GPU-only mode too. By default only GPU-only mode and single passes do:
// elsewhere the CPU's attention, overlapping the GPU's work on other batches, is faster than the GPU
// kernels (tiled: 0.83-0.94x at 128 tokens, 0.59-0.86x at 512).
void set_gpu_attention(bool on);
// For tests (macOS): false computes GPU attention with the per-row kernel instead of the tiled one
// (head sizes that are multiples of 8 up to 128; other heads always use the per-row kernel).
void set_gpu_attention_tiled(bool on);
// For tests (macOS): the GPU's circuits computations keep each run's inputs under this many bytes,
// splitting the work into several runs if needed; 0 restores the default (512 MB).
void set_circuits_run_bytes(int64_t bytes);

// Whether a single pass (Model::forward_from) of T rows through `layers` blocks of `layer_weights`
// weights each goes to the GPU, `started` saying whether it is running, with the batched passes'
// thresholds min_rows and min_work (gpu_min_rows, gpu_min_work). The batched passes' rule, or, once
// the GPU is running, from 2 rows when a layer holds kSinglePassLayerWeights weights or the rows x
// a layer's weights reach kSinglePassRowWork (tuned on an M5 Pro, models of 4x256 to 24x1024).
constexpr double kSinglePassLayerWeights = 3e6, kSinglePassRowWork = 5e7;
bool single_pass_on_gpu(int64_t T, int64_t layers, double layer_weights, bool started, int64_t min_rows,
                        double min_work);

// The sampler's weights (Sampler, generate.cpp): w[i] = exp((logits[i] - top) * scale) in float, for logits
// at most `top`; 0 where that exponent is below -87 (exp under fp32's smallest normal), and for NaN and
// -inf. The same bits on every platform: exp_weight's operations, four at a time on ARM.
float exp_weight(float x);  // exp(x) for x <= 0, 0 below -87 and for NaN
void exp_weights(std::span<const float> logits, float top, float scale, std::span<float> w);

// One layer's earlier keys or values for attention, (H, T, D) row-major: float32 (a cached run's hook_k /
// hook_v, a float32 KV cache) or fp16 bit patterns (a KV cache kept in fp16, read as it is). From a Tensor
// pointer implicitly (null: none).
struct KVRef {
    const float* f = nullptr;
    const uint16_t* h = nullptr;
    int64_t H = 0, T = 0, D = 0;
    KVRef() = default;
    KVRef(const Tensor* t) {  // NOLINT(google-explicit-constructor): RowBatch's prefixes were Tensor pointers
        if (!t) return;
        if (t->dim() != 3)
            throw std::invalid_argument("prefix keys / values must be (Hkv, T, D), got " + shape_str(t->shape));
        f = t->ptr();
        H = t->shape[0];
        T = t->shape[1];
        D = t->shape[2];
    }
    bool empty() const { return !f && !h; }
};

// Model::gradients_rows beyond attribution patching's use: run_with_gradients on the GPU.
struct GradientOptions {
    const Hooks* hooks = nullptr;           // the pass's hooks (one sequence only), as run_with_gradients takes them
    bool all_rows = false;                  // dmetric gets a segment's every row's logits (len, vocab), not its last
    std::vector<Tensor>* logits = nullptr;  // receives the logits dmetric got, per segment
};

struct RowBatch {
    std::vector<int64_t> start, len, offset;
    std::vector<KVRef> prefix_k, prefix_v;  // per layer; unused when every start is 0
    Gpu* gpu = nullptr;                     // macOS: if set, the blocks run on the GPU (Model::run_rows)
    bool fp16 = false;                      // ... with fp16 multiply inputs (Gpu::linear, Gpu::Seq)
    bool whole = false;  // one whole sequence (a single pass, Model::forward_from): hook_pattern runs as there
    // One segment continuing a cached sequence (Model::forward_cached): hook_pattern runs on its
    // (H, len, start + len) pattern, on the CPU.
    bool cached = false;
    // macOS, with `cached` and gpu: the cache's copy on the GPU, positions [0, start) in place. Then
    // attention runs on the GPU reading them there, the new positions' keys and values are appended to
    // it, and (without fp16) the whole pass is one GPU submission.
    GpuKV* kv = nullptr;
    // Several sequences (Model::forward_cached on a KVBatch): segment s's earlier keys and values, per
    // layer, instead of the shared prefix_k / prefix_v; and (with kv) where its region starts in the
    // GPU's buffers, which hold every sequence's positions, a region each.
    std::vector<std::vector<KVRef>> seg_prefix_k, seg_prefix_v;
    std::vector<int64_t> kbase;
    const KVRef& prefix_k_of(size_t s, int64_t layer) const {
        return seg_prefix_k.empty() ? prefix_k.at(layer) : seg_prefix_k.at(s).at(layer);
    }
    const KVRef& prefix_v_of(size_t s, int64_t layer) const {
        return seg_prefix_v.empty() ? prefix_v.at(layer) : seg_prefix_v.at(s).at(layer);
    }
    int64_t rows() const { return offset.empty() ? 0 : offset.back() + len.back(); }
    void add(int64_t start_pos, int64_t n) {
        offset.push_back(rows());
        start.push_back(start_pos);
        len.push_back(n);
    }
};

}  // namespace detail
}  // namespace tfi
