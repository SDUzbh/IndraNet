// Where the model's passes run: every rule that chooses between the CPU and the GPU for a forward pass, a
// gradient pass or a batch of patched passes, in one place. The rules weigh a pass's work against the GPU's fixed
// costs (gpu_min_rows, gpu_min_work in gpu/gpu.h), as measured on an M5 Pro, and follow the device setting
// (tfi/device.h). Callers ask here instead of comparing sizes themselves, so that passes meant to agree (e.g.
// attribution patching's clean and corrupted runs) are always placed alike. Internal.
#pragma once

#include <cstdint>

#include "tfi/tensor.h"

namespace tfi {

class Model;

namespace detail {

class Gpu;
struct RowBatch;

// The 2-D weights (multiply-adds per row) of all the model's blocks, and of one block on average.
double block_weights(const Model& m);
double block_weights_per_layer(const Model& m);

// The model's GPU for `work` multiply-adds, or null (then the CPU does it): only above gpu_min_work, unless the GPU
// has started already (its start-up is paid), and never with Device::Cpu.
Gpu* gpu_for_work(const Model& m, double work);

// Whether a single pass (Model::forward_from) of T rows through `layers` blocks of `layer_weights`
// weights each goes to the GPU, `started` saying whether it is running, with the batched passes'
// thresholds min_rows and min_work (gpu_min_rows, gpu_min_work). The batched passes' rule, or, once
// the GPU is running, from 2 rows when a layer holds kSinglePassLayerWeights weights or the rows x
// a layer's weights reach kSinglePassRowWork (tuned on an M5 Pro, models of 4x256 to 24x1024).
constexpr double kSinglePassLayerWeights = 3e6, kSinglePassRowWork = 5e7;
bool single_pass_on_gpu(int64_t T, int64_t layers, double layer_weights, bool started, int64_t min_rows,
                        double min_work);
// The GPU for a single pass of T rows from `layer` (single_pass_on_gpu), or null.
Gpu* single_pass_gpu(const Model& m, int64_t layer, int64_t T);

// A generated token's pass (forward_cached) goes to the GPU in both mode when the blocks hold at least
// this many weights: then reading them takes the CPU longer than the GPU's whole pass, fixed costs
// included (M5 Pro, fp16: Llama 3.2 1B's 1.0e9 is 1.5x faster on the GPU, Qwen2.5 0.5B's 3.6e8 about
// even, faster with long contexts; smaller models are faster on the CPU).
constexpr double kDecodeWeights = 3e8;
// The GPU for a cached pass (forward_cached) of `rows` new rows, or null: a first pass (`fresh`, an empty cache)
// as a single pass; later ones when the rows are many, the model large (kDecodeWeights), or in GPU-only mode.
// Never when a layer's heads are too wide for the GPU's attention. (A hook reading the pattern is the caller's to
// check: only the CPU's cached attention hands it over.)
Gpu* cached_pass_gpu(const Model& m, int64_t rows, bool fresh);

// Whether the GPU's attention kernels take every layer from `layer` on (Gpu::kMaxHeadDim).
bool gpu_attention_fits(const Model& m, int64_t layer);

// Batched passes (Model::forward_rows) with a GPU in `batch`: whether a linear layer over x's rows multiplies on
// the GPU; whether the unembedding does (a single pass's too: its rows were already judged worth it); whether
// the blocks run there with the residual stream kept on the GPU (Model::run_rows), R rows; and whether their
// attention does too from `layer` on.
bool linear_on_gpu(const RowBatch* batch, const Tensor& x);
bool unembed_on_gpu(const RowBatch* batch, const Tensor& normed);
bool blocks_on_gpu(const RowBatch& batch, int64_t R);
bool batch_attention_on_gpu(const Model& m, const RowBatch& batch, int64_t layer);
// Gradient passes on the GPU (Model::gradients_rows): whether attention and its backward pass run there too.
// In GPU-only mode, and in both mode where every layer's run on the neural accelerators: then faster than the
// CPU's backward pass (M5 Pro, 0.9B model, attribution patching with 4 steps: 3.0 s against 5.0 at 1024 tokens,
// 8.3 against 23.4 at 2048) and no pattern is kept per segment.
bool gradient_attention_on_gpu(const Model& m, Gpu& gpu);
// Whether GPU attention uses the tiled kernels (set_gpu_attention_tiled).
bool gpu_attention_tiled();

// Whether Model::run_with_gradients of T tokens runs on the GPU (macOS, where a single pass of them would). Its
// forward half then multiplies in float32 even with fp16 weights, unlike forward() there.
bool gradients_go_to_gpu(const Model& m, int64_t T);
// Attribution patching's gradient runs, `rows` rows in all (steps x positions): whether the GPU takes part (enough
// work to repay starting it; a forward and backward pass count as 3 passes), and the fewest rows of a batch that
// goes to it: a quarter of gpu_min_rows (default 64), where one such sequence already runs faster there.
bool gradient_runs_use_gpu(const Model& m, int64_t rows);
int64_t min_gradient_batch_rows();

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

}  // namespace detail
}  // namespace tfi
