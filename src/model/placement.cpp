#include "model/placement.h"

#include <algorithm>
#include <atomic>

#include "gpu/device.h"
#include "model/ops.h"
#include "tfi/device.h"
#include "tfi/model.h"
#ifdef TFI_USE_ACCELERATE
#include "gpu/gpu.h"
#endif

namespace tfi::detail {

namespace {
std::atomic<bool> g_gpu_resident{true};
std::atomic<bool> g_gpu_attention{false};  // tests: GPU attention outside GPU-only mode
std::atomic<bool> g_attn_tiled{true};      // tests: false uses the per-row attention kernel
}  // namespace

void set_gpu_resident(bool on) { g_gpu_resident = on; }
void set_gpu_attention(bool on) { g_gpu_attention = on; }
void set_gpu_attention_tiled(bool on) { g_attn_tiled = on; }
bool gpu_attention_tiled() { return g_attn_tiled.load(); }

double block_weights(const Model& m) {
    double n = 0;
    for (const auto& [name, w] : m.weights())
        if (w.dim() == 2 && name.starts_with("blocks.")) n += static_cast<double>(w.numel());
    return n;
}

double block_weights_per_layer(const Model& m) {
    return block_weights(m) / static_cast<double>(std::max<int64_t>(1, m.cfg().n_layers));
}

bool single_pass_on_gpu(int64_t T, int64_t layers, double layer_weights, bool started, int64_t min_rows,
                        double min_work) {
    const double work = static_cast<double>(T) * static_cast<double>(layers) * layer_weights;
    if (work <= 0) return false;
    if (T >= min_rows && (started || work >= min_work)) return true;  // the batched passes' rule
    // Once the GPU has started, shorter passes too, where it measured faster (M5 Pro): from 2 rows (a
    // 1-row pass is a matrix-vector product, cheap on the CPU), when each layer's weights take the
    // CPU longer to read than the GPU's fixed cost per layer (about 0.4 ms), or the multiply-adds
    // take it longer to do.
    return started && T >= 2 &&
           (layer_weights >= kSinglePassLayerWeights || static_cast<double>(T) * layer_weights >= kSinglePassRowWork);
}

#ifdef TFI_USE_ACCELERATE

Gpu* gpu_for_work(const Model& m, double work) {
    if (!use_gpu() || work <= 0 || (!m.has_gpu() && work < gpu_min_work())) return nullptr;
    return m.gpu();
}

Gpu* single_pass_gpu(const Model& m, int64_t layer, int64_t T) {
    if (!use_gpu()) return nullptr;
    const bool on = single_pass_on_gpu(T, m.cfg().n_layers - layer, block_weights_per_layer(m), m.has_gpu(),
                                       gpu_min_rows(), gpu_min_work());
    return on ? m.gpu() : nullptr;
}

Gpu* cached_pass_gpu(const Model& m, int64_t rows, bool fresh) {
    if (!use_gpu() || !gpu_attention_fits(m, 0)) return nullptr;
    if (fresh) return single_pass_gpu(m, 0, rows);
    if (gpu_only() || rows >= gpu_min_rows() || block_weights(m) >= kDecodeWeights) return m.gpu();
    return nullptr;
}

bool gpu_attention_fits(const Model& m, int64_t layer) {
    for (int64_t l = layer; l < m.cfg().n_layers; ++l)
        if (m.layer_cfg(l).head_dim() > Gpu::kMaxHeadDim) return false;
    return true;
}

bool linear_on_gpu(const RowBatch* batch, const Tensor& x) {
    return batch && batch->gpu && x.size(-1) && x.numel() / x.size(-1) >= gpu_min_rows();
}

bool unembed_on_gpu(const RowBatch* batch, const Tensor& normed) {
    return batch && batch->gpu && normed.size(-1) &&
           (batch->whole || normed.numel() / normed.size(-1) >= gpu_min_rows());
}

// A single pass (batch.whole) comes here only if single_pass_on_gpu chose the GPU for it.
bool blocks_on_gpu(const RowBatch& batch, int64_t R) {
    return g_gpu_resident.load() && batch.gpu && (batch.whole || batch.kv || R >= gpu_min_rows());
}

// The CPU's attention is faster only while it overlaps the GPU's work on other batches; a single pass has none to
// overlap, so its attention runs on the GPU too, if the kernels take every layer's head size (GPU-only mode has no
// CPU to fall back on: the GPU says why it can't). So does a pass reading a KV cache on the GPU (forward_cached
// checks the head sizes).
bool batch_attention_on_gpu(const Model& m, const RowBatch& batch, int64_t layer) {
    return gpu_only() || batch.kv || ((g_gpu_attention.load() || batch.whole) && gpu_attention_fits(m, layer));
}

bool gradient_attention_on_gpu(const Model& m, Gpu& gpu) {
    bool accelerated = g_attn_tiled.load();
    for (int64_t l = 0; accelerated && l < m.cfg().n_layers; ++l)
        accelerated = gpu_attention_backward_accelerated(gpu, m.layer_cfg(l).head_dim());
    return gpu_only() || ((g_gpu_attention.load() || accelerated) && gpu_attention_fits(m, 0));
}

bool gradients_go_to_gpu(const Model& m, int64_t T) { return T > 0 && single_pass_gpu(m, 0, T) != nullptr; }

bool gradient_runs_use_gpu(const Model& m, int64_t rows) {
    if (!use_gpu()) return false;
    if (m.has_gpu()) return rows >= gpu_min_rows();
    return 3.0 * static_cast<double>(rows * m.cfg().n_layers) * block_weights_per_layer(m) >= gpu_min_work();
}

int64_t min_gradient_batch_rows() { return std::max<int64_t>(1, gpu_min_rows() / 4); }

#else  // no GPU: every pass on the CPU

Gpu* gpu_for_work(const Model&, double) { return nullptr; }
Gpu* single_pass_gpu(const Model&, int64_t, int64_t) { return nullptr; }
Gpu* cached_pass_gpu(const Model&, int64_t, bool) { return nullptr; }
bool gpu_attention_fits(const Model&, int64_t) { return false; }
bool linear_on_gpu(const RowBatch*, const Tensor&) { return false; }
bool unembed_on_gpu(const RowBatch*, const Tensor&) { return false; }
bool blocks_on_gpu(const RowBatch&, int64_t) { return false; }
bool batch_attention_on_gpu(const Model&, const RowBatch&, int64_t) { return false; }
bool gradient_attention_on_gpu(const Model&, Gpu&) { return false; }
bool gradients_go_to_gpu(const Model&, int64_t) { return false; }
bool gradient_runs_use_gpu(const Model&, int64_t) { return false; }
int64_t min_gradient_batch_rows() { return 1; }

#endif

}  // namespace tfi::detail
