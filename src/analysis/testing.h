// Settings of the analysis tools that only tests change. Internal.
#pragma once

#include <cstdint>

namespace tfi::detail {

// For tests: rows per batched pass (residual-stream patching; head patching, ablation and
// reference means); 0 restores the defaults.
void set_patch_batch_rows(int64_t residual, int64_t heads);
// For tests (macOS): the GPU's circuits computations keep each run's inputs under this many bytes,
// splitting the work into several runs if needed; 0 restores the default (512 MB).
void set_circuits_run_bytes(int64_t bytes);

}  // namespace tfi::detail
