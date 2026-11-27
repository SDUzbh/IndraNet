// The device setting's internal side (tfi/device.h has the public one). Internal.
#pragma once

namespace tfi::detail {

// GPU-only mode (Device::Gpu): every size threshold is ignored, and GPU work is never replaced by the CPU's.
bool gpu_only();
void set_gpu_only(bool on);

}  // namespace tfi::detail
