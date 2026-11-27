// Process-wide execution settings: how many threads the independent passes use, and where the
// computations run (CPU, GPU or both). Every part of the library reads them; tfi/analysis.h
// includes this header too.
#pragma once

namespace tfi {

// Patching, ablation and head_means run their independent forward passes on this many threads.
// The default (n = 0) is the TFI_THREADS environment variable (read at first use), else every core.
// Results are the same for any thread count.
void set_num_threads(int n);
int num_threads();

// Where the computations run (results agree up to float rounding):
//   Both (the default): on macOS with a Metal GPU, the GPU for the work large enough to repay
//     starting it (the batched passes of patching, ablation and reference means, attribution's
//     gradient runs, the circuits computations), the CPU for the rest;
//   Cpu: everything on the CPU;
//   Gpu: the GPU for everything it can run, whatever the size, and never the CPU instead: setting it
//     throws if there is no Metal GPU (or not macOS), and GPU work that keeps failing is an error.
// Until set_device is called, the TFI_NO_GPU environment variable (any value but 0; read at first use)
// selects Cpu.
enum class Device { Cpu, Both, Gpu };
void set_device(Device device);
Device device();
// The earlier interface: set_use_gpu(false) is set_device(Device::Cpu), set_use_gpu(true) is
// set_device(Device::Both); use_gpu() is device() != Device::Cpu.
void set_use_gpu(bool use);
bool use_gpu();

}  // namespace tfi
