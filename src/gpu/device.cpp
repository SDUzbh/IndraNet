#include "gpu/device.h"

#include <atomic>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "tfi/device.h"
#ifdef TFI_USE_ACCELERATE
#include "gpu/gpu.h"
#endif

namespace tfi {

namespace {
std::atomic<int> g_device{-1};  // a Device, or -1: TFI_NO_GPU decides
std::atomic<bool> g_gpu_only{false};
}  // namespace

bool detail::gpu_only() { return g_gpu_only.load(); }
void detail::set_gpu_only(bool on) { g_gpu_only = on; }

void set_device(Device d) {
    if (d == Device::Gpu) {
#ifdef TFI_USE_ACCELERATE
        if (!detail::Gpu::create()) throw std::runtime_error("GPU-only: this Mac has no Metal GPU");
#else
        throw std::runtime_error("GPU-only needs macOS with a Metal GPU");
#endif
    }
    g_device = static_cast<int>(d);
    detail::set_gpu_only(d == Device::Gpu);
}

Device device() {
    if (const int d = g_device.load(); d >= 0) return static_cast<Device>(d);
    static const Device from_env = [] {  // read once: every linear layer asks
        const char* env = std::getenv("TFI_NO_GPU");
        return !env || !*env || std::string(env) == "0" ? Device::Both : Device::Cpu;
    }();
    return from_env;
}

void set_use_gpu(bool use) { set_device(use ? Device::Both : Device::Cpu); }

bool use_gpu() { return device() != Device::Cpu; }

}  // namespace tfi
