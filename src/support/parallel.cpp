#include "support/parallel.h"

#include <cstdlib>

namespace tfi {

namespace {
std::atomic<int> g_threads{0};  // 0 = TFI_THREADS, else every core
}  // namespace

void set_num_threads(int n) { g_threads = n < 0 ? 0 : n; }

int num_threads() {
    if (const int n = g_threads.load(); n > 0) return n;
    static const int from_env = [] {  // TFI_THREADS, else the cores: read once, as every linear layer asks
        if (const char* env = std::getenv("TFI_THREADS"))
            if (const int n = std::atoi(env); n > 0) return n;
        const unsigned n = std::thread::hardware_concurrency();
        return n ? static_cast<int>(n) : 1;
    }();
    return from_env;
}

}  // namespace tfi
