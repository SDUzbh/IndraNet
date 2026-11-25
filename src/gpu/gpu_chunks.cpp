#include "gpu_chunks.h"

#include <atomic>
#include <format>
#include <string>

namespace tfi {
namespace detail {

namespace {

constexpr double kDefaultChunkFlops = 1.5e12;  // about 0.25 s at the fp32 multiply rate of an M5 Pro
std::atomic<double> g_chunk_flops{0};
std::atomic<int> g_fail_chunks{0};
std::atomic<bool> g_gpu_only{false};

// Whether a test asked for this finished chunk to count as failed (using up one request).
bool injected_failure() {
    for (int n = g_fail_chunks.load(); n > 0; n = g_fail_chunks.load())
        if (g_fail_chunks.compare_exchange_weak(n, n - 1)) return true;
    return false;
}

bool finished_ok(ChunkBackend& backend, size_t ticket) {
    const bool ok = backend.wait(ticket);
    return ok && !injected_failure();
}

}  // namespace

bool gpu_only() { return g_gpu_only.load(); }
void set_gpu_only(bool on) { g_gpu_only = on; }

double gpu_chunk_flops() {
    const double f = g_chunk_flops.load();
    return f > 0 ? f : kDefaultChunkFlops;
}
void set_gpu_chunk_flops(double flops) { g_chunk_flops = flops; }
void set_gpu_fail_chunks(int n) { g_fail_chunks = n; }

std::vector<Chunk> pack_chunks(const std::vector<double>& group_flops, double budget) {
    std::vector<Chunk> chunks;
    double flops = 0;
    size_t first = 0;
    for (size_t i = 0; i < group_flops.size(); ++i) {
        if (i > first && flops + group_flops[i] > budget) {
            chunks.push_back({first, i});
            first = i;
            flops = 0;
        }
        flops += group_flops[i];
    }
    if (first < group_flops.size()) chunks.push_back({first, group_flops.size()});
    return chunks;
}

void run_chunks(ChunkBackend& backend, const std::vector<Chunk>& chunks, int retries) {
    std::vector<size_t> tickets;
    tickets.reserve(chunks.size());
    for (const Chunk& c : chunks) tickets.push_back(backend.submit(c));
    std::vector<size_t> failed;
    for (size_t i = 0; i < chunks.size(); ++i)
        if (!finished_ok(backend, tickets[i])) failed.push_back(i);
    for (size_t i : failed) {
        bool ok = false;
        for (int attempt = 0; attempt < retries && !ok; ++attempt) ok = finished_ok(backend, backend.submit(chunks[i]));
        if (!ok) throw GpuFailure(std::format("GPU: a chunk of work failed {} times", retries + 1));
    }
}

}  // namespace detail
}  // namespace tfi
