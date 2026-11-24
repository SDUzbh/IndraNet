// The backend-neutral half of running independent GPU work in short pieces (Gpu::Batch): groups of
// recorded work are packed into chunks under a work budget, every chunk is submitted before any is
// waited for, a chunk that fails runs again on its own, and GpuFailure is thrown when the retries
// run out (the caller then computes the result on the CPU). A GPU backend only turns a chunk into a
// submission and reports whether it completed (ChunkBackend).
//
// GPUs that also drive a display stop work that runs too long (macOS: MTLCommandBufferError
// timeout; Windows: TDR), so each chunk stays well under that limit.
//
// No GPU API here: this compiles, and is tested, on every platform. Internal.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tfi {
namespace detail {

// What run_chunks throws when a chunk still fails after its retries.
struct GpuFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Groups [first, last) of the recorded work.
struct Chunk {
    size_t first, last;
};

// Whole groups per chunk, in order, at most `budget` floating-point operations each; a group larger
// than the budget forms a chunk alone. group_flops[i] is group i's cost.
std::vector<Chunk> pack_chunks(const std::vector<double>& group_flops, double budget);

// A GPU backend's side of run_chunks.
class ChunkBackend {
  public:
    virtual ~ChunkBackend() = default;
    // Starts running chunk `c` without waiting for it; returns a ticket for wait().
    virtual size_t submit(const Chunk& c) = 0;
    // Waits until the ticket's work is done: true if it completed, false if the GPU failed it.
    virtual bool wait(size_t ticket) = 0;
};

// Runs every chunk: submits them all (so the GPU never waits for the CPU), then waits for each in
// order. A chunk that failed is submitted again on its own, up to `retries` times; GpuFailure if it
// still fails. Chunks must be independent (none reads a value another wrote), so that any of them
// can run again after the others.
void run_chunks(ChunkBackend& backend, const std::vector<Chunk>& chunks, int retries = 2);

// GPU-only mode (Device::Gpu in tfi/analysis.h): every size threshold is ignored, and GPU work is
// never replaced by the CPU's.
bool gpu_only();
void set_gpu_only(bool on);

// Chunks hold at most this many floating-point operations (about 0.25 s of work on an M5 Pro).
double gpu_chunk_flops();
// For tests: 0 restores the default.
void set_gpu_chunk_flops(double flops);
// For tests: the next `n` chunks to finish report a failure after running, as if the GPU had
// stopped them.
void set_gpu_fail_chunks(int n);

}  // namespace detail
}  // namespace tfi
