// Running independent pieces of work side by side on the threads num_threads() allows (tfi/device.h).
// Internal.
#pragma once

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

#include "tfi/device.h"

namespace tfi::detail {

// fn(0..n-1) on up to num_threads() threads of their own, for long jobs (whole forward passes). Jobs are
// independent; the first exception any job throws is re-thrown here after the workers stop.
template <std::invocable<int64_t> F>
void parallel_for(int64_t n, F&& fn) {
    const int64_t workers = std::min<int64_t>(num_threads(), n);
    if (workers <= 1) {
        for (int64_t i = 0; i < n; ++i) fn(i);
        return;
    }
    std::atomic<int64_t> next{0};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto work = [&] {
        for (int64_t i; (i = next.fetch_add(1)) < n;) {
            try {
                fn(i);
            } catch (...) {
                std::scoped_lock lock(error_mutex);
                if (!error) error = std::current_exception();
                next = n;  // stop handing out jobs
            }
        }
    };
    std::vector<std::jthread> threads;
    for (int64_t w = 1; w < workers; ++w) threads.emplace_back(work);
    work();
    threads.clear();  // joins them
    if (error) std::rethrow_exception(error);
}

// f(0) .. f(n - 1) side by side for short jobs (pieces of one multiply): macOS: libdispatch's threads, which
// stay alive between calls, so a call costs microseconds, not a thread start each. One thread allowed, or
// elsewhere: in turn. f must not throw.
template <std::invocable<int64_t> F>
void side_by_side(int64_t n, F&& f) {
#if defined(__APPLE__)
    if (n > 1 && num_threads() > 1) {
        struct Job {
            F* fn;
        } job{&f};
        dispatch_apply_f(static_cast<size_t>(n), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &job,
                         [](void* p, size_t i) { (*static_cast<Job*>(p)->fn)(static_cast<int64_t>(i)); });
        return;
    }
#endif
    for (int64_t i = 0; i < n; ++i) f(i);
}

}  // namespace tfi::detail
