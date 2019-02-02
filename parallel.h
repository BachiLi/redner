#pragma once

#include "vector.h"

#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <cstdint>
#include <cassert>
#include <algorithm>
// From https://github.com/mmp/pbrt-v3/blob/master/src/core/parallel.h

class Barrier {
  public:
    Barrier(int count) : count(count) { assert(count > 0); }
    ~Barrier() { assert(count == 0); }
    void Wait();

  private:
    std::mutex mutex;
    std::condition_variable cv;
    int count;
};

void parallel_for_host(const std::function<void(int64_t)> &func,
                  int64_t count,
                  int chunkSize = 1);
extern thread_local int ThreadIndex;
void parallel_for_host(
    std::function<void(Vector2i)> func, const Vector2i count);
int num_system_cores();

void parallel_init();
void parallel_cleanup();

#ifdef __CUDACC__
template <typename T>
__global__ void parallel_for_device_kernel(T functor, int count) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count) {
        return;
    }
    functor(idx);
}
template <typename T>
inline void parallel_for_device(T functor,
                                int count,
                                int work_per_thread = 256) {
    if (count <= 0) {
        return;
    }
    auto block_size = work_per_thread;
    auto block_count = idiv_ceil(count, block_size);
    parallel_for_device_kernel<T><<<block_count, block_size>>>(functor, count);
}
#endif

template <typename T>
inline void parallel_for(T functor,
                         int count,
                         bool use_gpu,
                         int work_per_thread = -1) {
    if (work_per_thread == -1) {
        work_per_thread = use_gpu ? 64 : 256;
    }
    if (count <= 0) {
        return;
    }
    if (use_gpu) {
#ifdef __CUDACC__
        auto block_size = work_per_thread;
        auto block_count = idiv_ceil(count, block_size);
        parallel_for_device_kernel<T><<<block_count, block_size>>>(functor, count);
#else
        assert(false);
#endif
    } else {
        auto num_threads = idiv_ceil(count, work_per_thread);
        parallel_for_host([&](int thread_index) {
            auto id_offset = work_per_thread * thread_index;
            auto work_end = std::min(id_offset + work_per_thread, count);
            for (int work_id = id_offset; work_id < work_end; work_id++) {
                auto idx = work_id;
                assert(idx < count);
                functor(idx);
            }
        }, num_threads);
    }
}
