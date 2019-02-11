#pragma once

#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <unordered_map>

#ifdef WIN32
#define DISPATCH(use_gpu, f, ...) \
    ((use_gpu) ? f(thrust::device, ##__VA_ARGS__) : f(thrust::host, ##__VA_ARGS__))
#define DISPATCH_CACHED(use_gpu, alloc, f, ...) \
    ((use_gpu) ? f(thrust::cuda::par(alloc), ##__VA_ARGS__) : f(thrust::host, ##__VA_ARGS__))
#else
#define DISPATCH(use_gpu, f, args...) \
    ((use_gpu) ? f(thrust::device, args) : f(thrust::host, args))
#define DISPATCH_CACHED(use_gpu, alloc, f, args...) \
    ((use_gpu) ? f(thrust::cuda::par(alloc), args) : f(thrust::host, args))
#endif

// Adopted from https://parallel-computing.pro/index.php/9-cuda/34-thrust-cuda-tip-reuse-temporary-buffers-across-transforms
// Probably can reuse more memory: 
struct ThrustCachedAllocator {
    using value_type = char;

    std::unordered_multimap<std::ptrdiff_t, char*> free_blocks;
    std::unordered_map<char*, std::ptrdiff_t> allocated_blocks;

    ThrustCachedAllocator() {}

    ~ThrustCachedAllocator() {
        free_all();
    }

    char* allocate(std::ptrdiff_t num_bytes) {
        char* result = nullptr;
        // Find a free block with matching number of bytes
        auto free_block = free_blocks.find(num_bytes);
        if (free_block != free_blocks.end()) {
            result = free_block->second;
            free_blocks.erase(free_block);
        } else {
            result = thrust::cuda::malloc<char>(num_bytes).get();
        }
        allocated_blocks.insert({result, num_bytes});
        return result;
    }

    void deallocate(char* ptr, size_t n) {
        auto it = allocated_blocks.find(ptr);
        auto num_bytes = it->second;
        allocated_blocks.erase(it);
        free_blocks.insert({num_bytes, ptr});
    }

    void free_all() {
        for (auto &it : free_blocks) {
            thrust::cuda::free(thrust::cuda::pointer<char>(it.second));
        }
        for (auto &it : allocated_blocks) {
            thrust::cuda::free(thrust::cuda::pointer<char>(it.first));
        }
    }
};

