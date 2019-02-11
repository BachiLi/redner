#pragma once

#ifdef __CUDACC__
    #include <thrust/system/cuda/vector.h>
    #include <thrust/system/cuda/execution_policy.h>
#endif
#include <unordered_map>

#ifdef __CUDACC__
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
#else
    #ifdef WIN32
    #define DISPATCH(use_gpu, f, ...) f(thrust::host, ##__VA_ARGS__)
    #define DISPATCH_CACHED(use_gpu, alloc, f, ...) f(thrust::host, ##__VA_ARGS__)
    #else
    #define DISPATCH(use_gpu, f, args...) f(thrust::host, args)
    #define DISPATCH_CACHED(use_gpu, alloc, f, args...) f(thrust::host, args)
    #endif
#endif

#ifdef __CUDACC__
// Assume thrust only allocate one temporary buffer
struct ThrustCachedAllocator {
    using value_type = char;

    int block_size;
    char* block_ptr;
    bool used;

    ThrustCachedAllocator(int init_block_size = 0) {
        block_size = init_block_size;
        block_ptr = nullptr;
        if (block_size > 0) {
            block_ptr = thrust::cuda::malloc<char>(init_block_size).get();
        }
        used = false;
    }

    ~ThrustCachedAllocator() {
        if (block_ptr != nullptr) {
            thrust::cuda::free(thrust::cuda::pointer<char>(block_ptr));
        }
    }

    char* allocate(std::ptrdiff_t num_bytes) {
        assert(!used);
        if (num_bytes > block_size) {
            if (block_ptr != nullptr) {
                thrust::cuda::free(thrust::cuda::pointer<char>(block_ptr));
            }
            block_size = num_bytes;
            block_ptr = thrust::cuda::malloc<char>(block_size).get();
        }
        used = true;
        return block_ptr;
    }

    void deallocate(char* ptr, size_t n) {
        used = false;
    }
};
#else
// Dummy class
struct ThrustCachedAllocator {
    ThrustCachedAllocator(int init_block_size = 0) {}
};
#endif
