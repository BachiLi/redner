#pragma once

#include <thrust/memory.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/execution_policy.h>
#include <unordered_map>

#ifdef WIN32
#define DISPATCH(use_gpu, f, ...) \
    ((use_gpu) ? f(thrust::device, ##__VA_ARGS__) : f(thrust::host, ##__VA_ARGS__))
#define DISPATCH_CACHED(use_gpu, alloc, f, ...) \
    ((use_gpu) ? f(thrust::device(alloc), ##__VA_ARGS__) : f(thrust::host(alloc), ##__VA_ARGS__))
#else
#define DISPATCH(use_gpu, f, args...) \
    ((use_gpu) ? f(thrust::device, args) : f(thrust::host, args))
#define DISPATCH_CACHED(use_gpu, alloc, f, args...) \
    ((use_gpu) ? f(thrust::device(alloc), args) : f(thrust::host(alloc), args))
#endif

// Assume thrust only allocate one temporary buffer
struct ThrustCachedAllocator {
    using value_type = char;

    bool use_gpu;
    int block_size;
    char* block_ptr;
    bool used;

    ThrustCachedAllocator(bool use_gpu, int init_block_size = 0) : use_gpu(use_gpu) {
        block_size = init_block_size;
        block_ptr = nullptr;
        if (block_size > 0) {
            if (use_gpu) {
                block_ptr = thrust::device_malloc<char>(block_size).get();
            } else {
                block_ptr = new char[block_size];
            }
        }
        used = false;
    }

    ~ThrustCachedAllocator() {
        if (block_ptr != nullptr) {
            if (use_gpu) {
                thrust::device_free(thrust::device_ptr<char>(block_ptr));
            } else {
                delete[] block_ptr;
            }
        }
    }

    char* allocate(std::ptrdiff_t num_bytes) {
        assert(!used);
        if (num_bytes > block_size) {
            if (block_ptr != nullptr) {
                if (use_gpu) {
                    thrust::device_free(thrust::device_ptr<char>(block_ptr));
                } else {
                    delete[] block_ptr;
                }
            }
            block_size = num_bytes;
            if (use_gpu) {
                block_ptr = thrust::device_malloc<char>(block_size).get();
            } else {
                block_ptr = new char[block_size];
            }
        }
        used = true;
        return block_ptr;
    }

    void deallocate(char* ptr, size_t n) {
        assert(used);
        used = false;
    }
};

