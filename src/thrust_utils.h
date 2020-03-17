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

    static constexpr int max_num_buffers = 2;
    int num_buffers;
    bool use_gpu;
    int block_size[max_num_buffers];
    char* block_ptr[max_num_buffers];
    bool used[max_num_buffers];

    ThrustCachedAllocator(bool use_gpu, int init_block_size = 0) : use_gpu(use_gpu) {
        num_buffers = use_gpu ? 1 : 2;
        for (int i = 0; i < num_buffers; i++) {
            block_size[i] = init_block_size;
            block_ptr[i] = nullptr;
            if (block_size[i] > 0) {
                if (use_gpu) {
                    block_ptr[i] = thrust::device_malloc<char>(block_size[i]).get();
                } else {
                    block_ptr[i] = new char[block_size[i]];
                }
            }
            used[i] = false;
        }
    }

    ~ThrustCachedAllocator() {
        for (int i = 0; i < num_buffers; i++) {
            if (block_ptr[i] != nullptr) {
                if (use_gpu) {
                    thrust::device_free(thrust::device_ptr<char>(block_ptr[i]));
                } else {
                    delete[] block_ptr[i];
                }
            }
        }
    }

    char* allocate(std::ptrdiff_t num_bytes) {
        for (int i = 0; i < num_buffers; i++) {
            // Find first unused buffer
            if (used[i]) {
                continue;
            }
            if (num_bytes > block_size[i]) {
                if (block_ptr[i] != nullptr) {
                    if (use_gpu) {
                        thrust::device_free(thrust::device_ptr<char>(block_ptr[i]));
                    } else {
                        delete[] block_ptr[i];
                    }
                }
                block_size[i] = (int)num_bytes;
                if (use_gpu) {
                    block_ptr[i] = thrust::device_malloc<char>(block_size[i]).get();
                } else {
                    block_ptr[i] = new char[block_size[i]];
                }
            }
            used[i] = true;
            return block_ptr[i];
        }
        assert(false);
        return nullptr;
    }

    void deallocate(char* ptr, size_t n) {
        for (int i = 0; i < num_buffers; i++) {
            if (block_ptr[i] == ptr) {
                assert(used[i]);
                used[i] = false;
                return;
            }
        }
        assert(false);
    }
};

