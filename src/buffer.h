#pragma once

#include "redner.h"
#include "cuda_utils.h"

#include <vector>
#include <cstdlib>
#include <iostream>

template <typename T>
struct BufferView {
    BufferView(T *data = nullptr, int count = 0) :
        data(data), count(count) {}

    int size() const {
        return count;
    }

    T* begin() {
        return data;
    }
    const T* begin() const {
        return data;
    }
    T* end() {
        return data + count;
    }
    const T* end() const {
        return data + count;
    }
    const T& operator[](int i) const {
        return data[i];
    }
    T& operator[](int i) {
        return data[i];
    }

    T *data;
    int count;
};

/**
 * A wrapper around the CUDA unified memory
 */
template <typename T>
struct Buffer {
private:
    Buffer(const Buffer &buffer) = delete;
public:
    Buffer(bool use_gpu = false, size_t count = 0)
            : use_gpu(use_gpu), data(nullptr), count(count) {
        if (count > 0) {
            if (use_gpu) {
#ifdef __CUDACC__
                checkCuda(cudaMallocManaged(&data, count * sizeof(T)));
#else
                assert(false);
#endif
            } else {
                data = (T*)malloc(count * sizeof(T));
            }
        }
    }

    Buffer(Buffer&& other)
        : use_gpu(std::move(other.use_gpu)),
          data(std::move(other.data)),
          count(std::move(other.count)) {
        other.data = nullptr;
        other.count = 0;
    }

    Buffer& operator=(Buffer &&other) {
        use_gpu = other.use_gpu;
        data = other.data;
        count = other.count;
        other.data = nullptr;
        other.count = 0;
        return *this;
    }

    ~Buffer() {
        if (data != nullptr) {
            if (use_gpu) {
#ifdef __CUDACC__
                checkCuda(cudaFree(data));
#else
                assert(false);
#endif
            } else {
                free(data);
            }
        }
    }

    int size() const { return (int)count; }
    int bytes() const { return (int)count * sizeof(T); }

    T* begin() {
        return data;
    }
    const T* begin() const {
        return data;
    }
    T* end() {
        return data + count;
    }
    const T* end() const {
        return data + count;
    }
    T& operator[](int idx) {
        return data[idx];
    }
    const T& operator[](int idx) const {
        return data[idx];
    }
    BufferView<T> view(int offset, int size) const {
        return BufferView<T>{data + offset, size};
    }
    
    bool use_gpu;
    T* data;
    size_t count;
};
