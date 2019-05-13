#pragma once

#ifdef __CUDACC__
    #include <cuda.h>
    #include <cuda_runtime.h>
#endif
#include <cstdio>
#include <cassert>
#include <limits>

#ifdef __CUDACC__
#define checkCuda(x) do { if((x)!=cudaSuccess) { \
    printf("CUDA Runtime Error: %s at %s:%d\n",\
    cudaGetErrorString(x),__FILE__,__LINE__);\
    exit(1);}} while(0)
#endif

template <typename T>
DEVICE
inline T infinity() {
#ifdef __CUDA_ARCH__
    const unsigned long long ieee754inf = 0x7ff0000000000000;
    return __longlong_as_double(ieee754inf);
#else
    return std::numeric_limits<T>::infinity();
#endif
}

template <>
DEVICE
inline double infinity() {
#ifdef __CUDA_ARCH__
    return __longlong_as_double(0x7ff0000000000000ULL);
#else
    return std::numeric_limits<double>::infinity();
#endif
}

template <>
DEVICE
inline float infinity() {
#ifdef __CUDA_ARCH__
    return __int_as_float(0x7f800000);
#else
    return std::numeric_limits<float>::infinity();
#endif
}

inline void cuda_synchronize() {
#ifdef __CUDACC__
    checkCuda(cudaDeviceSynchronize());
#endif
}
