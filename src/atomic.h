#pragma once

#include "redner.h"
#include "vector.h"
#include "matrix.h"

// Portable atomic operations that works on both CPU and GPU
// Partly inspired by https://github.com/mbitsnbites/atomic

DEVICE inline int atomic_increment(int *addr) {
#ifdef __CUDA_ARCH__
    return atomicAdd(addr, 1) + 1;
#else
    #if defined(USE_GCC_INTRINSICS)
        return __atomic_add_fetch(addr, 1, __ATOMIC_SEQ_CST);
    #elif defined(USE_MSVC_INTRINSICS)
        return _InterlockedIncrement(reinterpret_cast<volatile long*>(addr));
    #else
        assert(false);
        return 0;
    #endif
#endif
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static inline DEVICE double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val == 0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

#if defined(USE_GCC_INTRINSICS)
    template <typename T0, typename T1>
    DEVICE
    inline T0 atomic_add_(T0 &target, T1 source) {
    #ifdef __CUDA_ARCH__
        return atomicAdd(&target, (T0)source);
    #else
        T0 old_val;
        T0 new_val;
        do {
            old_val = target;
            new_val = old_val + source;
        } while (!__atomic_compare_exchange(&target, &old_val, &new_val, true,
            std::memory_order::memory_order_seq_cst,
            std::memory_order::memory_order_seq_cst));
        return old_val;
    #endif
    }

    DEVICE
    inline
    float atomic_add(float &target, float source) {
        return atomic_add_(target, source);
    }
    DEVICE
    inline
    double atomic_add(double &target, double source) {
        return atomic_add_(target, source);
    }
#elif defined(USE_MSVC_INTRINSICS)
    #define NOMINMAX
    #include <windows.h>
    DEVICE
    static float atomic_add(float &target, float source) {
    #ifdef __CUDA_ARCH__
        return atomicAdd(&target, source);
    #else
        union { int i; float f; } old_val;
        union { int i; float f; } new_val;
        do {
            old_val.f = target;
            new_val.f = old_val.f + (float)source;
        } while (InterlockedCompareExchange((LONG*)&target, (LONG)new_val.i, (LONG)old_val.i) != old_val.i);
        return old_val.f;
    #endif
    }
    DEVICE
    static double atomic_add(double &target, double source) {
    #ifdef __CUDA_ARCH__
        return atomicAdd(&target, (double)source);
    #else
        union { int64_t i; double f; } old_val;
        union { int64_t i; double f; } new_val;
        do {
            old_val.f = target;
            new_val.f = old_val.f + (double)source;
        } while (InterlockedCompareExchange64((LONG64*)&target, (LONG64)new_val.i, (LONG64)old_val.i) != old_val.i);
        return old_val.f;
    #endif
    }
#endif

template <typename T0, typename T1>
DEVICE
inline T0 atomic_add(T0 *target, T1 source) {
    return atomic_add(*target, source);
}

template <typename T0, typename T1>
DEVICE
inline void atomic_add(T0 *target, const TVector2<T1> &source) {
    atomic_add(target[0], (T0)source[0]);
    atomic_add(target[1], (T0)source[1]);
}

template <typename T0, typename T1>
DEVICE
inline void atomic_add(T0 *target, const TVector3<T1> &source) {
    atomic_add(target[0], (T0)source[0]);
    atomic_add(target[1], (T0)source[1]);
    atomic_add(target[2], (T0)source[2]);
}

template <typename T0, typename T1>
DEVICE
inline void atomic_add(T0 *target, const TMatrix3x3<T1> &source) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            atomic_add(target[3 * i + j], (T0)source(i, j));
        }
    }
}

template <typename T0, typename T1>
DEVICE
inline void atomic_add(T0 *target, const TMatrix4x4<T1> &source) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            atomic_add(target[4 * i + j], (T0)source(i, j));
        }
    }
}

void test_atomic();
