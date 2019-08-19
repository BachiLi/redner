#pragma once

#include "redner.h"
#include "vector.h"
#include "matrix.h"

// Portable atomic operations that works on both CPU and GPU
// Partly inspired by https://github.com/mbitsnbites/atomic

#if defined(__GNUC__) || defined(__clang__) || defined(__xlc__)
#define ATOMIC_USE_GCC
#elif defined(_MSC_VER)
#define ATOMIC_USE_MSVC
#include <intrin.h>
#pragma intrinsic (_InterlockedIncrement)
#endif

DEVICE inline int atomic_increment(int *addr) {
#ifdef __CUDA_ARCH__
	return atomicAdd(addr, 1) + 1;
#else
	#if defined(ATOMIC_USE_GCC)
		return __atomic_add_fetch(addr, 1, __ATOMIC_SEQ_CST);
	#elif defined(ATOMIC_USE_MSVC)
		return _InterlockedIncrement(addr);
	#else
		assert(false);
		return 0;
	#endif
#endif
}


template <typename T0, typename T1>
DEVICE
inline T0 atomic_add(T0 &target, T1 source) {
#ifdef __CUDA_ARCH__
    return atomicAdd(&target, (T0)source);
#else
    // TODO: windows
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
