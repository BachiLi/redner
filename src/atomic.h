#pragma once

#include "redner.h"
#include "vector.h"
#include "matrix.h"

// Portable atomic operations that works on both CPU and GPU
// Partly inspired by https://github.com/mbitsnbites/atomic

#if defined(USE_MSVC_INTRINSICS)
#include "atomic_msvc.h"
#endif

DEVICE inline int atomic_increment(int *addr) {
#ifdef __CUDA_ARCH__
	return atomicAdd(addr, 1) + 1;
#else
	#if defined(USE_GCC_INTRINSICS)
		return __atomic_add_fetch(addr, 1, __ATOMIC_SEQ_CST);
	#elif defined(USE_MSVC_INTRINSICS)
		return atomic::msvc::interlocked<int>::increment(addr);
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
    T0 old_val;
    T0 new_val;
    do {
        old_val = target;
        new_val = old_val + source;
  #if defined(USE_GCC_INTRINSICS)
    } while (!__atomic_compare_exchange(&target, &old_val, &new_val, true,
        std::memory_order::memory_order_seq_cst,
        std::memory_order::memory_order_seq_cst));
  #elif defined(USE_MSVC_INTRINSICS)
    } while(old_val != atomic::msvc::interlocked<T0>::compare_exchange(&target, new_val, old_val));
  #else
        assert(false);
    } while(false);
  #endif
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
