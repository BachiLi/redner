#pragma once

#include "redner.h"

// Portable atomic operations that works on both CPU and GPU
// Partly inspired by https://github.com/mbitsnbites/atomic

#if defined(__GNUC__) || defined(__clang__) || defined(__xlc__)
#define ATOMIC_USE_GCC
#elif defined(_MSC_VER)
#define ATOMIC_USE_MSVC
#include <intrin.h>
#pragma intrinsic (_InterlockedIncrement)
#endif

DEVICE int atomic_increment(int *addr) {
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
