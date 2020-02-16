#pragma once

#ifdef __NVCC__ 
    #define DEVICE __device__ __host__ 
#else
    #define DEVICE
#endif

#ifndef __NVCC__
    #include <cmath>
    namespace {
        inline float fmodf(float a, float b) {
            return std::fmod(a, b);
        }
        inline double fmod(double a, double b) {
            return std::fmod(a, b);
        }
    }
    using std::isfinite;
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(__xlc__)
#define USE_GCC_INTRINSICS
#elif defined(_MSC_VER)
#define USE_MSVC_INTRINSICS
#include <intrin.h>
// Define some MSVC replacements for the GCC intrinsics
#define msvc_popcount(x) __popcnt(x)
#define msvc_ffs(x) [](unsigned long mask){ unsigned long index; _BitScanForward(&index, mask); return index + 1U; }(x)
#define msvc_clzll(x) [](uint64_t mask){ unsigned long index; _BitScanReverse64(&index, mask); return sizeof(uint64_t) * 8U - (index + 1U); }(x)
#endif

#include <cstdint>
#include <atomic>

// We use Real for most of the internal computation.
// However, for PyTorch interfaces, Optix Prime and Embree queries
// we use float
using Real = double;

template <typename T>
DEVICE
inline T square(const T &x) {
    return x * x;
}

template <typename T>
DEVICE
inline T cubic(const T &x) {
    return x * x * x;
}

template <typename T>
DEVICE
inline T clamp(const T &v, const T &lo, const T &hi) {
    if (v < lo) return lo;
    else if (v > hi) return hi;
    else return v;
}

DEVICE
inline int modulo(int a, int b) {
    auto r = a % b;
    return (r < 0) ? r+b : r;
}

DEVICE
inline float modulo(float a, float b) {
    float r = ::fmodf(a, b);
    return (r < 0.0f) ? r+b : r;
}

DEVICE
inline double modulo(double a, double b) {
    double r = ::fmod(a, b);
    return (r < 0.0) ? r+b : r;
}

template <typename T>
DEVICE
inline T max(const T &a, const T &b) {
    return a > b ? a : b;
}

template <typename T>
DEVICE
inline T min(const T &a, const T &b) {
    return a < b ? a : b;
}

/// Return ceil(x/y) for integers x and y
inline int idiv_ceil(int x, int y) {
    return (x + y-1) / y;
}

template <typename T>
DEVICE
inline void swap_(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

inline double log2(double x) {
    return log(x) / log(Real(2));
}

template <typename T>
DEVICE
inline T safe_acos(const T &x) {
    if (x >= 1) return T(0);
    else if(x <= -1) return T(M_PI);
    return acos(x);
}

DEVICE
inline int clz(uint64_t x) {
#ifdef __CUDA_ARCH__
    return __clzll(x);
#else
  #if defined(USE_GCC_INTRINSICS)
    return x == 0 ? 64 : __builtin_clzll(x);
  #elif defined(USE_MSVC_INTRINSICS)
    return x == 0 ? 64 : msvc_clzll(x);
  #else
    assert(false);
    return 64;
  #endif
#endif
}

DEVICE
inline int ffs(uint8_t x) {
#ifdef __CUDA_ARCH__
    return __ffs(x);
#else
  #if defined(USE_GCC_INTRINSICS)
    return __builtin_ffs(x);
  #elif defined(USE_MSVC_INTRINSICS)
    return msvc_ffs(x);
  #else
    assert(false);
    return 0;
  #endif
#endif
}

DEVICE
inline int popc(uint8_t x) {
#ifdef __CUDA_ARCH__
    return __popc(x);
#else
  #if defined(USE_GCC_INTRINSICS)
  return __builtin_popcount(x);
  #elif defined(USE_MSVC_INTRINSICS)
    return msvc_popcount(x);
  #else
    assert(false);
    return 0;
  #endif
#endif
}

#define STR(x) #x
#define ASSERT(x) if (!(x)) { printf("Assertion failed: (%s), function %s, file %s, line %d.\n", STR(x), __PRETTY_FUNCTION__, __FILE__, __LINE__); abort(); }

