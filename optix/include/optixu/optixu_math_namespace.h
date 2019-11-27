
/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */
 
 
/**
 * @file   optixu_math_namespace.h
 * @author NVIDIA Corporation
 * @brief  OptiX public API
 *
 * This file implements common mathematical operations on vector types
 * (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 * The syntax is modeled on the Cg standard library.
 *
 * This file has also been modified from the original cutil_math.h file.
 * cutil_math.h is a subset of this file, and you should use this file in place
 * of any cutil_math.h file you wish to use.
*/

#ifndef __optixu_optixu_math_namespace_h__
#define __optixu_optixu_math_namespace_h__

#include "../optix.h"                     // For RT_HOSTDEVICE
#include "../internal/optix_datatypes.h"  // For optix::Ray
#include "optixu_vector_functions.h"
#include "../optix_sizet.h"

#if !defined(_WIN32) && !defined(__CUDACC_RTC__)
// On posix systems uint and ushort are defined when including this file, so we need to
// guarantee this file gets included in order to get these typedefs.
#  include <sys/types.h>
#endif

/** @cond */

// #define these constants such that we are sure
// 32b floats are emitted in ptx
#ifndef M_Ef
#define M_Ef        2.71828182845904523536f
#endif
#ifndef M_LOG2Ef
#define M_LOG2Ef    1.44269504088896340736f
#endif
#ifndef M_LOG10Ef
#define M_LOG10Ef   0.434294481903251827651f
#endif
#ifndef M_LN2f
#define M_LN2f      0.693147180559945309417f
#endif
#ifndef M_LN10f
#define M_LN10f     2.30258509299404568402f
#endif
#ifndef M_PIf
#define M_PIf       3.14159265358979323846f
#endif
#ifndef M_PI_2f
#define M_PI_2f     1.57079632679489661923f
#endif
#ifndef M_PI_4f
#define M_PI_4f     0.785398163397448309616f
#endif
#ifndef M_1_PIf
#define M_1_PIf     0.318309886183790671538f
#endif
#ifndef M_2_PIf
#define M_2_PIf     0.636619772367581343076f
#endif
#ifndef M_2_SQRTPIf
#define M_2_SQRTPIf 1.12837916709551257390f
#endif
#ifndef M_SQRT2f
#define M_SQRT2f    1.41421356237309504880f
#endif
#ifndef M_SQRT1_2f
#define M_SQRT1_2f  0.707106781186547524401f
#endif

/** @endcond */

// __forceinline__ works in cuda, VS, and with gcc.  Leave it as macro in case
// we need to make this per-platform or we want to switch off inlining globally.
#ifndef OPTIXU_INLINE 
#  define OPTIXU_INLINE_DEFINED 1
#  define OPTIXU_INLINE __forceinline__
#endif // OPTIXU_INLINE 

/******************************************************************************/
namespace optix {
#if (defined(_WIN32) && !defined(RT_UINT_USHORT_DEFINED)) || defined(__CUDACC_RTC__)
  // uint and ushort are not already defined on Windows systems or they could have been
  // defined in optixu_math.h.
  typedef unsigned int uint;
  typedef unsigned short ushort;
#else
  // On Posix systems these typedefs are defined in the global namespace, and to avoid
  // conflicts, we'll pull them into this namespace for consistency.
  using ::uint;
  using ::ushort;
#endif //defined(_WIN32)
} // end namespace optix

#if !defined(__CUDACC__)
/* Functions that CUDA provides for device code but are lacking on some host platform */

#include <math.h>

// On systems that declare (but not define) these functions we need to define them here.
// The system versions are not inlinable and cause slower performance.  In addition we
// can't define them in a namespace, because we need to override the one declared extern
// in the global namespace and subsequent overloaded versions need to qualify their call
// with the global namespace to avoid auto-casting from float to float3 and friends.
// Later we pull in the definitions into the optix namespace.
//
// On systems that don't have any version of these functions declared, we go ahead and
// define them in the optix namespace.

// On Windows pre VS 2013 these functions were not declared or defined.
// Define them in optix namespace.
#if defined(_WIN32) && (_MSC_VER < 1800)
#  define OPTIXU_MATH_DEFINE_IN_NAMESPACE
#endif
// On Windows VS 2013+ these functions are declared extern in global namespace.
// Override them by defining them in global namespace.
// Bring them in optix namespace
#if defined(_WIN32) && (_MSC_VER >= 1800)
#  define OPTIXU_MATH_BRING_IN_NAMESPACE
#endif
// On non-windows systems (POSIX) these are declared extern in global namespace.
// Override them by defining them in global namespace.
// Bring them in optix namespace.
#if !defined(_WIN32)
#  define OPTIXU_MATH_BRING_IN_NAMESPACE
#endif

#if defined(OPTIXU_MATH_DEFINE_IN_NAMESPACE)
namespace optix {
#endif

OPTIXU_INLINE float fminf(const float a, const float b)
{
  return a < b ? a : b;
}

OPTIXU_INLINE float fmaxf(const float a, const float b)
{
  return a > b ? a : b;
}

/** copy sign-bit from src value to dst value */
OPTIXU_INLINE float copysignf(const float dst, const float src)
{
  union {
    float f;
    unsigned int i;
  } v1, v2, v3;
  v1.f = src;
  v2.f = dst;
  v3.i = (v2.i & 0x7fffffff) | (v1.i & 0x80000000);

  return v3.f;
}

#if defined(OPTIXU_MATH_DEFINE_IN_NAMESPACE)
} // end namespace optix
#endif

#if defined(OPTIXU_MATH_BRING_IN_NAMESPACE)
namespace optix {
  using ::fminf;
  using ::fmaxf;
  using ::copysignf;
} // end namespace optix
#endif

// Remove these definitions as they are no longer needed, and we don't want them escaping
// this header file.
#undef OPTIXU_MATH_BRING_IN_NAMESPACE
#undef OPTIXU_MATH_DEFINE_IN_NAMESPACE

#endif // #ifndef __CUDACC__

namespace optix {
  // On Posix systems these functions are defined in the global namespace, but we need to
  // pull them into the optix namespace in order for them to be on the same level as
  // the other overloaded functions in optix::.

#if !defined(_WIN32) || defined (__CUDACC__)
  // These functions are in the global namespace on POSIX (not _WIN32) and in CUDA C.
  using ::fminf;
  using ::fmaxf;
  using ::copysignf;
#endif
  using ::expf;
  using ::floorf;

  // These are defined by CUDA in the global namespace.
#ifdef __CUDACC__
  using ::min;
  using ::max;
#else
#if defined(_WIN32) && !defined(NOMINMAX)
#  error "optixu_math_namespace.h needs NOMINMAX defined on windows."
#endif
  OPTIXU_INLINE int max(int a, int b)
  {
      return a > b ? a : b;
  }

  OPTIXU_INLINE int min(int a, int b)
  {
      return a < b ? a : b;
  }

  OPTIXU_INLINE long long max(long long a, long long b)
  {
      return a > b ? a : b;
  }

  OPTIXU_INLINE long long min(long long a, long long b)
  {
      return a < b ? a : b;
  }

  OPTIXU_INLINE unsigned int max(unsigned int a, unsigned int b)
  {
      return a > b ? a : b;
  }

  OPTIXU_INLINE unsigned int min(unsigned int a, unsigned int b)
  {
      return a < b ? a : b;
  }

  OPTIXU_INLINE unsigned long long max(unsigned long long a, unsigned long long b)
  {
      return a > b ? a : b;
  }

  OPTIXU_INLINE unsigned long long min(unsigned long long a, unsigned long long b)
  {
      return a < b ? a : b;
  }
#endif

} // end namespace optix


namespace optix {

/* Bit preserving casting functions */
/******************************************************************************/

#ifdef __CUDACC__

  using ::float_as_int;
  using ::int_as_float;

#else

/** Bit preserving casting function */
OPTIXU_INLINE int float_as_int(const float f)
{
  union {
    float f;
    int i;
  } v1;

  v1.f = f;
  return v1.i;
}

/** Bit preserving casting function */
OPTIXU_INLINE float int_as_float(int i)
{
  union {
    float f;
    int i;
  } v1;

  v1.i = i;
  return v1.f;
}

#endif 


/* float functions */
/******************************************************************************/

/** lerp */
OPTIXU_INLINE RT_HOSTDEVICE float lerp(const float a, const float b, const float t)
{
  return a + t*(b-a);
}

/** bilerp */
OPTIXU_INLINE RT_HOSTDEVICE float bilerp(const float x00, const float x10, const float x01, const float x11,
                                         const float u, const float v)
{
  return lerp( lerp( x00, x10, u ), lerp( x01, x11, u ), v );
}

/** clamp */
OPTIXU_INLINE RT_HOSTDEVICE float clamp(const float f, const float a, const float b)
{
  return fmaxf(a, fminf(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE float getByIndex(const float1& v, int i)
{
  return ((float*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(float1& v, int i, float x)
{
  ((float*)(&v))[i] = x;
}
  
/* float2 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float2 make_float2(const float s)
{
  return make_float2(s, s);
}
OPTIXU_INLINE RT_HOSTDEVICE float2 make_float2(const int2& a)
{
  return make_float2(float(a.x), float(a.y));
}
OPTIXU_INLINE RT_HOSTDEVICE float2 make_float2(const uint2& a)
{
  return make_float2(float(a.x), float(a.y));
}
/** @} */

/** negate */
OPTIXU_INLINE RT_HOSTDEVICE float2 operator-(const float2& a)
{
  return make_float2(-a.x, -a.y);
}

/** min 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float2 fminf(const float2& a, const float2& b)
{
  return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
}
OPTIXU_INLINE RT_HOSTDEVICE float fminf(const float2& a)
{
  return fminf(a.x, a.y);
}
/** @} */

/** max 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float2 fmaxf(const float2& a, const float2& b)
{
  return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}
OPTIXU_INLINE RT_HOSTDEVICE float fmaxf(const float2& a)
{
  return fmaxf(a.x, a.y);
}
/** @} */

/** add 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float2 operator+(const float2& a, const float2& b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE float2 operator+(const float2& a, const float b)
{
  return make_float2(a.x + b, a.y + b);
}
OPTIXU_INLINE RT_HOSTDEVICE float2 operator+(const float a, const float2& b)
{
  return make_float2(a + b.x, a + b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator+=(float2& a, const float2& b)
{
  a.x += b.x; a.y += b.y;
}
/** @} */

/** subtract 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float2 operator-(const float2& a, const float2& b)
{
  return make_float2(a.x - b.x, a.y - b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE float2 operator-(const float2& a, const float b)
{
  return make_float2(a.x - b, a.y - b);
}
OPTIXU_INLINE RT_HOSTDEVICE float2 operator-(const float a, const float2& b)
{
  return make_float2(a - b.x, a - b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator-=(float2& a, const float2& b)
{
  a.x -= b.x; a.y -= b.y;
}
/** @} */

/** multiply 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float2 operator*(const float2& a, const float2& b)
{
  return make_float2(a.x * b.x, a.y * b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE float2 operator*(const float2& a, const float s)
{
  return make_float2(a.x * s, a.y * s);
}
OPTIXU_INLINE RT_HOSTDEVICE float2 operator*(const float s, const float2& a)
{
  return make_float2(a.x * s, a.y * s);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(float2& a, const float2& s)
{
  a.x *= s.x; a.y *= s.y;
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(float2& a, const float s)
{
  a.x *= s; a.y *= s;
}
/** @} */

/** divide 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float2 operator/(const float2& a, const float2& b)
{
  return make_float2(a.x / b.x, a.y / b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE float2 operator/(const float2& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
OPTIXU_INLINE RT_HOSTDEVICE float2 operator/(const float s, const float2& a)
{
  return make_float2( s/a.x, s/a.y );
}
OPTIXU_INLINE RT_HOSTDEVICE void operator/=(float2& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}
/** @} */

/** lerp */
OPTIXU_INLINE RT_HOSTDEVICE float2 lerp(const float2& a, const float2& b, const float t)
{
  return a + t*(b-a);
}

/** bilerp */
OPTIXU_INLINE RT_HOSTDEVICE float2 bilerp(const float2& x00, const float2& x10, const float2& x01, const float2& x11,
                                          const float u, const float v)
{
  return lerp( lerp( x00, x10, u ), lerp( x01, x11, u ), v );
}

/** clamp 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float2 clamp(const float2& v, const float a, const float b)
{
  return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

OPTIXU_INLINE RT_HOSTDEVICE float2 clamp(const float2& v, const float2& a, const float2& b)
{
  return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** dot product */
OPTIXU_INLINE RT_HOSTDEVICE float dot(const float2& a, const float2& b)
{
  return a.x * b.x + a.y * b.y;
}

/** length */
OPTIXU_INLINE RT_HOSTDEVICE float length(const float2& v)
{
  return sqrtf(dot(v, v));
}

/** normalize */
OPTIXU_INLINE RT_HOSTDEVICE float2 normalize(const float2& v)
{
  float invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}

/** floor */
OPTIXU_INLINE RT_HOSTDEVICE float2 floor(const float2& v)
{
  return make_float2(::floorf(v.x), ::floorf(v.y));
}

/** reflect */
OPTIXU_INLINE RT_HOSTDEVICE float2 reflect(const float2& i, const float2& n)
{
  return i - 2.0f * n * dot(n,i);
}

/** Faceforward
* Returns N if dot(i, nref) > 0; else -N; 
* Typical usage is N = faceforward(N, -ray.dir, N);
* Note that this is opposite of what faceforward does in Cg and GLSL */
OPTIXU_INLINE RT_HOSTDEVICE float2 faceforward(const float2& n, const float2& i, const float2& nref)
{
  return n * copysignf( 1.0f, dot(i, nref) );
}

/** exp */
OPTIXU_INLINE RT_HOSTDEVICE float2 expf(const float2& v)
{
  return make_float2(::expf(v.x), ::expf(v.y));
}

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE float getByIndex(const float2& v, int i)
{
  return ((float*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(float2& v, int i, float x)
{
  ((float*)(&v))[i] = x;
}


/* float3 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float3 make_float3(const float s)
{
  return make_float3(s, s, s);
}
OPTIXU_INLINE RT_HOSTDEVICE float3 make_float3(const float2& a)
{
  return make_float3(a.x, a.y, 0.0f);
}
OPTIXU_INLINE RT_HOSTDEVICE float3 make_float3(const int3& a)
{
  return make_float3(float(a.x), float(a.y), float(a.z));
}
OPTIXU_INLINE RT_HOSTDEVICE float3 make_float3(const uint3& a)
{
  return make_float3(float(a.x), float(a.y), float(a.z));
}
/** @} */

/** negate */
OPTIXU_INLINE RT_HOSTDEVICE float3 operator-(const float3& a)
{
  return make_float3(-a.x, -a.y, -a.z);
}

/** min 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float3 fminf(const float3& a, const float3& b)
{
  return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
OPTIXU_INLINE RT_HOSTDEVICE float fminf(const float3& a)
{
  return fminf(fminf(a.x, a.y), a.z);
}
/** @} */

/** max 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float3 fmaxf(const float3& a, const float3& b)
{
  return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
OPTIXU_INLINE RT_HOSTDEVICE float fmaxf(const float3& a)
{
  return fmaxf(fmaxf(a.x, a.y), a.z);
}
/** @} */

/** add 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float3 operator+(const float3& a, const float3& b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE float3 operator+(const float3& a, const float b)
{
  return make_float3(a.x + b, a.y + b, a.z + b);
}
OPTIXU_INLINE RT_HOSTDEVICE float3 operator+(const float a, const float3& b)
{
  return make_float3(a + b.x, a + b.y, a + b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator+=(float3& a, const float3& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float3 operator-(const float3& a, const float3& b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE float3 operator-(const float3& a, const float b)
{
  return make_float3(a.x - b, a.y - b, a.z - b);
}
OPTIXU_INLINE RT_HOSTDEVICE float3 operator-(const float a, const float3& b)
{
  return make_float3(a - b.x, a - b.y, a - b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator-=(float3& a, const float3& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float3 operator*(const float3& a, const float3& b)
{
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE float3 operator*(const float3& a, const float s)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}
OPTIXU_INLINE RT_HOSTDEVICE float3 operator*(const float s, const float3& a)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(float3& a, const float3& s)
{
  a.x *= s.x; a.y *= s.y; a.z *= s.z;
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(float3& a, const float s)
{
  a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float3 operator/(const float3& a, const float3& b)
{
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE float3 operator/(const float3& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
OPTIXU_INLINE RT_HOSTDEVICE float3 operator/(const float s, const float3& a)
{
  return make_float3( s/a.x, s/a.y, s/a.z );
}
OPTIXU_INLINE RT_HOSTDEVICE void operator/=(float3& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}
/** @} */

/** lerp */
OPTIXU_INLINE RT_HOSTDEVICE float3 lerp(const float3& a, const float3& b, const float t)
{
  return a + t*(b-a);
}

/** bilerp */
OPTIXU_INLINE RT_HOSTDEVICE float3 bilerp(const float3& x00, const float3& x10, const float3& x01, const float3& x11,
                                          const float u, const float v)
{
  return lerp( lerp( x00, x10, u ), lerp( x01, x11, u ), v );
}

/** clamp 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float3 clamp(const float3& v, const float a, const float b)
{
  return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

OPTIXU_INLINE RT_HOSTDEVICE float3 clamp(const float3& v, const float3& a, const float3& b)
{
  return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** dot product */
OPTIXU_INLINE RT_HOSTDEVICE float dot(const float3& a, const float3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/** cross product */
OPTIXU_INLINE RT_HOSTDEVICE float3 cross(const float3& a, const float3& b)
{
  return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

/** length */
OPTIXU_INLINE RT_HOSTDEVICE float length(const float3& v)
{
  return sqrtf(dot(v, v));
}

/** normalize */
OPTIXU_INLINE RT_HOSTDEVICE float3 normalize(const float3& v)
{
  float invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}

/** floor */
OPTIXU_INLINE RT_HOSTDEVICE float3 floor(const float3& v)
{
  return make_float3(::floorf(v.x), ::floorf(v.y), ::floorf(v.z));
}

/** reflect */
OPTIXU_INLINE RT_HOSTDEVICE float3 reflect(const float3& i, const float3& n)
{
  return i - 2.0f * n * dot(n,i);
}

/** Faceforward
* Returns N if dot(i, nref) > 0; else -N;
* Typical usage is N = faceforward(N, -ray.dir, N);
* Note that this is opposite of what faceforward does in Cg and GLSL */
OPTIXU_INLINE RT_HOSTDEVICE float3 faceforward(const float3& n, const float3& i, const float3& nref)
{
  return n * copysignf( 1.0f, dot(i, nref) );
}

/** exp */
OPTIXU_INLINE RT_HOSTDEVICE float3 expf(const float3& v)
{
  return make_float3(::expf(v.x), ::expf(v.y), ::expf(v.z));
}

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE float getByIndex(const float3& v, int i)
{
  return ((float*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(float3& v, int i, float x)
{
  ((float*)(&v))[i] = x;
}
  
/* float4 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float4 make_float4(const float s)
{
  return make_float4(s, s, s, s);
}
OPTIXU_INLINE RT_HOSTDEVICE float4 make_float4(const float3& a)
{
  return make_float4(a.x, a.y, a.z, 0.0f);
}
OPTIXU_INLINE RT_HOSTDEVICE float4 make_float4(const int4& a)
{
  return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
OPTIXU_INLINE RT_HOSTDEVICE float4 make_float4(const uint4& a)
{
  return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
/** @} */

/** negate */
OPTIXU_INLINE RT_HOSTDEVICE float4 operator-(const float4& a)
{
  return make_float4(-a.x, -a.y, -a.z, -a.w);
}

/** min 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float4 fminf(const float4& a, const float4& b)
{
  return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}
OPTIXU_INLINE RT_HOSTDEVICE float fminf(const float4& a)
{
  return fminf(fminf(a.x, a.y), fminf(a.z, a.w));
}
/** @} */

/** max 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float4 fmaxf(const float4& a, const float4& b)
{
  return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}
OPTIXU_INLINE RT_HOSTDEVICE float fmaxf(const float4& a)
{
  return fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w));
}
/** @} */

/** add 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float4 operator+(const float4& a, const float4& b)
{
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE float4 operator+(const float4& a, const float b)
{
  return make_float4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
OPTIXU_INLINE RT_HOSTDEVICE float4 operator+(const float a, const float4& b)
{
  return make_float4(a + b.x, a + b.y, a + b.z,  a + b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator+=(float4& a, const float4& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
/** @} */

/** subtract 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float4 operator-(const float4& a, const float4& b)
{
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE float4 operator-(const float4& a, const float b)
{
  return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
OPTIXU_INLINE RT_HOSTDEVICE float4 operator-(const float a, const float4& b)
{
  return make_float4(a - b.x, a - b.y, a - b.z,  a - b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator-=(float4& a, const float4& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float4 operator*(const float4& a, const float4& s)
{
  return make_float4(a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w);
}
OPTIXU_INLINE RT_HOSTDEVICE float4 operator*(const float4& a, const float s)
{
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
OPTIXU_INLINE RT_HOSTDEVICE float4 operator*(const float s, const float4& a)
{
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(float4& a, const float4& s)
{
  a.x *= s.x; a.y *= s.y; a.z *= s.z; a.w *= s.w;
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(float4& a, const float s)
{
  a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
/** @} */

/** divide 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float4 operator/(const float4& a, const float4& b)
{
  return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE float4 operator/(const float4& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
OPTIXU_INLINE RT_HOSTDEVICE float4 operator/(const float s, const float4& a)
{
  return make_float4( s/a.x, s/a.y, s/a.z, s/a.w );
}
OPTIXU_INLINE RT_HOSTDEVICE void operator/=(float4& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}
/** @} */

/** lerp */
OPTIXU_INLINE RT_HOSTDEVICE float4 lerp(const float4& a, const float4& b, const float t)
{
  return a + t*(b-a);
}

/** bilerp */
OPTIXU_INLINE RT_HOSTDEVICE float4 bilerp(const float4& x00, const float4& x10, const float4& x01, const float4& x11,
                                          const float u, const float v)
{
  return lerp( lerp( x00, x10, u ), lerp( x01, x11, u ), v );
}

/** clamp 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE float4 clamp(const float4& v, const float a, const float b)
{
  return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

OPTIXU_INLINE RT_HOSTDEVICE float4 clamp(const float4& v, const float4& a, const float4& b)
{
  return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** dot product */
OPTIXU_INLINE RT_HOSTDEVICE float dot(const float4& a, const float4& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/** length */
OPTIXU_INLINE RT_HOSTDEVICE float length(const float4& r)
{
  return sqrtf(dot(r, r));
}

/** normalize */
OPTIXU_INLINE RT_HOSTDEVICE float4 normalize(const float4& v)
{
  float invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}

/** floor */
OPTIXU_INLINE RT_HOSTDEVICE float4 floor(const float4& v)
{
  return make_float4(::floorf(v.x), ::floorf(v.y), ::floorf(v.z), ::floorf(v.w));
}

/** reflect */
OPTIXU_INLINE RT_HOSTDEVICE float4 reflect(const float4& i, const float4& n)
{
  return i - 2.0f * n * dot(n,i);
}

/** 
* Faceforward
* Returns N if dot(i, nref) > 0; else -N;
* Typical usage is N = faceforward(N, -ray.dir, N);
* Note that this is opposite of what faceforward does in Cg and GLSL 
*/
OPTIXU_INLINE RT_HOSTDEVICE float4 faceforward(const float4& n, const float4& i, const float4& nref)
{
  return n * copysignf( 1.0f, dot(i, nref) );
}

/** exp */
OPTIXU_INLINE RT_HOSTDEVICE float4 expf(const float4& v)
{
  return make_float4(::expf(v.x), ::expf(v.y), ::expf(v.z), ::expf(v.w));
}

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE float getByIndex(const float4& v, int i)
{
  return ((float*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(float4& v, int i, float x)
{
  ((float*)(&v))[i] = x;
}
  
  
/* int functions */
/******************************************************************************/

/** clamp */
OPTIXU_INLINE RT_HOSTDEVICE int clamp(const int f, const int a, const int b)
{
  return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE int getByIndex(const int1& v, int i)
{
  return ((int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(int1& v, int i, int x)
{
  ((int*)(&v))[i] = x;
}
  

/* int2 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int2 make_int2(const int s)
{
  return make_int2(s, s);
}
OPTIXU_INLINE RT_HOSTDEVICE int2 make_int2(const float2& a)
{
  return make_int2(int(a.x), int(a.y));
}
/** @} */

/** negate */
OPTIXU_INLINE RT_HOSTDEVICE int2 operator-(const int2& a)
{
  return make_int2(-a.x, -a.y);
}

/** min */
OPTIXU_INLINE RT_HOSTDEVICE int2 min(const int2& a, const int2& b)
{
  return make_int2(min(a.x,b.x), min(a.y,b.y));
}

/** max */
OPTIXU_INLINE RT_HOSTDEVICE int2 max(const int2& a, const int2& b)
{
  return make_int2(max(a.x,b.x), max(a.y,b.y));
}

/** add 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int2 operator+(const int2& a, const int2& b)
{
  return make_int2(a.x + b.x, a.y + b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator+=(int2& a, const int2& b)
{
  a.x += b.x; a.y += b.y;
}
/** @} */

/** subtract 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int2 operator-(const int2& a, const int2& b)
{
  return make_int2(a.x - b.x, a.y - b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE int2 operator-(const int2& a, const int b)
{
  return make_int2(a.x - b, a.y - b);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator-=(int2& a, const int2& b)
{
  a.x -= b.x; a.y -= b.y;
}
/** @} */

/** multiply 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int2 operator*(const int2& a, const int2& b)
{
  return make_int2(a.x * b.x, a.y * b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE int2 operator*(const int2& a, const int s)
{
  return make_int2(a.x * s, a.y * s);
}
OPTIXU_INLINE RT_HOSTDEVICE int2 operator*(const int s, const int2& a)
{
  return make_int2(a.x * s, a.y * s);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(int2& a, const int s)
{
  a.x *= s; a.y *= s;
}
/** @} */

/** clamp 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int2 clamp(const int2& v, const int a, const int b)
{
  return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}

OPTIXU_INLINE RT_HOSTDEVICE int2 clamp(const int2& v, const int2& a, const int2& b)
{
  return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE bool operator==(const int2& a, const int2& b)
{
  return a.x == b.x && a.y == b.y;
}

OPTIXU_INLINE RT_HOSTDEVICE bool operator!=(const int2& a, const int2& b)
{
  return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE int getByIndex(const int2& v, int i)
{
  return ((int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(int2& v, int i, int x)
{
  ((int*)(&v))[i] = x;
}
  

/* int3 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int3 make_int3(const int s)
{
  return make_int3(s, s, s);
}
OPTIXU_INLINE RT_HOSTDEVICE int3 make_int3(const float3& a)
{
  return make_int3(int(a.x), int(a.y), int(a.z));
}
/** @} */

/** negate */
OPTIXU_INLINE RT_HOSTDEVICE int3 operator-(const int3& a)
{
  return make_int3(-a.x, -a.y, -a.z);
}

/** min */
OPTIXU_INLINE RT_HOSTDEVICE int3 min(const int3& a, const int3& b)
{
  return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

/** max */
OPTIXU_INLINE RT_HOSTDEVICE int3 max(const int3& a, const int3& b)
{
  return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

/** add 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int3 operator+(const int3& a, const int3& b)
{
  return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator+=(int3& a, const int3& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int3 operator-(const int3& a, const int3& b)
{
  return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

OPTIXU_INLINE RT_HOSTDEVICE void operator-=(int3& a, const int3& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int3 operator*(const int3& a, const int3& b)
{
  return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE int3 operator*(const int3& a, const int s)
{
  return make_int3(a.x * s, a.y * s, a.z * s);
}
OPTIXU_INLINE RT_HOSTDEVICE int3 operator*(const int s, const int3& a)
{
  return make_int3(a.x * s, a.y * s, a.z * s);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(int3& a, const int s)
{
  a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int3 operator/(const int3& a, const int3& b)
{
  return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE int3 operator/(const int3& a, const int s)
{
  return make_int3(a.x / s, a.y / s, a.z / s);
}
OPTIXU_INLINE RT_HOSTDEVICE int3 operator/(const int s, const int3& a)
{
  return make_int3(s /a.x, s / a.y, s / a.z);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator/=(int3& a, const int s)
{
  a.x /= s; a.y /= s; a.z /= s;
}
/** @} */

/** clamp 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int3 clamp(const int3& v, const int a, const int b)
{
  return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

OPTIXU_INLINE RT_HOSTDEVICE int3 clamp(const int3& v, const int3& a, const int3& b)
{
  return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** equality 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE bool operator==(const int3& a, const int3& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

OPTIXU_INLINE RT_HOSTDEVICE bool operator!=(const int3& a, const int3& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE int getByIndex(const int3& v, int i)
{
  return ((int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(int3& v, int i, int x)
{
  ((int*)(&v))[i] = x;
}
  

/* int4 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int4 make_int4(const int s)
{
  return make_int4(s, s, s, s);
}
OPTIXU_INLINE RT_HOSTDEVICE int4 make_int4(const float4& a)
{
  return make_int4((int)a.x, (int)a.y, (int)a.z, (int)a.w);
}
/** @} */

/** negate */
OPTIXU_INLINE RT_HOSTDEVICE int4 operator-(const int4& a)
{
  return make_int4(-a.x, -a.y, -a.z, -a.w);
}

/** min */
OPTIXU_INLINE RT_HOSTDEVICE int4 min(const int4& a, const int4& b)
{
  return make_int4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

/** max */
OPTIXU_INLINE RT_HOSTDEVICE int4 max(const int4& a, const int4& b)
{
  return make_int4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

/** add 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int4 operator+(const int4& a, const int4& b)
{
  return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator+=(int4& a, const int4& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
/** @} */

/** subtract 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int4 operator-(const int4& a, const int4& b)
{
  return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

OPTIXU_INLINE RT_HOSTDEVICE void operator-=(int4& a, const int4& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int4 operator*(const int4& a, const int4& b)
{
  return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE int4 operator*(const int4& a, const int s)
{
  return make_int4(a.x * s, a.y * s, a.z * s, a.w * s);
}
OPTIXU_INLINE RT_HOSTDEVICE int4 operator*(const int s, const int4& a)
{
  return make_int4(a.x * s, a.y * s, a.z * s, a.w * s);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(int4& a, const int s)
{
  a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
/** @} */

/** divide 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int4 operator/(const int4& a, const int4& b)
{
  return make_int4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE int4 operator/(const int4& a, const int s)
{
  return make_int4(a.x / s, a.y / s, a.z / s, a.w / s);
}
OPTIXU_INLINE RT_HOSTDEVICE int4 operator/(const int s, const int4& a)
{
  return make_int4(s / a.x, s / a.y, s / a.z, s / a.w);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator/=(int4& a, const int s)
{
  a.x /= s; a.y /= s; a.z /= s; a.w /= s;
}
/** @} */

/** clamp 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int4 clamp(const int4& v, const int a, const int b)
{
  return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

OPTIXU_INLINE RT_HOSTDEVICE int4 clamp(const int4& v, const int4& a, const int4& b)
{
  return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE bool operator==(const int4& a, const int4& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

OPTIXU_INLINE RT_HOSTDEVICE bool operator!=(const int4& a, const int4& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE int getByIndex(const int4& v, int i)
{
  return ((int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(int4& v, int i, int x)
{
  ((int*)(&v))[i] = x;
}
  

/* uint functions */
/******************************************************************************/

/** clamp */
OPTIXU_INLINE RT_HOSTDEVICE unsigned int clamp(const unsigned int f, const unsigned int a, const unsigned int b)
{
  return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE unsigned int getByIndex(const uint1& v, unsigned int i)
{
  return ((unsigned int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(uint1& v, int i, unsigned int x)
{
  ((unsigned int*)(&v))[i] = x;
}
  

/* uint2 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint2 make_uint2(const unsigned int s)
{
  return make_uint2(s, s);
}
OPTIXU_INLINE RT_HOSTDEVICE uint2 make_uint2(const float2& a)
{
  return make_uint2((unsigned int)a.x, (unsigned int)a.y);
}
/** @} */

/** min */
OPTIXU_INLINE RT_HOSTDEVICE uint2 min(const uint2& a, const uint2& b)
{
  return make_uint2(min(a.x,b.x), min(a.y,b.y));
}

/** max */
OPTIXU_INLINE RT_HOSTDEVICE uint2 max(const uint2& a, const uint2& b)
{
  return make_uint2(max(a.x,b.x), max(a.y,b.y));
}

/** add
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint2 operator+(const uint2& a, const uint2& b)
{
  return make_uint2(a.x + b.x, a.y + b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator+=(uint2& a, const uint2& b)
{
  a.x += b.x; a.y += b.y;
}
/** @} */

/** subtract
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint2 operator-(const uint2& a, const uint2& b)
{
  return make_uint2(a.x - b.x, a.y - b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE uint2 operator-(const uint2& a, const unsigned int b)
{
  return make_uint2(a.x - b, a.y - b);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator-=(uint2& a, const uint2& b)
{
  a.x -= b.x; a.y -= b.y;
}
/** @} */

/** multiply
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint2 operator*(const uint2& a, const uint2& b)
{
  return make_uint2(a.x * b.x, a.y * b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE uint2 operator*(const uint2& a, const unsigned int s)
{
  return make_uint2(a.x * s, a.y * s);
}
OPTIXU_INLINE RT_HOSTDEVICE uint2 operator*(const unsigned int s, const uint2& a)
{
  return make_uint2(a.x * s, a.y * s);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(uint2& a, const unsigned int s)
{
  a.x *= s; a.y *= s;
}
/** @} */

/** clamp
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint2 clamp(const uint2& v, const unsigned int a, const unsigned int b)
{
  return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}

OPTIXU_INLINE RT_HOSTDEVICE uint2 clamp(const uint2& v, const uint2& a, const uint2& b)
{
  return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE bool operator==(const uint2& a, const uint2& b)
{
  return a.x == b.x && a.y == b.y;
}

OPTIXU_INLINE RT_HOSTDEVICE bool operator!=(const uint2& a, const uint2& b)
{
  return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE unsigned int getByIndex(const uint2& v, unsigned int i)
{
  return ((unsigned int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(uint2& v, int i, unsigned int x)
{
  ((unsigned int*)(&v))[i] = x;
}
  

/* uint3 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint3 make_uint3(const unsigned int s)
{
  return make_uint3(s, s, s);
}
OPTIXU_INLINE RT_HOSTDEVICE uint3 make_uint3(const float3& a)
{
  return make_uint3((unsigned int)a.x, (unsigned int)a.y, (unsigned int)a.z);
}
/** @} */

/** min */
OPTIXU_INLINE RT_HOSTDEVICE uint3 min(const uint3& a, const uint3& b)
{
  return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

/** max */
OPTIXU_INLINE RT_HOSTDEVICE uint3 max(const uint3& a, const uint3& b)
{
  return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

/** add 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint3 operator+(const uint3& a, const uint3& b)
{
  return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator+=(uint3& a, const uint3& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint3 operator-(const uint3& a, const uint3& b)
{
  return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

OPTIXU_INLINE RT_HOSTDEVICE void operator-=(uint3& a, const uint3& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint3 operator*(const uint3& a, const uint3& b)
{
  return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE uint3 operator*(const uint3& a, const unsigned int s)
{
  return make_uint3(a.x * s, a.y * s, a.z * s);
}
OPTIXU_INLINE RT_HOSTDEVICE uint3 operator*(const unsigned int s, const uint3& a)
{
  return make_uint3(a.x * s, a.y * s, a.z * s);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(uint3& a, const unsigned int s)
{
  a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint3 operator/(const uint3& a, const uint3& b)
{
  return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE uint3 operator/(const uint3& a, const unsigned int s)
{
  return make_uint3(a.x / s, a.y / s, a.z / s);
}
OPTIXU_INLINE RT_HOSTDEVICE uint3 operator/(const unsigned int s, const uint3& a)
{
  return make_uint3(s / a.x, s / a.y, s / a.z);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator/=(uint3& a, const unsigned int s)
{
  a.x /= s; a.y /= s; a.z /= s;
}
/** @} */

/** clamp 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint3 clamp(const uint3& v, const unsigned int a, const unsigned int b)
{
  return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

OPTIXU_INLINE RT_HOSTDEVICE uint3 clamp(const uint3& v, const uint3& a, const uint3& b)
{
  return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** equality 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE bool operator==(const uint3& a, const uint3& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

OPTIXU_INLINE RT_HOSTDEVICE bool operator!=(const uint3& a, const uint3& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory 
*/
OPTIXU_INLINE RT_HOSTDEVICE unsigned int getByIndex(const uint3& v, unsigned int i)
{
  return ((unsigned int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory 
*/
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(uint3& v, int i, unsigned int x)
{
  ((unsigned int*)(&v))[i] = x;
}
  

/* uint4 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint4 make_uint4(const unsigned int s)
{
  return make_uint4(s, s, s, s);
}
OPTIXU_INLINE RT_HOSTDEVICE uint4 make_uint4(const float4& a)
{
  return make_uint4((unsigned int)a.x, (unsigned int)a.y, (unsigned int)a.z, (unsigned int)a.w);
}
/** @} */

/** min
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint4 min(const uint4& a, const uint4& b)
{
  return make_uint4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}
/** @} */

/** max 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint4 max(const uint4& a, const uint4& b)
{
  return make_uint4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}
/** @} */

/** add
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint4 operator+(const uint4& a, const uint4& b)
{
  return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator+=(uint4& a, const uint4& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
/** @} */

/** subtract 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint4 operator-(const uint4& a, const uint4& b)
{
  return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

OPTIXU_INLINE RT_HOSTDEVICE void operator-=(uint4& a, const uint4& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint4 operator*(const uint4& a, const uint4& b)
{
  return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE uint4 operator*(const uint4& a, const unsigned int s)
{
  return make_uint4(a.x * s, a.y * s, a.z * s, a.w * s);
}
OPTIXU_INLINE RT_HOSTDEVICE uint4 operator*(const unsigned int s, const uint4& a)
{
  return make_uint4(a.x * s, a.y * s, a.z * s, a.w * s);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(uint4& a, const unsigned int s)
{
  a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
/** @} */

/** divide 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint4 operator/(const uint4& a, const uint4& b)
{
  return make_uint4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE uint4 operator/(const uint4& a, const unsigned int s)
{
  return make_uint4(a.x / s, a.y / s, a.z / s, a.w / s);
}
OPTIXU_INLINE RT_HOSTDEVICE uint4 operator/(const unsigned int s, const uint4& a)
{
  return make_uint4(s / a.x, s / a.y, s / a.z, s / a.w);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator/=(uint4& a, const unsigned int s)
{
  a.x /= s; a.y /= s; a.z /= s; a.w /= s;
}
/** @} */

/** clamp 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE uint4 clamp(const uint4& v, const unsigned int a, const unsigned int b)
{
  return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

OPTIXU_INLINE RT_HOSTDEVICE uint4 clamp(const uint4& v, const uint4& a, const uint4& b)
{
  return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE bool operator==(const uint4& a, const uint4& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

OPTIXU_INLINE RT_HOSTDEVICE bool operator!=(const uint4& a, const uint4& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory 
*/
OPTIXU_INLINE RT_HOSTDEVICE unsigned int getByIndex(const uint4& v, unsigned int i)
{
  return ((unsigned int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory 
*/
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(uint4& v, int i, unsigned int x)
{
  ((unsigned int*)(&v))[i] = x;
}

/* long long functions */
/******************************************************************************/

/** clamp */
OPTIXU_INLINE RT_HOSTDEVICE long long clamp(const long long f, const long long a, const long long b)
{
    return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE long long getByIndex(const longlong1& v, int i)
{
    return ((long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(longlong1& v, int i, long long x)
{
    ((long long*)(&v))[i] = x;
}


/* longlong2 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong2 make_longlong2(const long long s)
{
    return make_longlong2(s, s);
}
OPTIXU_INLINE RT_HOSTDEVICE longlong2 make_longlong2(const float2& a)
{
    return make_longlong2(int(a.x), int(a.y));
}
/** @} */

/** negate */
OPTIXU_INLINE RT_HOSTDEVICE longlong2 operator-(const longlong2& a)
{
    return make_longlong2(-a.x, -a.y);
}

/** min */
OPTIXU_INLINE RT_HOSTDEVICE longlong2 min(const longlong2& a, const longlong2& b)
{
    return make_longlong2(min(a.x, b.x), min(a.y, b.y));
}

/** max */
OPTIXU_INLINE RT_HOSTDEVICE longlong2 max(const longlong2& a, const longlong2& b)
{
    return make_longlong2(max(a.x, b.x), max(a.y, b.y));
}

/** add
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong2 operator+(const longlong2& a, const longlong2& b)
{
    return make_longlong2(a.x + b.x, a.y + b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator+=(longlong2& a, const longlong2& b)
{
    a.x += b.x; a.y += b.y;
}
/** @} */

/** subtract
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong2 operator-(const longlong2& a, const longlong2& b)
{
    return make_longlong2(a.x - b.x, a.y - b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE longlong2 operator-(const longlong2& a, const long long b)
{
    return make_longlong2(a.x - b, a.y - b);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator-=(longlong2& a, const longlong2& b)
{
    a.x -= b.x; a.y -= b.y;
}
/** @} */

/** multiply
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong2 operator*(const longlong2& a, const longlong2& b)
{
    return make_longlong2(a.x * b.x, a.y * b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE longlong2 operator*(const longlong2& a, const long long s)
{
    return make_longlong2(a.x * s, a.y * s);
}
OPTIXU_INLINE RT_HOSTDEVICE longlong2 operator*(const long long s, const longlong2& a)
{
    return make_longlong2(a.x * s, a.y * s);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(longlong2& a, const long long s)
{
    a.x *= s; a.y *= s;
}
/** @} */

/** clamp
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong2 clamp(const longlong2& v, const long long a, const long long b)
{
    return make_longlong2(clamp(v.x, a, b), clamp(v.y, a, b));
}

OPTIXU_INLINE RT_HOSTDEVICE longlong2 clamp(const longlong2& v, const longlong2& a, const longlong2& b)
{
    return make_longlong2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE bool operator==(const longlong2& a, const longlong2& b)
{
    return a.x == b.x && a.y == b.y;
}

OPTIXU_INLINE RT_HOSTDEVICE bool operator!=(const longlong2& a, const longlong2& b)
{
    return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE long long getByIndex(const longlong2& v, int i)
{
    return ((long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(longlong2& v, int i, long long x)
{
    ((long long*)(&v))[i] = x;
}


/* longlong3 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong3 make_longlong3(const long long s)
{
    return make_longlong3(s, s, s);
}
OPTIXU_INLINE RT_HOSTDEVICE longlong3 make_longlong3(const float3& a)
{
    return make_longlong3( (long long)a.x, (long long)a.y, (long long)a.z);
}
/** @} */

/** negate */
OPTIXU_INLINE RT_HOSTDEVICE longlong3 operator-(const longlong3& a)
{
    return make_longlong3(-a.x, -a.y, -a.z);
}

/** min */
OPTIXU_INLINE RT_HOSTDEVICE longlong3 min(const longlong3& a, const longlong3& b)
{
    return make_longlong3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

/** max */
OPTIXU_INLINE RT_HOSTDEVICE longlong3 max(const longlong3& a, const longlong3& b)
{
    return make_longlong3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

/** add
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong3 operator+(const longlong3& a, const longlong3& b)
{
    return make_longlong3(a.x + b.x, a.y + b.y, a.z + b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator+=(longlong3& a, const longlong3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong3 operator-(const longlong3& a, const longlong3& b)
{
    return make_longlong3(a.x - b.x, a.y - b.y, a.z - b.z);
}

OPTIXU_INLINE RT_HOSTDEVICE void operator-=(longlong3& a, const longlong3& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong3 operator*(const longlong3& a, const longlong3& b)
{
    return make_longlong3(a.x * b.x, a.y * b.y, a.z * b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE longlong3 operator*(const longlong3& a, const long long s)
{
    return make_longlong3(a.x * s, a.y * s, a.z * s);
}
OPTIXU_INLINE RT_HOSTDEVICE longlong3 operator*(const long long s, const longlong3& a)
{
    return make_longlong3(a.x * s, a.y * s, a.z * s);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(longlong3& a, const long long s)
{
    a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong3 operator/(const longlong3& a, const longlong3& b)
{
    return make_longlong3(a.x / b.x, a.y / b.y, a.z / b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE longlong3 operator/(const longlong3& a, const long long s)
{
    return make_longlong3(a.x / s, a.y / s, a.z / s);
}
OPTIXU_INLINE RT_HOSTDEVICE longlong3 operator/(const long long s, const longlong3& a)
{
    return make_longlong3(s /a.x, s / a.y, s / a.z);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator/=(longlong3& a, const long long s)
{
    a.x /= s; a.y /= s; a.z /= s;
}
/** @} */

/** clamp
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong3 clamp(const longlong3& v, const long long a, const long long b)
{
    return make_longlong3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

OPTIXU_INLINE RT_HOSTDEVICE longlong3 clamp(const longlong3& v, const longlong3& a, const longlong3& b)
{
    return make_longlong3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** equality
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE bool operator==(const longlong3& a, const longlong3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

OPTIXU_INLINE RT_HOSTDEVICE bool operator!=(const longlong3& a, const longlong3& b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE long long getByIndex(const longlong3& v, int i)
{
    return ((long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(longlong3& v, int i, int x)
{
    ((long long*)(&v))[i] = x;
}


/* longlong4 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong4 make_longlong4(const long long s)
{
    return make_longlong4(s, s, s, s);
}
OPTIXU_INLINE RT_HOSTDEVICE longlong4 make_longlong4(const float4& a)
{
    return make_longlong4((long long)a.x, (long long)a.y, (long long)a.z, (long long)a.w);
}
/** @} */

/** negate */
OPTIXU_INLINE RT_HOSTDEVICE longlong4 operator-(const longlong4& a)
{
    return make_longlong4(-a.x, -a.y, -a.z, -a.w);
}

/** min */
OPTIXU_INLINE RT_HOSTDEVICE longlong4 min(const longlong4& a, const longlong4& b)
{
    return make_longlong4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

/** max */
OPTIXU_INLINE RT_HOSTDEVICE longlong4 max(const longlong4& a, const longlong4& b)
{
    return make_longlong4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

/** add
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong4 operator+(const longlong4& a, const longlong4& b)
{
    return make_longlong4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator+=(longlong4& a, const longlong4& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
/** @} */

/** subtract
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong4 operator-(const longlong4& a, const longlong4& b)
{
    return make_longlong4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

OPTIXU_INLINE RT_HOSTDEVICE void operator-=(longlong4& a, const longlong4& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong4 operator*(const longlong4& a, const longlong4& b)
{
    return make_longlong4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE longlong4 operator*(const longlong4& a, const long long s)
{
    return make_longlong4(a.x * s, a.y * s, a.z * s, a.w * s);
}
OPTIXU_INLINE RT_HOSTDEVICE longlong4 operator*(const long long s, const longlong4& a)
{
    return make_longlong4(a.x * s, a.y * s, a.z * s, a.w * s);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(longlong4& a, const long long s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
/** @} */

/** divide
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong4 operator/(const longlong4& a, const longlong4& b)
{
    return make_longlong4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE longlong4 operator/(const longlong4& a, const long long s)
{
    return make_longlong4(a.x / s, a.y / s, a.z / s, a.w / s);
}
OPTIXU_INLINE RT_HOSTDEVICE longlong4 operator/(const long long s, const longlong4& a)
{
    return make_longlong4(s / a.x, s / a.y, s / a.z, s / a.w);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator/=(longlong4& a, const long long s)
{
    a.x /= s; a.y /= s; a.z /= s; a.w /= s;
}
/** @} */

/** clamp
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE longlong4 clamp(const longlong4& v, const long long a, const long long b)
{
    return make_longlong4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

OPTIXU_INLINE RT_HOSTDEVICE longlong4 clamp(const longlong4& v, const longlong4& a, const longlong4& b)
{
    return make_longlong4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE bool operator==(const longlong4& a, const longlong4& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

OPTIXU_INLINE RT_HOSTDEVICE bool operator!=(const longlong4& a, const longlong4& b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE long long getByIndex(const longlong4& v, int i)
{
    return ((long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(longlong4& v, int i, long long x)
{
    ((long long*)(&v))[i] = x;
}

/* ulonglong functions */
/******************************************************************************/

/** clamp */
OPTIXU_INLINE RT_HOSTDEVICE unsigned long long clamp(const unsigned long long f, const unsigned long long a, const unsigned long long b)
{
    return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE unsigned long long getByIndex(const ulonglong1& v, unsigned int i)
{
    return ((unsigned long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(ulonglong1& v, int i, unsigned long long x)
{
    ((unsigned long long*)(&v))[i] = x;
}


/* ulonglong2 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong2 make_ulonglong2(const unsigned long long s)
{
    return make_ulonglong2(s, s);
}
OPTIXU_INLINE RT_HOSTDEVICE ulonglong2 make_ulonglong2(const float2& a)
{
    return make_ulonglong2((unsigned long long)a.x, (unsigned long long)a.y);
}
/** @} */

/** min */
OPTIXU_INLINE RT_HOSTDEVICE ulonglong2 min(const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(min(a.x, b.x), min(a.y, b.y));
}

/** max */
OPTIXU_INLINE RT_HOSTDEVICE ulonglong2 max(const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(max(a.x, b.x), max(a.y, b.y));
}

/** add
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong2 operator+(const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(a.x + b.x, a.y + b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator+=(ulonglong2& a, const ulonglong2& b)
{
    a.x += b.x; a.y += b.y;
}
/** @} */

/** subtract
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong2 operator-(const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(a.x - b.x, a.y - b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE ulonglong2 operator-(const ulonglong2& a, const unsigned long long b)
{
    return make_ulonglong2(a.x - b, a.y - b);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator-=(ulonglong2& a, const ulonglong2& b)
{
    a.x -= b.x; a.y -= b.y;
}
/** @} */

/** multiply
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong2 operator*(const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(a.x * b.x, a.y * b.y);
}
OPTIXU_INLINE RT_HOSTDEVICE ulonglong2 operator*(const ulonglong2& a, const unsigned long long s)
{
    return make_ulonglong2(a.x * s, a.y * s);
}
OPTIXU_INLINE RT_HOSTDEVICE ulonglong2 operator*(const unsigned long long s, const ulonglong2& a)
{
    return make_ulonglong2(a.x * s, a.y * s);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(ulonglong2& a, const unsigned long long s)
{
    a.x *= s; a.y *= s;
}
/** @} */

/** clamp
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong2 clamp(const ulonglong2& v, const unsigned long long a, const unsigned long long b)
{
    return make_ulonglong2(clamp(v.x, a, b), clamp(v.y, a, b));
}

OPTIXU_INLINE RT_HOSTDEVICE ulonglong2 clamp(const ulonglong2& v, const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE bool operator==(const ulonglong2& a, const ulonglong2& b)
{
    return a.x == b.x && a.y == b.y;
}

OPTIXU_INLINE RT_HOSTDEVICE bool operator!=(const ulonglong2& a, const ulonglong2& b)
{
    return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE unsigned long long getByIndex(const ulonglong2& v, unsigned int i)
{
    return ((unsigned long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(ulonglong2& v, int i, unsigned long long x)
{
    ((unsigned long long*)(&v))[i] = x;
}


/* ulonglong3 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 make_ulonglong3(const unsigned long long s)
{
    return make_ulonglong3(s, s, s);
}
OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 make_ulonglong3(const float3& a)
{
    return make_ulonglong3((unsigned long long)a.x, (unsigned long long)a.y, (unsigned long long)a.z);
}
/** @} */

/** min */
OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 min(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

/** max */
OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 max(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

/** add
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 operator+(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(a.x + b.x, a.y + b.y, a.z + b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator+=(ulonglong3& a, const ulonglong3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 operator-(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(a.x - b.x, a.y - b.y, a.z - b.z);
}

OPTIXU_INLINE RT_HOSTDEVICE void operator-=(ulonglong3& a, const ulonglong3& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 operator*(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(a.x * b.x, a.y * b.y, a.z * b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 operator*(const ulonglong3& a, const unsigned long long s)
{
    return make_ulonglong3(a.x * s, a.y * s, a.z * s);
}
OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 operator*(const unsigned long long s, const ulonglong3& a)
{
    return make_ulonglong3(a.x * s, a.y * s, a.z * s);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(ulonglong3& a, const unsigned long long s)
{
    a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 operator/(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(a.x / b.x, a.y / b.y, a.z / b.z);
}
OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 operator/(const ulonglong3& a, const unsigned long long s)
{
    return make_ulonglong3(a.x / s, a.y / s, a.z / s);
}
OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 operator/(const unsigned long long s, const ulonglong3& a)
{
    return make_ulonglong3(s / a.x, s / a.y, s / a.z);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator/=(ulonglong3& a, const unsigned long long s)
{
    a.x /= s; a.y /= s; a.z /= s;
}
/** @} */

/** clamp
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 clamp(const ulonglong3& v, const unsigned long long a, const unsigned long long b)
{
    return make_ulonglong3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 clamp(const ulonglong3& v, const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** equality
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE bool operator==(const ulonglong3& a, const ulonglong3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

OPTIXU_INLINE RT_HOSTDEVICE bool operator!=(const ulonglong3& a, const ulonglong3& b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory
*/
OPTIXU_INLINE RT_HOSTDEVICE unsigned long long getByIndex(const ulonglong3& v, unsigned int i)
{
    return ((unsigned long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory
*/
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(ulonglong3& v, int i, unsigned long long x)
{
    ((unsigned long long*)(&v))[i] = x;
}


/* ulonglong4 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 make_ulonglong4(const unsigned long long s)
{
    return make_ulonglong4(s, s, s, s);
}
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 make_ulonglong4(const float4& a)
{
    return make_ulonglong4((unsigned long long)a.x, (unsigned long long)a.y, (unsigned long long)a.z, (unsigned long long)a.w);
}
/** @} */

/** min
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 min(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}
/** @} */

/** max
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 max(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}
/** @} */

/** add
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 operator+(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator+=(ulonglong4& a, const ulonglong4& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
/** @} */

/** subtract
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 operator-(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

OPTIXU_INLINE RT_HOSTDEVICE void operator-=(ulonglong4& a, const ulonglong4& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 operator*(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 operator*(const ulonglong4& a, const unsigned long long s)
{
    return make_ulonglong4(a.x * s, a.y * s, a.z * s, a.w * s);
}
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 operator*(const unsigned long long s, const ulonglong4& a)
{
    return make_ulonglong4(a.x * s, a.y * s, a.z * s, a.w * s);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator*=(ulonglong4& a, const unsigned long long s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
/** @} */

/** divide
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 operator/(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 operator/(const ulonglong4& a, const unsigned long long s)
{
    return make_ulonglong4(a.x / s, a.y / s, a.z / s, a.w / s);
}
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 operator/(const unsigned long long s, const ulonglong4& a)
{
    return make_ulonglong4(s / a.x, s / a.y, s / a.z, s / a.w);
}
OPTIXU_INLINE RT_HOSTDEVICE void operator/=(ulonglong4& a, const unsigned long long s)
{
    a.x /= s; a.y /= s; a.z /= s; a.w /= s;
}
/** @} */

/** clamp
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 clamp(const ulonglong4& v, const unsigned long long a, const unsigned long long b)
{
    return make_ulonglong4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 clamp(const ulonglong4& v, const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE bool operator==(const ulonglong4& a, const ulonglong4& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

OPTIXU_INLINE RT_HOSTDEVICE bool operator!=(const ulonglong4& a, const ulonglong4& b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory
*/
OPTIXU_INLINE RT_HOSTDEVICE unsigned long long getByIndex(const ulonglong4& v, unsigned int i)
{
    return ((unsigned long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory
*/
OPTIXU_INLINE RT_HOSTDEVICE void setByIndex(ulonglong4& v, int i, unsigned long long x)
{
    ((unsigned long long*)(&v))[i] = x;
}


/******************************************************************************/

/** Narrowing functions
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int2 make_int2(const int3& v0) { return make_int2( v0.x, v0.y ); }
OPTIXU_INLINE RT_HOSTDEVICE int2 make_int2(const int4& v0) { return make_int2( v0.x, v0.y ); }
OPTIXU_INLINE RT_HOSTDEVICE int3 make_int3(const int4& v0) { return make_int3( v0.x, v0.y, v0.z ); }
OPTIXU_INLINE RT_HOSTDEVICE uint2 make_uint2(const uint3& v0) { return make_uint2( v0.x, v0.y ); }
OPTIXU_INLINE RT_HOSTDEVICE uint2 make_uint2(const uint4& v0) { return make_uint2( v0.x, v0.y ); }
OPTIXU_INLINE RT_HOSTDEVICE uint3 make_uint3(const uint4& v0) { return make_uint3( v0.x, v0.y, v0.z ); }
OPTIXU_INLINE RT_HOSTDEVICE longlong2 make_longlong2(const longlong3& v0) { return make_longlong2( v0.x, v0.y ); }
OPTIXU_INLINE RT_HOSTDEVICE longlong2 make_longlong2(const longlong4& v0) { return make_longlong2( v0.x, v0.y ); }
OPTIXU_INLINE RT_HOSTDEVICE longlong3 make_longlong3(const longlong4& v0) { return make_longlong3( v0.x, v0.y, v0.z ); }
OPTIXU_INLINE RT_HOSTDEVICE ulonglong2 make_ulonglong2(const ulonglong3& v0) { return make_ulonglong2( v0.x, v0.y ); }
OPTIXU_INLINE RT_HOSTDEVICE ulonglong2 make_ulonglong2(const ulonglong4& v0) { return make_ulonglong2( v0.x, v0.y ); }
OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 make_ulonglong3(const ulonglong4& v0) { return make_ulonglong3( v0.x, v0.y, v0.z ); }
OPTIXU_INLINE RT_HOSTDEVICE float2 make_float2(const float3& v0) { return make_float2( v0.x, v0.y ); }
OPTIXU_INLINE RT_HOSTDEVICE float2 make_float2(const float4& v0) { return make_float2( v0.x, v0.y ); }
OPTIXU_INLINE RT_HOSTDEVICE float3 make_float3(const float4& v0) { return make_float3( v0.x, v0.y, v0.z ); }
/** @} */

/** Assemble functions from smaller vectors 
* @{
*/
OPTIXU_INLINE RT_HOSTDEVICE int3 make_int3(const int v0, const int2& v1) { return make_int3( v0, v1.x, v1.y ); }
OPTIXU_INLINE RT_HOSTDEVICE int3 make_int3(const int2& v0, const int v1) { return make_int3( v0.x, v0.y, v1 ); }
OPTIXU_INLINE RT_HOSTDEVICE int4 make_int4(const int v0, const int v1, const int2& v2) { return make_int4( v0, v1, v2.x, v2.y ); }
OPTIXU_INLINE RT_HOSTDEVICE int4 make_int4(const int v0, const int2& v1, const int v2) { return make_int4( v0, v1.x, v1.y, v2 ); }
OPTIXU_INLINE RT_HOSTDEVICE int4 make_int4(const int2& v0, const int v1, const int v2) { return make_int4( v0.x, v0.y, v1, v2 ); }
OPTIXU_INLINE RT_HOSTDEVICE int4 make_int4(const int v0, const int3& v1) { return make_int4( v0, v1.x, v1.y, v1.z ); }
OPTIXU_INLINE RT_HOSTDEVICE int4 make_int4(const int3& v0, const int v1) { return make_int4( v0.x, v0.y, v0.z, v1 ); }
OPTIXU_INLINE RT_HOSTDEVICE int4 make_int4(const int2& v0, const int2& v1) { return make_int4( v0.x, v0.y, v1.x, v1.y ); }
OPTIXU_INLINE RT_HOSTDEVICE uint3 make_uint3(const unsigned int v0, const uint2& v1) { return make_uint3( v0, v1.x, v1.y ); }
OPTIXU_INLINE RT_HOSTDEVICE uint3 make_uint3(const uint2& v0, const unsigned int v1) { return make_uint3( v0.x, v0.y, v1 ); }
OPTIXU_INLINE RT_HOSTDEVICE uint4 make_uint4(const unsigned int v0, const unsigned int v1, const uint2& v2) { return make_uint4( v0, v1, v2.x, v2.y ); }
OPTIXU_INLINE RT_HOSTDEVICE uint4 make_uint4(const unsigned int v0, const uint2& v1, const unsigned int v2) { return make_uint4( v0, v1.x, v1.y, v2 ); }
OPTIXU_INLINE RT_HOSTDEVICE uint4 make_uint4(const uint2& v0, const unsigned int v1, const unsigned int v2) { return make_uint4( v0.x, v0.y, v1, v2 ); }
OPTIXU_INLINE RT_HOSTDEVICE uint4 make_uint4(const unsigned int v0, const uint3& v1) { return make_uint4( v0, v1.x, v1.y, v1.z ); }
OPTIXU_INLINE RT_HOSTDEVICE uint4 make_uint4(const uint3& v0, const unsigned int v1) { return make_uint4( v0.x, v0.y, v0.z, v1 ); }
OPTIXU_INLINE RT_HOSTDEVICE uint4 make_uint4(const uint2& v0, const uint2& v1) { return make_uint4( v0.x, v0.y, v1.x, v1.y ); }
OPTIXU_INLINE RT_HOSTDEVICE longlong3 make_longlong3(const long long v0, const longlong2& v1) { return make_longlong3(v0, v1.x, v1.y); }
OPTIXU_INLINE RT_HOSTDEVICE longlong3 make_longlong3(const longlong2& v0, const long long v1) { return make_longlong3(v0.x, v0.y, v1); }
OPTIXU_INLINE RT_HOSTDEVICE longlong4 make_longlong4(const long long v0, const long long v1, const longlong2& v2) { return make_longlong4(v0, v1, v2.x, v2.y); }
OPTIXU_INLINE RT_HOSTDEVICE longlong4 make_longlong4(const long long v0, const longlong2& v1, const long long v2) { return make_longlong4(v0, v1.x, v1.y, v2); }
OPTIXU_INLINE RT_HOSTDEVICE longlong4 make_longlong4(const longlong2& v0, const long long v1, const long long v2) { return make_longlong4(v0.x, v0.y, v1, v2); }
OPTIXU_INLINE RT_HOSTDEVICE longlong4 make_longlong4(const long long v0, const longlong3& v1) { return make_longlong4(v0, v1.x, v1.y, v1.z); }
OPTIXU_INLINE RT_HOSTDEVICE longlong4 make_longlong4(const longlong3& v0, const long long v1) { return make_longlong4(v0.x, v0.y, v0.z, v1); }
OPTIXU_INLINE RT_HOSTDEVICE longlong4 make_longlong4(const longlong2& v0, const longlong2& v1) { return make_longlong4(v0.x, v0.y, v1.x, v1.y); }
OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 make_ulonglong3(const unsigned long long v0, const ulonglong2& v1) { return make_ulonglong3(v0, v1.x, v1.y); }
OPTIXU_INLINE RT_HOSTDEVICE ulonglong3 make_ulonglong3(const ulonglong2& v0, const unsigned long long v1) { return make_ulonglong3(v0.x, v0.y, v1); }
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 make_ulonglong4(const unsigned long long v0, const unsigned long long v1, const ulonglong2& v2) { return make_ulonglong4(v0, v1, v2.x, v2.y); }
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 make_ulonglong4(const unsigned long long v0, const ulonglong2& v1, const unsigned long long v2) { return make_ulonglong4(v0, v1.x, v1.y, v2); }
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 make_ulonglong4(const ulonglong2& v0, const unsigned long long v1, const unsigned long long v2) { return make_ulonglong4(v0.x, v0.y, v1, v2); }
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 make_ulonglong4(const unsigned long long v0, const ulonglong3& v1) { return make_ulonglong4(v0, v1.x, v1.y, v1.z); }
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 make_ulonglong4(const ulonglong3& v0, const unsigned long long v1) { return make_ulonglong4(v0.x, v0.y, v0.z, v1); }
OPTIXU_INLINE RT_HOSTDEVICE ulonglong4 make_ulonglong4(const ulonglong2& v0, const ulonglong2& v1) { return make_ulonglong4(v0.x, v0.y, v1.x, v1.y); }
OPTIXU_INLINE RT_HOSTDEVICE float3 make_float3(const float2& v0, const float v1) { return make_float3(v0.x, v0.y, v1); }
OPTIXU_INLINE RT_HOSTDEVICE float3 make_float3(const float v0, const float2& v1) { return make_float3( v0, v1.x, v1.y ); }
OPTIXU_INLINE RT_HOSTDEVICE float4 make_float4(const float v0, const float v1, const float2& v2) { return make_float4( v0, v1, v2.x, v2.y ); }
OPTIXU_INLINE RT_HOSTDEVICE float4 make_float4(const float v0, const float2& v1, const float v2) { return make_float4( v0, v1.x, v1.y, v2 ); }
OPTIXU_INLINE RT_HOSTDEVICE float4 make_float4(const float2& v0, const float v1, const float v2) { return make_float4( v0.x, v0.y, v1, v2 ); }
OPTIXU_INLINE RT_HOSTDEVICE float4 make_float4(const float v0, const float3& v1) { return make_float4( v0, v1.x, v1.y, v1.z ); }
OPTIXU_INLINE RT_HOSTDEVICE float4 make_float4(const float3& v0, const float v1) { return make_float4( v0.x, v0.y, v0.z, v1 ); }
OPTIXU_INLINE RT_HOSTDEVICE float4 make_float4(const float2& v0, const float2& v1) { return make_float4( v0.x, v0.y, v1.x, v1.y ); }
/** @} */


/* Common helper functions */
/******************************************************************************/

/** Return a smooth value in [0,1], where the transition from 0
*   to 1 takes place for values of x in [edge0,edge1]. 
*/
OPTIXU_INLINE RT_HOSTDEVICE float smoothstep(const float edge0, const float edge1, const float x)
{
  /** assert( edge1 > edge0 ); */
  const float t = clamp( (x-edge0) / (edge1-edge0), 0.0f, 1.0f );
  return t*t * ( 3.0f - 2.0f*t );
}

/** Simple mapping from [0,1] to a temperature-like RGB color. 
*/
OPTIXU_INLINE RT_HOSTDEVICE float3 temperature(const float t)
{
  const float b = t < 0.25f ? smoothstep( -0.25f, 0.25f, t ) : 1.0f-smoothstep( 0.25f, 0.5f, t );
  const float g = t < 0.5f  ? smoothstep( 0.0f, 0.5f, t ) : (t < 0.75f ? 1.0f : 1.0f-smoothstep( 0.75f, 1.0f, t ));
  const float r = smoothstep( 0.5f, 0.75f, t );
  return make_float3( r, g, b );
}

/** Branchless intersection avoids divergence.
*/
OPTIXU_INLINE RT_HOSTDEVICE bool intersect_triangle_branchless(const Ray&    ray,
                                                               const float3& p0,
                                                               const float3& p1,
                                                               const float3& p2,
                                                                     float3& n,
                                                                     float&  t,
                                                                     float&  beta,
                                                                     float&  gamma)
{
  const float3 e0 = p1 - p0;
  const float3 e1 = p0 - p2;
  n  = cross( e1, e0 );

  const float3 e2 = ( 1.0f / dot( n, ray.direction ) ) * ( p0 - ray.origin );
  const float3 i  = cross( ray.direction, e2 );

  beta  = dot( i, e1 );
  gamma = dot( i, e0 );
  t     = dot( n, e2 );

  return ( (t<ray.tmax) & (t>ray.tmin) & (beta>=0.0f) & (gamma>=0.0f) & (beta+gamma<=1) );
}

/** Intersection with early exit.
*/
OPTIXU_INLINE RT_HOSTDEVICE bool intersect_triangle_earlyexit(const Ray&    ray,
                                                              const float3& p0,
                                                              const float3& p1,
                                                              const float3& p2,
                                                                    float3& n,
                                                                    float&  t,
                                                                    float&  beta,
                                                                    float&  gamma)
{
  float3 e0 = p1 - p0;
  float3 e1 = p0 - p2;
  n  = cross( e0, e1 );

  float v   = dot( n, ray.direction );
  float r   = 1.0f / v;

  float3 e2 = p0 - ray.origin;
  float va  = dot( n, e2 );
  t         = r*va;

  // Initialize these to reduce their liveness when we leave the function without
  // computing their value.
  beta = 0;
  gamma = 0;
  
  if(t < ray.tmax && t > ray.tmin) {
    float3 i   = cross( e2, ray.direction );
    float v1   = dot( i, e1 );
    beta = r*v1;
    if(beta >= 0.0f){
      float v2 = dot( i, e0 );
      gamma = r*v2;
      n = -n;
      return ( (v1+v2)*v <= v*v && gamma >= 0.0f );
    }
  }
  return false;
}

/** Intersect ray with CCW wound triangle.  Returns non-normalize normal vector. */ 
OPTIXU_INLINE RT_HOSTDEVICE bool intersect_triangle(const Ray&    ray,
                                                    const float3& p0,
                                                    const float3& p1,
                                                    const float3& p2,
                                                          float3& n,
                                                          float&  t,
                                                          float&  beta,
                                                          float&  gamma)
{
  return intersect_triangle_branchless(ray, p0, p1, p2, n, t, beta, gamma);  
}


/**
*  Calculates refraction direction
*  r   : refraction vector
*  i   : incident vector
*  n   : surface normal
*  ior : index of refraction ( n2 / n1 )
*  returns false in case of total internal reflection, in that case r is
*          initialized to (0,0,0).
*/
OPTIXU_INLINE RT_HOSTDEVICE bool refract(float3& r, const float3& i, const float3& n, const float ior)
{
  float3 nn = n;
  float negNdotV = dot(i,nn);
  float eta;

  if (negNdotV > 0.0f)
  {
    eta = ior;
    nn = -n;
    negNdotV = -negNdotV;
  }
  else
  {
    eta = 1.f / ior;
  }

  const float k = 1.f - eta*eta * (1.f - negNdotV * negNdotV);

  if (k < 0.0f) {
    // Initialize this value, so that r always leaves this function initialized.
    r = make_float3(0.f);
    return false;
  } else {
    r = normalize(eta*i - (eta*negNdotV + sqrtf(k)) * nn);
    return true;
  }
}

/** Schlick approximation of Fresnel reflectance
*/
OPTIXU_INLINE RT_HOSTDEVICE float fresnel_schlick(const float cos_theta, const float exponent = 5.0f,
                                                  const float minimum = 0.0f, const float maximum = 1.0f)
{
  /**
    Clamp the result of the arithmetic due to floating point precision:
    the result should lie strictly within [minimum, maximum]
    return clamp(minimum + (maximum - minimum) * powf(1.0f - cos_theta, exponent),
                 minimum, maximum);
  */

  /** The max doesn't seem like it should be necessary, but without it you get
      annoying broken pixels at the center of reflective spheres where cos_theta ~ 1.
  */
  return clamp(minimum + (maximum - minimum) * powf(fmaxf(0.0f,1.0f - cos_theta), exponent),
               minimum, maximum);
}

OPTIXU_INLINE RT_HOSTDEVICE float3 fresnel_schlick(const float cos_theta, const float exponent,
                                                   const float3& minimum, const float3& maximum)
{
  return make_float3(fresnel_schlick(cos_theta, exponent, minimum.x, maximum.x),
                     fresnel_schlick(cos_theta, exponent, minimum.y, maximum.y),
                     fresnel_schlick(cos_theta, exponent, minimum.z, maximum.z));
}


/** Calculate the NTSC luminance value of an rgb triple
*/
OPTIXU_INLINE RT_HOSTDEVICE float luminance(const float3& rgb)
{
  const float3 ntsc_luminance = { 0.30f, 0.59f, 0.11f };
  return  dot( rgb, ntsc_luminance );
}

/** Calculate the CIE luminance value of an rgb triple
*/
OPTIXU_INLINE RT_HOSTDEVICE float luminanceCIE(const float3& rgb)
{
  const float3 cie_luminance = { 0.2126f, 0.7152f, 0.0722f };
  return  dot( rgb, cie_luminance );
}

OPTIXU_INLINE RT_HOSTDEVICE void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
  // Uniformly sample disk.
  const float r   = sqrtf( u1 );
  const float phi = 2.0f*M_PIf * u2;
  p.x = r * cosf( phi );
  p.y = r * sinf( phi );

  // Project up to hemisphere.
  p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) );
}

/** Maps concentric squares to concentric circles (Shirley and Chiu)
*/
OPTIXU_INLINE RT_HOSTDEVICE float2 square_to_disk(const float2& sample)
{
  float phi, r;

  const float a = 2.0f * sample.x - 1.0f;
  const float b = 2.0f * sample.y - 1.0f;

  if (a > -b)
  {
    if (a > b)
    {
      r = a;
      phi = (float)M_PI_4f * (b/a);
    }
    else
    {
      r = b;
      phi = (float)M_PI_4f * (2.0f - (a/b));
    }
  }
  else
  {
    if (a < b)
    {
      r = -a;
      phi = (float)M_PI_4f * (4.0f + (b/a));
    }
    else
    {
      r = -b;
      phi = (b) ? (float)M_PI_4f * (6.0f - (a/b)) : 0.0f;
    }
  }

  return make_float2( r * cosf(phi), r * sinf(phi) );
}

/**
* Convert Cartesian coordinates to polar coordinates
*/
OPTIXU_INLINE RT_HOSTDEVICE float3 cart_to_pol(const float3& v)
{
  float azimuth;
  float elevation;
  float radius = length(v);

  float r = sqrtf(v.x*v.x + v.y*v.y);
  if (r > 0.0f)
  {
    azimuth   = atanf(v.y / v.x);
    elevation = atanf(v.z / r);

    if (v.x < 0.0f)
      azimuth += M_PIf;
    else if (v.y < 0.0f)
      azimuth += M_PIf * 2.0f;
  }
  else
  {
    azimuth = 0.0f;

    if (v.z > 0.0f)
      elevation = +M_PI_2f;
    else
      elevation = -M_PI_2f;
  }

  return make_float3(azimuth, elevation, radius);
}

/**
* Orthonormal basis
*/
struct Onb
{
  OPTIXU_INLINE RT_HOSTDEVICE Onb(const float3& normal)
  {
    m_normal = normal;

    if( fabs(m_normal.x) > fabs(m_normal.z) )
    {
      m_binormal.x = -m_normal.y;
      m_binormal.y =  m_normal.x;
      m_binormal.z =  0;
    }
    else
    {
      m_binormal.x =  0;
      m_binormal.y = -m_normal.z;
      m_binormal.z =  m_normal.y;
    }

    m_binormal = normalize(m_binormal);
    m_tangent = cross( m_binormal, m_normal );
  }

  OPTIXU_INLINE RT_HOSTDEVICE void inverse_transform(float3& p) const
  {
    p = p.x*m_tangent + p.y*m_binormal + p.z*m_normal;
  }

  float3 m_tangent;
  float3 m_binormal;
  float3 m_normal;
};

} // end namespace optix

/*
 * When looking for operators for a type, only the scope the type is defined in (plus the
 * global scope) is searched.  In order to make the operators behave properly we are
 * pulling them into the global namespace.
 */
#if defined(RT_PULL_IN_VECTOR_TYPES)
using optix::operator-;
using optix::operator-=;
using optix::operator+;
using optix::operator+=;
using optix::operator*;
using optix::operator*=;
using optix::operator/;
using optix::operator/=;
using optix::operator==;
#endif // #if defined(RT_PULL_IN_VECTOR_TYPES)

#if !defined(RT_NO_GLOBAL_NAMESPACE_INJECTION)
/*
 * Here are a list of functions that are overloaded in both the global and optix
 * namespace.  If they have a global namespace version, then the overloads in the optix
 * namespace need to be pulled in, so that all the overloads are on the same level.
 */

/* These are defined by CUDA in the global namespace */
#if defined(RT_PULL_IN_VECTOR_FUNCTIONS)
#define RT_DEFINE_HELPER(type) \
  using optix::make_##type##1; \
  using optix::make_##type##2; \
  using optix::make_##type##3; \
  using optix::make_##type##4;

RT_DEFINE_HELPER(char)
RT_DEFINE_HELPER(uchar)
RT_DEFINE_HELPER(short)
RT_DEFINE_HELPER(ushort)
RT_DEFINE_HELPER(int)
RT_DEFINE_HELPER(uint)
RT_DEFINE_HELPER(long)
RT_DEFINE_HELPER(ulong)
RT_DEFINE_HELPER(float)
RT_DEFINE_HELPER(longlong)
RT_DEFINE_HELPER(ulonglong)
RT_DEFINE_HELPER(double)

#undef RT_DEFINE_HELPER

#endif // #if defined(RT_PULL_IN_VECTOR_FUNCTIONS)

/* These are defined by CUDA and non-Windows platforms in the global namespace. */
#if !defined(_WIN32) || defined (__CUDACC__)
using optix::fmaxf;
using optix::fminf;
using optix::copysignf;
#endif

/* These are always in the global namespace. */
using optix::expf;
using optix::floor;
#endif // #if !defined(RT_NO_GLOBAL_NAMESPACE_INJECTION)

#ifdef OPTIXU_INLINE_DEFINED
#  undef OPTIXU_INLINE_DEFINED
#  undef OPTIXU_INLINE
#endif

#endif // #ifndef __optixu_optixu_math_namespace_h__
