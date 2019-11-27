
/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */
 
 /**
 * @file   optixu_math_stream_namespace.h
 * @author NVIDIA Corporation
 * @brief  OptiX public API
 *
 * Stream operators for CUDA vector types
 */

#ifndef __optixu_optixu_math_stream_namespace_h__
#define __optixu_optixu_math_stream_namespace_h__

#include "optixu_math_namespace.h"
#include "optixu_matrix_namespace.h"
#include "optixu_aabb_namespace.h"

#include <iostream>

namespace optix {

  /**
  * Provide access to stream functionalities with CUDA float vector types
  * @{
  */
  inline std::ostream& operator<<(std::ostream& os, const optix::float4& v) { os << '[' << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ']'; return os; }
  inline std::istream& operator>>(std::istream& is, optix::float4& v) { char st; is >> st >> v.x >> st >> v.y >> st >> v.z >> st >> v.w >> st; return is; }
  inline std::ostream& operator<<(std::ostream& os, const optix::float3& v) { os << '[' << v.x << ", " << v.y << ", " << v.z << ']'; return os; }
  inline std::istream& operator>>(std::istream& is, optix::float3& v) { char st; is >> st >> v.x >> st >> v.y >> st >> v.z >> st; return is; }
  inline std::ostream& operator<<(std::ostream& os, const optix::float2& v) { os << '[' << v.x << ", " << v.y << ']'; return os; }
  inline std::istream& operator>>(std::istream& is, optix::float2& v) { char st; is >> st >> v.x >> st >> v.y >> st; return is; }
  /** @} */

  /**
  * Provide access to stream functionalities with CUDA int vector types
  * @{
  */
  inline std::ostream& operator<<(std::ostream& os, const optix::int4& v) { os << '[' << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ']'; return os; }
  inline std::istream& operator>>(std::istream& is, optix::int4& v) { char st; is >> st >> v.x >> st >> v.y >> st >> v.z >> st >> v.w >> st; return is; }
  inline std::ostream& operator<<(std::ostream& os, const optix::int3& v) { os << '[' << v.x << ", " << v.y << ", " << v.z << ']'; return os; }
  inline std::istream& operator>>(std::istream& is, optix::int3& v) { char st; is >> st >> v.x >> st >> v.y >> st >> v.z >> st; return is; }
  inline std::ostream& operator<<(std::ostream& os, const optix::int2& v) { os << '[' << v.x << ", " << v.y << ']'; return os; }
  inline std::istream& operator>>(std::istream& is, optix::int2& v) { char st; is >> st >> v.x >> st >> v.y >> st; return is; }
  /** @} */

  /**
  * Provide access to stream functionalities with CUDA uint vector types
  * @{
  */
  inline std::ostream& operator<<(std::ostream& os, const optix::uint4& v) { os << '[' << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ']'; return os; }
  inline std::istream& operator>>(std::istream& is, optix::uint4& v) { char st; is >> st >> v.x >> st >> v.y >> st >> v.z >> st >> v.w >> st; return is; }
  inline std::ostream& operator<<(std::ostream& os, const optix::uint3& v) { os << '[' << v.x << ", " << v.y << ", " << v.z << ']'; return os; }
  inline std::istream& operator>>(std::istream& is, optix::uint3& v) { char st; is >> st >> v.x >> st >> v.y >> st >> v.z >> st; return is; }
  inline std::ostream& operator<<(std::ostream& os, const optix::uint2& v) { os << '[' << v.x << ", " << v.y << ']'; return os; }
  inline std::istream& operator>>(std::istream& is, optix::uint2& v) { char st; is >> st >> v.x >> st >> v.y >> st; return is; }
  /** @} */

  /**
  * Provide access to stream functionalities with OptiX axis-aligned bounding box type
  */
  inline std::ostream& operator<<( std::ostream& os, const optix::Aabb& aabb )
  { os << aabb[0] << " | " << aabb[1];  return os; }

  /**
  * Provide access to stream functionalities with OptiX matrix type
  * @{
  */
  template<unsigned int M, unsigned int N>
  inline std::ostream& operator<<( std::ostream& os, const optix::Matrix<M,N>& m )
  {
    os << '[';
    for (unsigned int i = 0; i < N; ++i) {
      os << '[';
      for (unsigned int j = 0; j < M; ++j) {
        os << m[i*N+j];
        if(j < M-1) os << ", ";
      }
      os << ']';
    }
    os << ']';
    return os;
  }

  template<unsigned int M, unsigned int N>
  inline std::istream& operator>>( std::istream& is, optix::Matrix<M,N>& m )
  {
    char st;
    is >> st;
    for (unsigned int i = 0; i < N; ++i) {
      is >> st;
      for (unsigned int j = 0; j < M; ++j) {
        is >> m[i*N+j];
        if(j < M-1) is >> st;
      }
      is >> st;
    }
    is >> st;
    return is;
  }
  /** @} */
}

/*
 * When looking for operators for a type, only the scope the type is defined in (plus the
 * global scope) is searched.  In order to make the operators behave properly we are
 * pulling them into the global namespace.
 */
#if defined(RT_PULL_IN_VECTOR_TYPES)
namespace {
  using optix::operator<<;
  using optix::operator>>;
}
#endif

#endif // #ifndef __optixu_optixu_math_stream_namespace_h__

