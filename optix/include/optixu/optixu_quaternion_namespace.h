
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
 * @file   optixu_quaternion_namespace.h
 * @author NVIDIA Corporation
 * @brief  OptiX public API
 *
 * OptiX public API Reference - Public QUATERNION namespace
 */
 
#ifndef __optixu_optixu_quaternion_namespace_h__
#define __optixu_optixu_quaternion_namespace_h__

#include "optixu_math_namespace.h"

// __forceinline__ works in cuda, VS, and with gcc.  Leave it as macro in case
// we need to make this per-platform or we want to switch off inlining globally.
#ifndef OPTIXU_INLINE 
#  define OPTIXU_INLINE_DEFINED 1
#  define OPTIXU_INLINE __forceinline__
#endif // OPTIXU_INLINE 

namespace optix {

 /**
  * @brief Quaternion
  * 
  * @ingroup CUDACTypes
  * 
  * <B>Description</B>
  * 
  * @ref Quaternion is a utility class for handling quaternions which 
  * are primarily useful for representing directions and rotations.
  *
  * <B>History</B>
  * 
  * @ref Quaternion was introduced in OptiX 5.0.
  */
  class Quaternion
  {
  public:

    /** Construct identity quaternion */
    RT_HOSTDEVICE Quaternion();

    /** Construct from coordinates x, y, z, w */
    RT_HOSTDEVICE Quaternion( float x, float y, float z, float w );

    /** Construct from float4 */
    RT_HOSTDEVICE Quaternion( float4 v );

    /** Copy constructor */
    RT_HOSTDEVICE Quaternion( const Quaternion& other );

    /** Construct from axis and angle (in degrees) */
    RT_HOSTDEVICE Quaternion( const float3&  axis, float angle );

    /** From quaternion to rotation matrix */
    RT_HOSTDEVICE void toMatrix( float m[16] ) const;

    /** quaternion x, y, z, w */
    float4 m_q;

  };

  OPTIXU_INLINE RT_HOSTDEVICE Quaternion::Quaternion()
  {
    m_q = make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
  }

  OPTIXU_INLINE RT_HOSTDEVICE Quaternion::Quaternion( float x, float y, float z, float w )
  {
    m_q = make_float4( x, y, z, w );
  }

  OPTIXU_INLINE RT_HOSTDEVICE Quaternion::Quaternion( float4 v )
  {
    m_q = v;
  }

  OPTIXU_INLINE RT_HOSTDEVICE Quaternion::Quaternion( const Quaternion& other )
  {
    m_q = other.m_q;
  }
  
  OPTIXU_INLINE RT_HOSTDEVICE Quaternion::Quaternion( const float3&  axis, float angle )
  {
    const float3 naxis = optix::normalize( axis );
    const float radian = angle * M_PIf / 180.0f;
    const float s = sinf(radian/2.0f);
    m_q.x = naxis.x * s;   
    m_q.y = naxis.y * s; 
    m_q.z = naxis.z * s; 
    m_q.w = cosf(radian/2.0f);
  }

  OPTIXU_INLINE RT_HOSTDEVICE void Quaternion::toMatrix( float m[16] ) const
  {
    m[0] = 1.0f - 2.0f*(m_q.y*m_q.y + m_q.z*m_q.z);
    m[1] = 2.0f*(m_q.x*m_q.y - m_q.z*m_q.w);
    m[2] = 2.0f*(m_q.x*m_q.z + m_q.y*m_q.w);
    m[3] = 0.0f;

    m[4] = 2.0f*(m_q.x*m_q.y + m_q.z*m_q.w);
    m[5] = 1.0f - 2.0f*(m_q.x*m_q.x + m_q.z*m_q.z);
    m[6] = 2.0f*(m_q.y*m_q.z - m_q.x*m_q.w);
    m[7] = 0.0f;

    m[8]  = 2.0f*(m_q.x*m_q.z - m_q.y*m_q.w);
    m[9]  = 2.0f*(m_q.y*m_q.z + m_q.x*m_q.w);
    m[10] = 1.0f - 2.0f*(m_q.x*m_q.x + m_q.y*m_q.y);
    m[11] = 0.0f;

    m[12] = 0.0f;
    m[13] = 0.0f;
    m[14] = 0.0f;
    m[15] = 1.0f;
  }

  OPTIXU_INLINE RT_HOSTDEVICE float3 operator*( const Quaternion& quat, const float3& v )
  {
  const float x =
    (1.0f - 2.0f*(quat.m_q.y*quat.m_q.y + quat.m_q.z*quat.m_q.z)) * v.x +
    (2.0f*(quat.m_q.x*quat.m_q.y - quat.m_q.z*quat.m_q.w)) * v.y +
    (2.0f*(quat.m_q.x*quat.m_q.z + quat.m_q.y*quat.m_q.w)) * v.z;

  const float y =
    (2.0f*(quat.m_q.x*quat.m_q.y + quat.m_q.z*quat.m_q.w)) * v.x +
    (1.0f - 2.0f*(quat.m_q.x*quat.m_q.x + quat.m_q.z*quat.m_q.z)) * v.y +
    (2.0f*(quat.m_q.y*quat.m_q.z - quat.m_q.x*quat.m_q.w)) * v.z;

  const float z =
    (2.0f*(quat.m_q.x*quat.m_q.z - quat.m_q.y*quat.m_q.w)) * v.x +
    (2.0f*(quat.m_q.y*quat.m_q.z + quat.m_q.x*quat.m_q.w)) * v.y +
    (1.0f - 2.0f*(quat.m_q.x*quat.m_q.x + quat.m_q.y*quat.m_q.y)) * v.z;

    return make_float3( x, y, z );
  }
  
  OPTIXU_INLINE RT_HOSTDEVICE float4 operator*( const Quaternion& quat, const float4& v )
  {
    const float3 r = quat * make_float3(v);
    return make_float4( r, 1.0f );
  }

  OPTIXU_INLINE RT_HOSTDEVICE Quaternion nlerp( const Quaternion& quat0, const Quaternion& quat1, float t )
  {
      Quaternion q;
      q.m_q = optix::lerp(quat0.m_q, quat1.m_q, t);
      q.m_q = optix::normalize( q.m_q );
      return q;
  }

} // end namespace optix

#ifdef OPTIXU_INLINE_DEFINED
#  undef OPTIXU_INLINE_DEFINED
#  undef OPTIXU_INLINE
#endif


#endif // #ifndef __optixu_optixu_quaternion_namespace_h__
