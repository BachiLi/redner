
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
 * @file   optixu_aabb_namespace.h
 * @author NVIDIA Corporation
 * @brief  OptiX public API
 *
 * OptiX public API Reference - Public AABB namespace
 */
 
#ifndef __optixu_optixu_aabb_namespace_h__
#define __optixu_optixu_aabb_namespace_h__

#include "optixu_math_namespace.h"

#ifndef __CUDACC__
#  include <assert.h>
#  define RT_AABB_ASSERT assert
#else
#  define RT_AABB_ASSERT(x)
#endif

// __forceinline__ works in cuda, VS, and with gcc.  Leave it as macro in case
// we need to make this per-platform or we want to switch off inlining globally.
#ifndef OPTIXU_INLINE 
#  define OPTIXU_INLINE_DEFINED 1
#  define OPTIXU_INLINE __forceinline__
#endif // OPTIXU_INLINE 

namespace optix {

 /**
  * @brief Axis-aligned bounding box
  * 
  * @ingroup CUDACTypes
  * 
  * <B>Description</B>
  * 
  * @ref Aabb is a utility class for computing and manipulating axis-aligned
  * bounding boxes (aabbs).  Aabb is primarily useful in the bounding box
  * program associated with geometry objects. Aabb
  * may also be useful in other computation and can be used in both host
  * and device code. 
  *
  * <B>History</B>
  * 
  * @ref Aabb was introduced in OptiX 1.0.
  * 
  * <B>See also</B>
  * @ref RT_PROGRAM,
  * @ref rtGeometrySetBoundingBoxProgram
  * 
  */
  class Aabb
  {
  public:

    /** Construct an invalid box */
    RT_HOSTDEVICE Aabb(); 

    /** Construct from min and max vectors */
    RT_HOSTDEVICE Aabb( const float3& min, const float3& max ); 

    /** Construct from three points (e.g. triangle) */
    RT_HOSTDEVICE Aabb( const float3& v0, const float3& v1, const float3& v2 );

    /** Exact equality */
    RT_HOSTDEVICE bool operator==( const Aabb& other ) const;

    /** Array access */
    RT_HOSTDEVICE float3& operator[]( int i );

    /** Const array access */
    RT_HOSTDEVICE const float3& operator[]( int i ) const;

    /** Set using two vectors */
    RT_HOSTDEVICE void set( const float3& min, const float3& max );

    /** Set using three points (e.g. triangle) */
    RT_HOSTDEVICE void set( const float3& v0, const float3& v1, const float3& v2 );

    /** Invalidate the box */
    RT_HOSTDEVICE void invalidate();

    /** Check if the box is valid */
    RT_HOSTDEVICE bool valid() const;

    /** Check if the point is in the box */
    RT_HOSTDEVICE bool contains( const float3& p ) const;

    /** Check if the box is fully contained in the box */
    RT_HOSTDEVICE bool contains( const Aabb& bb ) const;

    /** Extend the box to include the given point */
    RT_HOSTDEVICE void include( const float3& p );

    /** Extend the box to include the given box */
    RT_HOSTDEVICE void include( const Aabb& other );

    /** Extend the box to include the given box */
    RT_HOSTDEVICE void include( const float3& min, const float3& max );

    /** Compute the box center */
    RT_HOSTDEVICE float3 center() const;

    /** Compute the box center in the given dimension */
    RT_HOSTDEVICE float center( int dim ) const;

    /** Compute the box extent */
    RT_HOSTDEVICE float3 extent() const;

    /** Compute the box extent in the given dimension */
    RT_HOSTDEVICE float extent( int dim ) const;

    /** Compute the volume of the box */
    RT_HOSTDEVICE float volume() const;

    /** Compute the surface area of the box */
    RT_HOSTDEVICE float area() const;

    /** Compute half the surface area of the box */
    RT_HOSTDEVICE float halfArea() const;

    /** Get the index of the longest axis */
    RT_HOSTDEVICE int longestAxis() const;

    /** Get the extent of the longest axis */
    RT_HOSTDEVICE float maxExtent() const;

    /** Check for intersection with another box */
    RT_HOSTDEVICE bool intersects( const Aabb& other ) const;

    /** Make the current box be the intersection between this one and another one */
    RT_HOSTDEVICE void intersection( const Aabb& other );

    /** Enlarge the box by moving both min and max by 'amount' */
    RT_HOSTDEVICE void enlarge( float amount );

    /** Check if the box is flat in at least one dimension  */
    RT_HOSTDEVICE bool isFlat() const;

    /** Compute the minimum Euclidean distance from a point on the
     surface of this Aabb to the point of interest */
    RT_HOSTDEVICE float distance( const float3& x ) const;

    /** Compute the minimum squared Euclidean distance from a point on the
     surface of this Aabb to the point of interest */
    RT_HOSTDEVICE float distance2( const float3& x ) const;

    /** Compute the minimum Euclidean distance from a point on the surface
      of this Aabb to the point of interest.
      If the point of interest lies inside this Aabb, the result is negative  */
    RT_HOSTDEVICE float signedDistance( const float3& x ) const;

    /** Min bound */
    float3 m_min;
    /** Max bound */
    float3 m_max;
  };


  OPTIXU_INLINE RT_HOSTDEVICE Aabb::Aabb()
  {
    invalidate();
  }

  OPTIXU_INLINE RT_HOSTDEVICE Aabb::Aabb( const float3& min, const float3& max )
  {
    set( min, max );
  }

  OPTIXU_INLINE RT_HOSTDEVICE Aabb::Aabb( const float3& v0, const float3& v1, const float3& v2 )
  {
    set( v0, v1, v2 );
  }

  OPTIXU_INLINE RT_HOSTDEVICE bool Aabb::operator==( const Aabb& other ) const
  {
    return m_min.x == other.m_min.x &&
           m_min.y == other.m_min.y &&
           m_min.z == other.m_min.z &&
           m_max.x == other.m_max.x &&
           m_max.y == other.m_max.y &&
           m_max.z == other.m_max.z;
  }

  OPTIXU_INLINE RT_HOSTDEVICE float3& Aabb::operator[]( int i )
  {
    RT_AABB_ASSERT( i>=0 && i<=1 );
    return (&m_min)[i];
  }

  OPTIXU_INLINE RT_HOSTDEVICE const float3& Aabb::operator[]( int i ) const
  {
    RT_AABB_ASSERT( i>=0 && i<=1 );
    return (&m_min)[i];
  }

  OPTIXU_INLINE RT_HOSTDEVICE void Aabb::set( const float3& min, const float3& max )
  {
    m_min = min;
    m_max = max;
  }

  OPTIXU_INLINE RT_HOSTDEVICE void Aabb::set( const float3& v0, const float3& v1, const float3& v2 )
  {
    m_min = fminf( v0, fminf(v1,v2) );
    m_max = fmaxf( v0, fmaxf(v1,v2) );
  }

  OPTIXU_INLINE RT_HOSTDEVICE void Aabb::invalidate()
  {
    m_min = make_float3(  1e37f );
    m_max = make_float3( -1e37f );
  }

  OPTIXU_INLINE RT_HOSTDEVICE bool Aabb::valid() const
  {
    return m_min.x <= m_max.x &&
      m_min.y <= m_max.y &&
      m_min.z <= m_max.z;
  }

  OPTIXU_INLINE RT_HOSTDEVICE bool Aabb::contains( const float3& p ) const
  {
    return  p.x >= m_min.x && p.x <= m_max.x &&
            p.y >= m_min.y && p.y <= m_max.y &&
            p.z >= m_min.z && p.z <= m_max.z;
  }

  OPTIXU_INLINE RT_HOSTDEVICE bool Aabb::contains( const Aabb& bb ) const
  {
    return contains( bb.m_min ) && contains( bb.m_max );
  }

  OPTIXU_INLINE RT_HOSTDEVICE void Aabb::include( const float3& p )
  {
    m_min = fminf( m_min, p );
    m_max = fmaxf( m_max, p );
  }

  OPTIXU_INLINE RT_HOSTDEVICE void Aabb::include( const Aabb& other )
  {
    m_min = fminf( m_min, other.m_min );
    m_max = fmaxf( m_max, other.m_max );
  }

  OPTIXU_INLINE RT_HOSTDEVICE void Aabb::include( const float3& min, const float3& max )
  {
    m_min = fminf( m_min, min );
    m_max = fmaxf( m_max, max );
  }

  OPTIXU_INLINE RT_HOSTDEVICE float3 Aabb::center() const
  {
    RT_AABB_ASSERT( valid() );
    return (m_min+m_max) * 0.5f;
  }

  OPTIXU_INLINE RT_HOSTDEVICE float Aabb::center( int dim ) const
  {
    RT_AABB_ASSERT( valid() );
    RT_AABB_ASSERT( dim>=0 && dim<=2 );
    return ( ((const float*)(&m_min))[dim] + ((const float*)(&m_max))[dim] ) * 0.5f;
  }

  OPTIXU_INLINE RT_HOSTDEVICE float3 Aabb::extent() const
  {
    RT_AABB_ASSERT( valid() );
    return m_max - m_min;
  }

  OPTIXU_INLINE RT_HOSTDEVICE float Aabb::extent( int dim ) const
  {
    RT_AABB_ASSERT( valid() );
    return ((const float*)(&m_max))[dim] - ((const float*)(&m_min))[dim];
  }

  OPTIXU_INLINE RT_HOSTDEVICE float Aabb::volume() const
  {
    RT_AABB_ASSERT( valid() );
    const float3 d = extent();
    return d.x*d.y*d.z;
  }

  OPTIXU_INLINE RT_HOSTDEVICE float Aabb::area() const
  {
    return 2.0f * halfArea();
  }

  OPTIXU_INLINE RT_HOSTDEVICE float Aabb::halfArea() const
  {
    RT_AABB_ASSERT( valid() );
    const float3 d = extent();
    return d.x*d.y + d.y*d.z + d.z*d.x;
  }

  OPTIXU_INLINE RT_HOSTDEVICE int Aabb::longestAxis() const
  {
    RT_AABB_ASSERT( valid() );
    const float3 d = extent();

    if( d.x > d.y )
      return d.x > d.z ? 0 : 2;
    return d.y > d.z ? 1 : 2;
  }

  OPTIXU_INLINE RT_HOSTDEVICE float Aabb::maxExtent() const
  {
    return extent( longestAxis() );
  }

  OPTIXU_INLINE RT_HOSTDEVICE bool Aabb::intersects( const Aabb& other ) const
  {
    if( other.m_min.x > m_max.x || other.m_max.x < m_min.x ) return false;
    if( other.m_min.y > m_max.y || other.m_max.y < m_min.y ) return false;
    if( other.m_min.z > m_max.z || other.m_max.z < m_min.z ) return false;
    return true;
  }

  OPTIXU_INLINE RT_HOSTDEVICE void Aabb::intersection( const Aabb& other )
  {
    m_min.x = fmaxf( m_min.x, other.m_min.x );
    m_min.y = fmaxf( m_min.y, other.m_min.y );
    m_min.z = fmaxf( m_min.z, other.m_min.z );
    m_max.x = fminf( m_max.x, other.m_max.x );
    m_max.y = fminf( m_max.y, other.m_max.y );
    m_max.z = fminf( m_max.z, other.m_max.z );
  }

  OPTIXU_INLINE RT_HOSTDEVICE void Aabb::enlarge( float amount )
  {
    RT_AABB_ASSERT( valid() );
    m_min -= make_float3( amount );
    m_max += make_float3( amount );
  }

  OPTIXU_INLINE RT_HOSTDEVICE bool Aabb::isFlat() const
  {
    return m_min.x == m_max.x ||
           m_min.y == m_max.y ||
           m_min.z == m_max.z;
  }

  OPTIXU_INLINE RT_HOSTDEVICE float Aabb::distance( const float3& x ) const
  {
    return sqrtf(distance2(x));
  }

  OPTIXU_INLINE RT_HOSTDEVICE float Aabb::signedDistance( const float3& x ) const
  {
    if( m_min.x <= x.x && x.x <= m_max.x &&
        m_min.y <= x.y && x.y <= m_max.y &&
        m_min.z <= x.z && x.z <= m_max.z) {
      float distance_x = fminf( x.x - m_min.x, m_max.x - x.x);
      float distance_y = fminf( x.y - m_min.y, m_max.y - x.y);
      float distance_z = fminf( x.z - m_min.z, m_max.z - x.z);

      float min_distance = fminf(distance_x, fminf(distance_y, distance_z));
      return -min_distance;
    }

    return distance(x);
  }

  OPTIXU_INLINE RT_HOSTDEVICE float Aabb::distance2( const float3& x ) const
  {
    float3 box_dims = m_max - m_min;

    // compute vector from min corner of box
    float3 v = x - m_min;

    float dist2 = 0;
    float excess;

    // project vector from box min to x on each axis,
    // yielding distance to x along that axis, and count
    // any excess distance outside box extents

    excess = 0;
    if( v.x < 0 )
      excess = v.x;
    else if( v.x > box_dims.x )
      excess = v.x - box_dims.x;
    dist2 += excess * excess;

    excess = 0;
    if( v.y < 0 )
      excess = v.y;
    else if( v.y > box_dims.y )
      excess = v.y - box_dims.y;
    dist2 += excess * excess;

    excess = 0;
    if( v.z < 0 )
      excess = v.z;
    else if( v.z > box_dims.z )
      excess = v.z - box_dims.z;
    dist2 += excess * excess;

    return dist2;
  }

} // end namespace optix

#ifdef OPTIXU_INLINE_DEFINED
#  undef OPTIXU_INLINE_DEFINED
#  undef OPTIXU_INLINE
#endif

#undef RT_AABB_ASSERT

#endif // #ifndef __optixu_optixu_aabb_namespace_h__
