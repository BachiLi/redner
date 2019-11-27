
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
 * @file   optix_datatypes.h
 * @author NVIDIA Corporation
 * @brief  OptiX public API
 *
 * OptiX public API Reference - Datatypes
 */

#ifndef __optix_optix_datatypes_h__
#define __optix_optix_datatypes_h__

#include "../optixu/optixu_vector_types.h"        /* for float3 */
#include "optix_declarations.h"         /* for RT_HOSTDEVICE */

#ifdef __cplusplus
namespace optix {
#endif

/** Max t for a ray */
#define RT_DEFAULT_MAX 1.e27f

/*
   Rays
*/

/**
  * @brief Ray class
  *
  * @ingroup CUDACTypes
  *
  * <B>Description</B>
  * 
  * @ref Ray is an encapsulation of a ray mathematical entity.  The
  * origin and direction members specify the ray, while the @ref ray_type
  * member specifies which closest-hit/any-hit pair will be used when the
  * ray hits a geometry object.  The tmin/tmax members specify the
  * interval over which the ray is valid.
  *
  * To avoid numerical range problems, the value @ref RT_DEFAULT_MAX can be
  * used to specify an infinite extent.
  *
  * During C++ compilation, Ray is contained within the \a optix:: namespace
  * but has global scope during C compilation. @ref Ray's constructors are not
  * available during C compilation.
  *
  * <B>Members</B>
  * 
  *@code
  *  // The origin of the ray
  *  float3 origin;
  *
  *  // The direction of the ray
  *  float3 direction;
  *
  *  // The ray type associated with this ray
  *  unsigned int ray_type;
  *
  *  // The min and max extents associated with this ray
  *  float tmin;
  *  float tmax;
  *@endcode
  *
  * <B>Constructors</B>
  *
  *@code
  *  // Create a Ray with undefined member values
  *  Ray( void );
  *
  *  // Create a Ray copied from an exemplar
  *  Ray( const Ray &r );
  *
  *  // Create a ray with a specified origin, direction, ray_type, and min/max extents.
  *  // When tmax is not given, it defaults to @ref RT_DEFAULT_MAX.
  *  Ray( float3 origin, float3 direction, unsigned int ray_type,
  *       float tmin, float tmax = RT_DEFAULT_MAX);
  *@endcode
  *
  * <B>Functions</B>
  *
  *@code
  *    // Create a ray with a specified origin, direction, ray type, and min/max extents.
  *    Ray make_Ray( float3 origin, 
  *              float3 direction, 
  *              unsigned int ray_type, 
  *              float tmin, 
  *              float tmax );
  *@endcode
  *
  * <B>History</B>
  * 
  * @ref Ray was introduced in OptiX 1.0.
  * 
  * <B>See also</B>
  * @ref rtContextSetRayTypeCount,
  * @ref rtMaterialSetAnyHitProgram,
  * @ref rtMaterialSetClosestHitProgram
  */
struct Ray {

#ifdef __cplusplus
  /** Create a Ray with undefined member values */
  __inline__ RT_HOSTDEVICE
  Ray(){}

  /** Create a Ray copied from an exemplar */
  __inline__ RT_HOSTDEVICE
  Ray( const Ray &r)
    :origin(r.origin),direction(r.direction),ray_type(r.ray_type),tmin(r.tmin),tmax(r.tmax){}

  /** Create a ray with a specified origin, direction, ray_type, and min/max extents. When tmax is not given, it defaults to RT_DEFAULT_MAX. */
  __inline__ RT_HOSTDEVICE
  Ray( float3 origin_, float3 direction_, unsigned int ray_type_, float tmin_, float tmax_ = RT_DEFAULT_MAX )
    :origin(origin_),direction(direction_),ray_type(ray_type_),tmin(tmin_),tmax(tmax_){}
#endif

  /** The origin of the ray */
  float3 origin;
  /** The direction of the ray */
  float3 direction;
  /** The ray type associated with this ray */
  unsigned int ray_type;
  /** The min extent associated with this ray */
  float tmin;
  /** The max extent associated with this ray */
  float tmax;
};

static __inline__ RT_HOSTDEVICE
Ray make_Ray( float3 origin, float3 direction, unsigned int ray_type, float tmin, float tmax )
{
  Ray ray;
  ray.origin = origin;
  ray.direction = direction;
  ray.ray_type = ray_type;
  ray.tmin = tmin;
  ray.tmax = tmax;
  return ray;
}

#ifdef __cplusplus
} // namespace
#endif

#endif /* __optix_optix_datatypes_h__ */
