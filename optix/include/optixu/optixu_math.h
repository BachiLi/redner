
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

#ifndef __optixu_optixu_math_h__
#define __optixu_optixu_math_h__

/*
 * This is a backward compatibility header designed to keep all the CUDA vector types in
 * the global namespace.  If you wish to have an unpolluted global namespace use
 * <file>_namespace.h version of the headers instead.  NOTE: within a project, the usage
 * of either the optix or global namespace versions of the header files must be
 * consistent.  Mixing the two can cause errors either at compile time or link time.
 */

/*
 * These need to be included first in order to make sure the CUDA vector types are defined
 * in the global namespace necessary to maintain backward compatibility.
 */
#include <vector_types.h>
#include <vector_functions.h>

/*
 * Any types that were defined in the global namespace in previous versions need to be
 * declared here.  We will also set RT_UINT_USHORT_DEFINED to signal to
 * optixu_math_namespace.h that these types have already been defined here.
 */
#if defined(_WIN32)
  /* uint and ushort are not already defined on Windows systems */
  typedef unsigned int uint;
  typedef unsigned short ushort;
# define RT_UINT_USHORT_DEFINED
#endif

#include "optixu_math_namespace.h"

/*
 * In order to maintain backward compatibility we are pulling all the functions in optix's
 * namespace into the global namespace.  If you need your global namespace unpolluted,
 * include optixu_math_namespace.h instead.
 */


/* From optixu_math_namespace.h */

/* Types */
using optix::uint;
using optix::ushort;

/* Functions that also exist on the system */
using optix::copysignf;
using optix::expf;
using optix::floor;
using optix::fmaxf;
using optix::fminf;
using optix::max;
using optix::min;

/* Useful graphics functions */
using optix::clamp;
using optix::cross;
using optix::dot;
using optix::faceforward;
using optix::fresnel_schlick;
using optix::length;
using optix::lerp;
using optix::normalize;
using optix::reflect;
using optix::refract;
using optix::smoothstep;
using optix::temperature;

#endif /* #ifndef __optixu_optixu_math_h__ */
