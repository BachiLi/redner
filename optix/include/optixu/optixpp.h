
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

#ifndef __optixu_optixpp_h__
#define __optixu_optixpp_h__

/*
  optixpp.h used to include optix_math.h, so we include it here for backward
  compatibility.  Please include optixpp_namespace.h and optixu_math_namespace.h separately
  in the future.

  This needs to be included first in order to make sure the CUDA vector types are defined
  in the global namespace necessary to maintain backward compatibility.

*/

#include "optixu_math.h"

#include "optixpp_namespace.h"

/*
 * optixpp's classes originally were in the optixu namespace but have subsequently been
 * moved to the optix namespace.  For backward compatibility we provide the optixu
 * namespace here.
 */

namespace optixu {

  using optix::AccelerationObj;
  using optix::BufferObj;
  using optix::ContextObj;
  using optix::GeometryObj;
  using optix::GeometryTrianglesObj;
  using optix::GeometryGroupObj;
  using optix::GeometryInstanceObj;
  using optix::GroupObj;
  using optix::MaterialObj;
  using optix::ProgramObj;
  using optix::SelectorObj;
  using optix::TextureSamplerObj;
  using optix::TransformObj;
  using optix::VariableObj;

  using optix::APIObj;
  using optix::DestroyableObj;
  using optix::ScopedObj;

  using optix::Handle;

  using optix::Acceleration;
  using optix::Buffer;
  using optix::Context;
  using optix::Geometry;
  using optix::GeometryTriangles;
  using optix::GeometryGroup;
  using optix::GeometryInstance;
  using optix::Group;
  using optix::Material;
  using optix::Program;
  using optix::Selector;
  using optix::TextureSampler;
  using optix::Transform;
  using optix::Variable;

  using optix::Exception;

  using optix::bufferId;
} /* end namespace optixu */


#endif /* __optixu_optixpp_h__ */


