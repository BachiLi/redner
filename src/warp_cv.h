#pragma once

#include "redner.h"
#include "shape.h"
#include "camera.h"

struct Scene;
#include "scene.h"

#include "parallel.h"
#include "thrust_utils.h"
#include "ltc.inc"

#include "warp_common.h"

void accumulate_control_variates(const Scene& scene,
                                 const KernelParameters& kernel_parameters,
                                 const BufferView<int>& active_pixels,
                                 const BufferView<SurfacePoint>& control_points,
                                 const BufferView<Intersection>& control_isects,
                                 const BufferView<Ray>& control_rays,
                                 const BufferView<RayDifferential>& control_ray_differentials,
                                 const BufferView<CameraSample>& control_samples,
                                 const BufferView<Vector3>& control_mean_grad_contrib,
                                 const BufferView<Real>& control_mean_contrib,
                                 const BufferView<Matrix3x3>& control_sample_covariance,
                                 const BufferView<DShape>& d_shapes,
                                 const Real weight,
                                 float* debug_image);