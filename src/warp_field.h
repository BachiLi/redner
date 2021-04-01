#pragma once

#include "redner.h"
#include "shape.h"
#include "camera.h"
#include "channels.h"
#include "edge_tree.h"

#include <memory>

struct Scene;
#include "edge.h"
#include "scene.h"
#include "parallel.h"
#include "thrust_utils.h"
#include "ltc.inc"
#include "shape_adjacency.h"

#include "aux.h"
#include "warp_cv.h"

#include <memory>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/remove.h>


/*
 * Samples a stopping count 'N' for the 
 * number of auxillary rays to process.
 * This process is part of the Russian Roulette debiasing method.
 */

void aux_sample_sample_counts( const KernelParameters& kernel_parameters,
                    const Scene& scene,
                    const BufferView<int> &active_pixels,
                    const BufferView<AuxCountSample> &aux_count_samples,
                    BufferView<uint> &aux_sample_counts);


/*
 * Computes the boundary term for the auxillary ray weight.
 */
DEVICE
inline
Real warp_boundary_term( const KernelParameters &kernel_parameters,
                  const Shape *shapes,
                  const ShapeAdjacency *shape_adjacencies,
                  const Ray &primary,
                  const Ray &auxillary,
                  const Intersection &aux_isect,
                  const SurfacePoint &aux_point,
                  const SurfacePoint &shading_point,
                  Real &horizon_term );

/*
 * Computes warp horizon function used for scoring edges
 */
DEVICE
inline
Real warp_horizon_term(const KernelParameters &kernel_parameters,
                    const Shape *shapes,
                    const ShapeAdjacency *shape_adjacencies,
                    const SurfacePoint &shading_point,
                    const SurfacePoint &aux_point,
                    const Intersection &aux_isect,
                    int edge_idx);


/*
 * Computes the asymptotic weight of the auxillary ray's contribution to
 * the warp field. This is independent of the parameter being differentiated against.
 */
DEVICE
inline
Real warp_weight( const KernelParameters &kernel_parameters,
                  const Shape *shapes,
                  const ShapeAdjacency *shape_adjacencies,
                  const Ray &primary,
                  const Ray &auxillary,
                  const Intersection &aux_isect,
                  const SurfacePoint &aux_point,
                  const SurfacePoint &shading_point,
                  Real &boundary_term,
                  Real &horizon_term );


/*
 * Computes the derivative of the asymptotic weight of the 
 * auxillary ray's contribution to
 * the warp field. This is independent of the parameter.
 */
DEVICE
inline
Vector3 warp_weight_grad( const KernelParameters &kernel_parameters,
                  const Shape *shapes,
                  const ShapeAdjacency *shape_adjacencies,
                  const Ray &primary,
                  const Ray &auxillary,
                  const Intersection &aux_isect,
                  const SurfacePoint &aux_point,
                  const SurfacePoint &shading_point );


/*
 * Parallel accumulator that computes the warp components
 * of the derivatives (i.e v.df/dx + f.dv/dx) that accounts for
 * motion of sharp boundaries.
 */
void accumulate_warp_derivatives(const Scene &scene,
                                 const BufferView<int> &active_pixels,
                                 const BufferView<SurfacePoint> &shading_points,
                                 const BufferView<Intersection> &shading_isects,
                                 const BufferView<Ray> &primary_rays,
                                 const BufferView<Intersection> &primary_isects,
                                 const BufferView<SurfacePoint> &primary_points,
                                 const BufferView<uint> &aux_sample_counts,
                                 const BufferView<Ray> &aux_rays,
                                 const BufferView<Intersection> &aux_isects,
                                 const BufferView<SurfacePoint> &aux_points,
                                 const BufferView<AuxSample> &aux_camera_samples,
                                 const BufferView<Vector3> &path_contribs,
                                 const BufferView<Vector3> &d_wos,
                                 const Real weight,
                                 const ChannelInfo channel_info,
                                 const KernelParameters &kernel_parameters,
                                 const float* d_rendered_image,
                                 const bool enable_control_variates, // auxillary control variates
                                 BufferView<SurfacePoint> d_shading_points,
                                 BufferView<DShape> d_shapes,
                                 BufferView<CameraSample> camera_samples,
                                 BufferView<Vector2> d_camera_samples,
                                 ShapeAdjacency* adjacency,
                                 DScene* d_scene,
                                 float* debug_image,
                                 float* screen_gradient_image,
                                 BufferView<Vector3> mean_grad_contrib,
                                 BufferView<Real> mean_contrib,
                                 BufferView<Matrix3x3> sample_covariance);


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