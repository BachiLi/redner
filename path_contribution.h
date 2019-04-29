#pragma once

#include "redner.h"
#include "buffer.h"
#include "ray.h"
#include "intersection.h"
#include "shape.h"
#include "texture.h"
#include "area_light.h"
#include "material.h"
#include "matrix.h"

struct Scene;
struct ChannelInfo;

/// Compute the contribution at a path vertex, by combining next event estimation & BSDF sampling. 
void accumulate_path_contribs(const Scene &scene,
                              const BufferView<int> &active_pixels,
                              const BufferView<Vector3> &throughputs,
                              const BufferView<Ray> &incoming_rays,
                              const BufferView<Intersection> &shading_isects,
                              const BufferView<SurfacePoint> &shading_points,
                              const BufferView<Intersection> &light_isects,
                              const BufferView<SurfacePoint> &light_points,
                              const BufferView<Ray> &light_rays,
                              const BufferView<Intersection> &bsdf_isects,
                              const BufferView<SurfacePoint> &bsdf_points,
                              const BufferView<Ray> &bsdf_rays,
                              const BufferView<Real> &min_roughness,
                              const Real weight,
                              const ChannelInfo &channel_info,
                              BufferView<Vector3> next_throughputs,
                              float *rendered_image,
                              BufferView<Real> edge_contribs);

/// The backward version of the function above.
void d_accumulate_path_contribs(const Scene &scene,
                                const BufferView<int> &active_pixels,
                                const BufferView<Vector3> &throughputs,
                                const BufferView<Ray> &incoming_rays,
                                const BufferView<RayDifferential> &ray_differentials,
                                const BufferView<LightSample> &light_samples,
                                const BufferView<BSDFSample> &bsdf_samples,
                                const BufferView<Intersection> &shading_isects,
                                const BufferView<SurfacePoint> &shading_points,
                                const BufferView<Intersection> &light_isects,
                                const BufferView<SurfacePoint> &light_points,
                                const BufferView<Ray> &light_rays,
                                const BufferView<Intersection> &bsdf_isects,
                                const BufferView<SurfacePoint> &bsdf_points,
                                const BufferView<Ray> &bsdf_rays,
                                const BufferView<RayDifferential> &bsdf_ray_differentials,
                                const BufferView<Real> &min_roughness,
                                const Real weight,
                                const ChannelInfo &channel_info,
                                const float *d_rendered_image,
                                const BufferView<Vector3> &d_next_throughputs,
                                const BufferView<DRay> &d_next_rays,
                                const BufferView<RayDifferential> &d_next_ray_differentials,
                                const BufferView<SurfacePoint> &d_next_points,
                                BufferView<DVertex> d_light_vertices,
                                BufferView<DVertex> d_bsdf_vertices,
                                BufferView<DTexture3> d_diffuse_texs,
                                BufferView<DTexture3> d_specular_texs,
                                BufferView<DTexture1> d_roughness_texs,
                                BufferView<DAreaLightInst> d_nee_lights,
                                BufferView<DAreaLightInst> d_bsdf_lights,
                                BufferView<DTexture3> d_envmap_vals,
                                BufferView<Matrix4x4> d_world_to_envs,
                                BufferView<Vector3> d_throughputs,
                                BufferView<DRay> d_incoming_rays,
                                BufferView<RayDifferential> d_incoming_ray_differentials,
                                BufferView<SurfacePoint> d_shading_points);