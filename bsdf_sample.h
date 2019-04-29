#pragma once

#include "redner.h"
#include "buffer.h"
#include "ray.h"
#include "intersection.h"
#include "material.h"

struct Scene;

/// Given incoming rays & intersected surfaces, sample the next rays based on the material
void bsdf_sample(const Scene &scene,
                 const BufferView<int> &active_pixels,
                 const BufferView<Ray> &incoming_rays,
                 const BufferView<RayDifferential> &incoming_ray_differentials,
                 const BufferView<Intersection> &shading_isects,
                 const BufferView<SurfacePoint> &shading_points,
                 const BufferView<BSDFSample> &bsdf_samples,
                 const BufferView<Real> &min_roughness,
                 BufferView<Ray> next_rays,
                 BufferView<RayDifferential> bsdf_ray_differentials,
                 BufferView<Real> next_min_roughness);