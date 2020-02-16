#pragma once

#include "redner.h"
#include "buffer.h"
#include "camera.h"
#include "ray.h"
#include "intersection.h"
#include "shape.h"

struct Scene;
struct DScene;

/// Backpropagate the primary ray intersection to hit vertices & camera
void d_primary_intersection(const Scene &scene,
                            const BufferView<int> &active_pixels,
                            const BufferView<CameraSample> &samples,
                            const BufferView<Ray> &rays,
                            const BufferView<RayDifferential> &primary_ray_differentials,
                            const BufferView<Intersection> &intersections,
                            const BufferView<DRay> &d_rays,
                            const BufferView<RayDifferential> &d_ray_differentials,
                            const BufferView<SurfacePoint> &d_surface_points,
                            DScene *d_scene,
                            float *screen_gradient_image);