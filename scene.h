#pragma once

#include "redner.h"
#include "vector.h"
#include "buffer.h"
#include "ray.h"
#include "intersection.h"
#include "camera.h"
#include "area_light.h"
#include "shape.h"
#include "material.h"
#include "envmap.h"
#include "edge.h"
#include <vector>
#include <memory>
#include <embree3/rtcore.h>
#ifdef COMPILE_WITH_CUDA
  #include <optix_prime/optix_primepp.h>
#endif

struct Scene {
    /// XXX should use py::list to avoid copy?
    Scene(const Camera &camera,
          const std::vector<const Shape*> &shapes,
          const std::vector<const Material*> &materials,
          const std::vector<const AreaLight*> &area_lights,
          const std::shared_ptr<const EnvironmentMap> &envmap,
          bool use_gpu,
          int gpu_index,
          bool use_primary_edge_sampling,
          bool use_secondary_edge_sampling);
    ~Scene();

    // Flatten arrays of scene content
    Camera camera;
    Buffer<Shape> shapes;
    Buffer<Material> materials;
    Buffer<AreaLight> area_lights;
    EnvironmentMap *envmap;

    // Is the scene stored in GPU or CPU
    bool use_gpu;
    int gpu_index;
    bool use_primary_edge_sampling;
    bool use_secondary_edge_sampling;

    // For G-buffer rendering with textures of arbitrary number of channels.
    int max_generic_texture_dimension;

#ifdef COMPILE_WITH_CUDA
    // Optix handles
    optix::prime::Context optix_context;
    std::vector<optix::prime::Model> optix_models;
    std::vector<RTPmodel> optix_instances;
    std::vector<Matrix4x4f> transforms;
    optix::prime::Model optix_scene;
#endif

    // Embree handles
    RTCDevice embree_device;
    RTCScene embree_scene;

    // Light sampling
    Buffer<Real> light_pmf;
    Buffer<Real> light_cdf;
    Buffer<Real> light_areas;
    Buffer<Real*> area_cdfs;
    Buffer<Real> area_cdf_pool;

    // For edge sampling
    EdgeSampler edge_sampler;
};

inline
bool has_lights(const Scene &scene) {
    return scene.area_lights.size() > 0 || scene.envmap != nullptr;
}

struct FlattenScene {
    Shape *shapes;
    Material *materials;
    AreaLight *area_lights;
    int num_lights;
    Real *light_pmf;
    Real *light_cdf;
    Real *light_areas;
    Real **area_cdfs;
    EnvironmentMap *envmap;
    // For G-buffer rendering with textures of arbitrary number of channels.
    int max_generic_texture_dimension;
};

// XXX: Again, some unnecessary copy from Python
struct DScene {
    DScene() {}
    DScene(const DCamera &camera,
           const std::vector<DShape*> &shapes,
           const std::vector<DMaterial*> &materials,
           const std::vector<DAreaLight*> &lights,
           const std::shared_ptr<DEnvironmentMap> &envmap,
           bool use_gpu,
           int gpu_index);
    ~DScene();

    DCamera camera;
    Buffer<DShape> shapes;
    Buffer<DMaterial> materials;
    Buffer<DAreaLight> area_lights;
    DEnvironmentMap *envmap;
    bool use_gpu;
    int gpu_index;
};

FlattenScene get_flatten_scene(const Scene &scene);

void intersect(const Scene &scene,
               const BufferView<int> &active_pixels,
               BufferView<Ray> rays,
               const BufferView<RayDifferential> &ray_differentials,
               BufferView<Intersection> intersections,
               BufferView<SurfacePoint> surface_points,
               BufferView<RayDifferential> new_ray_differentials,
               BufferView<OptiXRay> optix_rays,
               BufferView<OptiXHit> optix_hits);
// Set ray.tmax to negative if occluded
void occluded(const Scene &scene,
              const BufferView<int> &active_pixels,
              BufferView<Ray> rays,
              BufferView<OptiXRay> optix_rays,
              BufferView<OptiXHit> optix_hits);
void sample_point_on_light(const Scene &scene,
                           const BufferView<int> &active_pixels,
                           const BufferView<SurfacePoint> &shading_points,
                           const BufferView<LightSample> &samples,
                           BufferView<Intersection> light_isects,
                           BufferView<SurfacePoint> light_points,
                           BufferView<Ray> shadow_ray);

void test_scene_intersect(bool use_gpu);
void test_sample_point_on_light(bool use_gpu);
