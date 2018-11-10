#pragma once

#include "redner.h"
#include "vector.h"
#include "buffer.h"
#include "ray.h"
#include "intersection.h"
#include "camera.h"
#include "light.h"
#include "shape.h"
#include "material.h"
#include "edge.h"
#include <vector>
#include <memory>
#include <mutex>
#include <embree3/rtcore.h>
#ifdef COMPILE_WITH_CUDA
  #include <optix_prime/optix_primepp.h>
#endif

struct Scene {
    /// XXX should use py::list to avoid copy?
    Scene(const Camera &camera,
          const std::vector<const Shape*> &shapes,
          const std::vector<const Material*> &materials,
          const std::vector<const Light*> &lights,
          bool use_gpu);
    ~Scene();

    // Flatten arrays of scene content
    Camera camera;
    Buffer<Shape> shapes;
    Buffer<Material> materials;
    Buffer<Light> lights;

    // Is the scene stored in GPU or CPU
    bool use_gpu;

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

    // For material derivatives
    std::vector<std::mutex> material_mutexes;

    // For edge sampling
    EdgeSampler edge_sampler;
};

struct FlattenScene {
    Shape *shapes;
    Material *materials;
    Light *lights;
    int num_lights;
    Real *light_pmf;
    Real *light_cdf;
    Real *light_areas;
    Real **area_cdfs;
};

// XXX: Again, some unnecessary copy from Python
struct DScene {
    DScene() {}
    DScene(const DCamera &camera,
           const std::vector<DShape*> &shapes,
           const std::vector<DMaterial*> &materials,
           const std::vector<DLight*> &lights,
           bool use_gpu);
    ~DScene();

    DCamera camera;
    Buffer<DShape> shapes;
    Buffer<DMaterial> materials;
    Buffer<DLight> lights;
};

FlattenScene get_flatten_scene(const Scene &scene);

void intersect(const Scene &scene,
               const BufferView<int> &active_pixels,
               const BufferView<Ray> &rays,
               BufferView<Intersection> intersections,
               BufferView<SurfacePoint> surface_points);
// Set intersection to invalid if occluded
void occluded(const Scene &scene,
              const BufferView<int> &active_pixels,
              const BufferView<Ray> &rays,
              BufferView<Intersection> intersections);
void sample_point_on_light(const Scene &scene,
                           const BufferView<int> &active_pixels,
                           const BufferView<SurfacePoint> &shading_points,
                           const BufferView<LightSample> &samples,
                           BufferView<Intersection> light_isects,
                           BufferView<SurfacePoint> light_points,
                           BufferView<Ray> shadow_ray);

void test_scene_intersect(bool use_gpu);
void test_sample_point_on_light(bool use_gpu);
