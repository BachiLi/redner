#include "scene.h"
#include "cuda_utils.h"
#include "intersection.h"
#include "parallel.h"
#include "test_utils.h"
#include "edge.h"
#include "thrust_utils.h"

#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <embree3/rtcore_ray.h>
#include <algorithm>

struct vector3f_min {
    DEVICE Vector3f operator()(const Vector3f &a, const Vector3f &b) const {
        return Vector3{min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
    }
};

struct vector3f_max {
    DEVICE Vector3f operator()(const Vector3f &a, const Vector3f &b) const {
        return Vector3{max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)};
    }
};

struct area_computer {
    DEVICE void operator()(int idx) {
        area[idx] = get_area(shape, idx);
    }

    Shape shape;
    Real *area;
};

Real compute_area_cdf(const Shape &shape, Real *cdf, bool use_gpu) {
    parallel_for(area_computer{shape, cdf}, shape.num_triangles, use_gpu);
    // cdf now stores the areas
    // First ask for the total area
    #pragma warning(disable : 940 969)
    auto total_area = DISPATCH(use_gpu, thrust::reduce,
        cdf, cdf + shape.num_triangles, Real(0), thrust::plus<Real>());
    #pragma warning(default : 940 969)
    // In-place prefix sum
    // XXX: for some reason the program crashes when I use exclusive_scan
    //thrust::exclusive_scan(dev_ptr, dev_ptr + shape.num_triangles, dev_ptr);
    DISPATCH(use_gpu, thrust::transform_exclusive_scan,
             cdf, cdf + shape.num_triangles, cdf,
             thrust::identity<Real>(), Real(0), thrust::plus<Real>());
    // Normalize the CDF by total area
    DISPATCH(use_gpu, thrust::transform,
             cdf, cdf + shape.num_triangles,
             thrust::make_constant_iterator(total_area),
             cdf, thrust::divides<Real>());
    if (use_gpu) {
        cuda_synchronize();
    }
    return total_area;
}

Scene::Scene(const Camera &camera,
             const std::vector<const Shape*> &shapes,
             const std::vector<const Material*> &materials,
             const std::vector<const AreaLight*> &area_lights,
             const std::shared_ptr<const EnvironmentMap> &envmap,
             bool use_gpu,
             int gpu_index,
             bool use_primary_edge_sampling,
             bool use_secondary_edge_sampling)
        : camera(camera), use_gpu(use_gpu), gpu_index(gpu_index),
          use_primary_edge_sampling(use_primary_edge_sampling),
          use_secondary_edge_sampling(use_secondary_edge_sampling) {
#ifdef __NVCC__
    int old_device_id = -1;
#endif
    if (use_gpu) {
#ifdef __NVCC__
        // Initialize the scene in another thread, since optix prime calls cudaSetDeviceFlags
        // and becomes unhappy if we create a context in the main thread
        checkCuda(cudaGetDevice(&old_device_id));
        if (gpu_index != -1) {
            checkCuda(cudaSetDevice(gpu_index));
        }
        // Initialize Optix prime scene
        // FIXME: optix context creation calls cudaDeviceSetFlags(), but we already
        // activate CUDA before this. Ideally we want to move context creation to an initialization
        // phase, but we also want to have a context for each GPU.
        // We should create a context array in global memory and fetch the corresponding context.
        optix_context = optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);
        if (gpu_index != -1) {
            optix_context->setCudaDeviceNumbers({(uint32_t)gpu_index});
        }
        optix_models.resize(shapes.size());
        optix_instances.resize(shapes.size());
        transforms.resize(shapes.size(), Matrix4x4f::identity());
        for (int shape_id = 0; shape_id < (int)shapes.size(); shape_id++) {
            const Shape *shape = shapes[shape_id];
            optix_models[shape_id] = optix_context->createModel();
            optix_models[shape_id]->setTriangles(
                shape->num_triangles, RTP_BUFFER_TYPE_CUDA_LINEAR, shape->indices,
                shape->num_vertices, RTP_BUFFER_TYPE_CUDA_LINEAR, shape->vertices);
            optix_models[shape_id]->update(RTP_MODEL_HINT_ASYNC);
            optix_instances[shape_id] = optix_models[shape_id]->getRTPmodel();
        }

        for (int shape_id = 0; shape_id < (int)shapes.size(); shape_id++) {
            optix_models[shape_id]->finish();
        }

        optix_scene = optix_context->createModel();
        if (shapes.size() > 0) {
            optix_scene->setInstances(
                (int)shapes.size(), RTP_BUFFER_TYPE_HOST, &optix_instances[0], 
                RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x4, RTP_BUFFER_TYPE_HOST, &transforms[0]);
        } else {
            // Hack: the last argument is the pointer to a buffer, but optix prime
            // complains if we pass nullptr. Therefore we give it a non zero number.
            optix_scene->setTriangles(0, RTP_BUFFER_TYPE_CUDA_LINEAR, (const void *)1);
        }
        optix_scene->update(RTP_MODEL_HINT_NONE);
        optix_scene->finish();
#else
        assert(false);
#endif
    } else {
        // Initialize Embree scene
        embree_device = rtcNewDevice(nullptr);
        embree_scene = rtcNewScene(embree_device);
        rtcSetSceneBuildQuality(embree_scene, RTC_BUILD_QUALITY_HIGH);
        rtcSetSceneFlags(embree_scene, RTC_SCENE_FLAG_ROBUST);
        // Copy the scene into Embree (since Embree requires 16 bytes alignment)
        for (const Shape *shape : shapes) {
            auto mesh = rtcNewGeometry(embree_device, RTC_GEOMETRY_TYPE_TRIANGLE);
            auto vertices = (Vector4f*)rtcSetNewGeometryBuffer(
                mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                sizeof(Vector4f), shape->num_vertices);
            for (auto i = 0; i < shape->num_vertices; i++) {
                auto vertex = get_vertex(*shape, i);
                vertices[i] = Vector4f{vertex[0], vertex[1], vertex[2], 0.f};
            }
            auto triangles = (Vector3i*) rtcSetNewGeometryBuffer(
                mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                sizeof(Vector3i), shape->num_triangles);
            for (auto i = 0; i < shape->num_triangles; i++) {
                triangles[i] = get_indices(*shape, i);
            }
            rtcSetGeometryVertexAttributeCount(mesh, 1);
            rtcCommitGeometry(mesh);
            rtcAttachGeometry(embree_scene, mesh);
            rtcReleaseGeometry(mesh);
        }
        rtcCommitScene(embree_scene);
    }

    // Compute bounding sphere
    Sphere bsphere;
    auto scene_min_pos = Vector3f{
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()};
    auto scene_max_pos = Vector3f{
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    for (int shape_id = 0; shape_id < (int)shapes.size(); shape_id++) {
        const auto &shape = *shapes[shape_id];
        const auto *vertices = (const Vector3f *)shape.vertices;
        auto min_pos = DISPATCH(use_gpu, thrust::reduce,
            vertices, vertices + shape.num_vertices,
            Vector3f{std::numeric_limits<float>::infinity(),
                     std::numeric_limits<float>::infinity(),
                     std::numeric_limits<float>::infinity()},
            vector3f_min{});
        auto max_pos = DISPATCH(use_gpu, thrust::reduce,
            vertices, vertices + shape.num_vertices,
            Vector3f{-std::numeric_limits<float>::infinity(),
                     -std::numeric_limits<float>::infinity(),
                     -std::numeric_limits<float>::infinity()},
            vector3f_max{});
        scene_min_pos = Vector3f{min(min_pos.x, scene_min_pos.x),
                                 min(min_pos.y, scene_min_pos.y),
                                 min(min_pos.y, scene_min_pos.z)};
        scene_max_pos = Vector3f{max(max_pos.x, scene_max_pos.x),
                                 max(max_pos.y, scene_max_pos.y),
                                 max(max_pos.y, scene_max_pos.z)};
    }
    if (shapes.size() > 0) {
        bsphere.center = 0.5f * (scene_min_pos + scene_max_pos);
        bsphere.radius = 0.5f * length(scene_max_pos - scene_min_pos);
    } else {
        bsphere.center = Vector3{0, 0, 0};
        bsphere.radius = 0;
    }

    if (area_lights.size() > 0 || envmap.get() != nullptr) {
        auto num_lights = (int)area_lights.size();
        if (envmap.get() != nullptr) {
            num_lights++;
        }
        auto envmap_id = (int)area_lights.size();
        // Build Light CDFs
        light_pmf = Buffer<Real>(use_gpu, num_lights);
        light_areas = Buffer<Real>(use_gpu, area_lights.size());
        // For each area light we build a CDF using area of triangles
        area_cdfs = Buffer<Real*>(use_gpu, area_lights.size());
        auto total_light_triangles = 0;
        for (int light_id = 0; light_id < (int)area_lights.size(); light_id++) {
            const AreaLight &light = *area_lights[light_id];
            const Shape &shape = *shapes[light.shape_id];
            total_light_triangles += shape.num_triangles;
        }
        area_cdf_pool = Buffer<Real>(use_gpu, total_light_triangles);
        auto cur_tri_id = 0;
        for (int light_id = 0; light_id < (int)area_lights.size(); light_id++) {
            const AreaLight &light = *area_lights[light_id];
            const Shape &shape = *shapes[light.shape_id];
            area_cdfs[light_id] = area_cdf_pool.begin() + cur_tri_id;
            cur_tri_id += shape.num_triangles;
        }
        auto total_importance = Real(0);
        for (int light_id = 0; light_id < (int)area_lights.size(); light_id++) {
            const AreaLight &light = *area_lights[light_id];
            const Shape &shape = *shapes[light.shape_id];
            auto area_sum = compute_area_cdf(shape, area_cdfs[light_id], use_gpu);
            light_areas[light_id] = area_sum;
            // Power of an area light
            light_pmf[light_id] = area_sum * luminance(light.intensity) * Real(M_PI);
            total_importance += light_pmf[light_id];
        }
        if (envmap.get() != nullptr) {
            auto surface_area = 4 * Real(M_PI) * square(bsphere.radius);
            if (surface_area > 0) {
                light_pmf[envmap_id] = surface_area / envmap->pdf_norm;
                total_importance += light_pmf[envmap_id];
            } else {
                light_pmf[envmap_id] = 1;
                total_importance += 1;
            }
        }

        assert(total_importance > Real(0));
        // Normalize PMF
        std::transform(light_pmf.data, light_pmf.data + num_lights,
                       light_pmf.data, [=](Real x) {return x / total_importance;});
        // Prefix sum for CDF
        light_cdf = Buffer<Real>(use_gpu, num_lights);
        light_cdf[0] = 0;
        for (int i = 1; i < num_lights; i++) {
            light_cdf[i] = light_cdf[i - 1] + light_pmf[i - 1];
        }
    }

    // Flatten the scene into array
    // TODO: use cudaMemcpyAsync for gpu code path
    if (shapes.size() > 0) {
        this->shapes = Buffer<Shape>(use_gpu, shapes.size());
        for (int shape_id = 0; shape_id < (int)shapes.size(); shape_id++) {
            this->shapes[shape_id] = *shapes[shape_id];
        }
    }
    if (materials.size() > 0) {
        this->materials = Buffer<Material>(use_gpu, materials.size());
        for (int material_id = 0; material_id < (int)materials.size(); material_id++) {
            this->materials[material_id] = *materials[material_id];
        }
    }
    if (area_lights.size() > 0) {
        this->area_lights = Buffer<AreaLight>(use_gpu, area_lights.size());
        for (int light_id = 0; light_id < (int)area_lights.size(); light_id++) {
            this->area_lights[light_id] = *area_lights[light_id];
        }
    }

    if (envmap.get() != nullptr) {
        if (use_gpu) {
#ifdef __NVCC__
            checkCuda(cudaMallocManaged(&this->envmap, sizeof(EnvironmentMap)));
#else
            assert(false);
#endif
        } else {
            this->envmap = new EnvironmentMap;
        }
        *(this->envmap) = *envmap;
    } else {
        this->envmap = nullptr;
    }

    max_generic_texture_dimension = 0;
    for (int material_id = 0; material_id < (int)materials.size(); material_id++) {
        if (materials[material_id]->generic_texture.num_levels > 0) {
            max_generic_texture_dimension =
                std::max(max_generic_texture_dimension,
                         materials[material_id]->generic_texture.channels);
        }
    }

    edge_sampler = EdgeSampler(shapes, *this);

#ifdef __NVCC__
    if (old_device_id != -1) {
        checkCuda(cudaSetDevice(old_device_id));
    }
#endif
}

Scene::~Scene() {
    if (!use_gpu) {
        rtcReleaseScene(embree_scene);
        rtcReleaseDevice(embree_device);
        delete envmap;
    } else {
#ifdef __NVCC__
        int old_device_id = -1;
        checkCuda(cudaGetDevice(&old_device_id));
        if (gpu_index != -1) {
            checkCuda(cudaSetDevice(gpu_index));
        }
        checkCuda(cudaFree(envmap));
        checkCuda(cudaSetDevice(old_device_id));
#else
        assert(false);
#endif
    }

}

DScene::DScene(const DCamera &camera,
               const std::vector<DShape*> &shapes,
               const std::vector<DMaterial*> &materials,
               const std::vector<DAreaLight*> &area_lights,
               const std::shared_ptr<DEnvironmentMap> &envmap,
               bool use_gpu,
               int gpu_index) : use_gpu(use_gpu), gpu_index(gpu_index) {
#ifdef __NVCC__
    int old_device_id = -1;
#endif
    if (use_gpu) {
#ifdef __NVCC__
        checkCuda(cudaGetDevice(&old_device_id));
        if (gpu_index != -1) {
            checkCuda(cudaSetDevice(gpu_index));
        }
#endif
        cuda_synchronize();
    }
    // Flatten the scene into array
    this->camera = camera;
    if (shapes.size() > 0) {
        this->shapes = Buffer<DShape>(use_gpu, shapes.size());
        for (int shape_id = 0; shape_id < (int)shapes.size(); shape_id++) {
            this->shapes[shape_id] = *shapes[shape_id];
        }
    }
    if (materials.size() > 0) {
        this->materials = Buffer<DMaterial>(use_gpu, materials.size());
        for (int material_id = 0; material_id < (int)materials.size(); material_id++) {
            this->materials[material_id] = *materials[material_id];
            
        }
    }
    if (area_lights.size() > 0) {
        this->area_lights = Buffer<DAreaLight>(use_gpu, area_lights.size());
        for (int light_id = 0; light_id < (int)area_lights.size(); light_id++) {
            this->area_lights[light_id] = *area_lights[light_id];
        }
    }
    if (envmap.get() != nullptr) {
        if (use_gpu) {
#ifdef __NVCC__
            checkCuda(cudaMallocManaged(&this->envmap, sizeof(DEnvironmentMap)));
#else
            assert(false);
#endif
        } else {
            this->envmap = new DEnvironmentMap;
        }
        *(this->envmap) = *envmap;
    } else {
        this->envmap = nullptr;
    }
#ifdef __NVCC__
    if (old_device_id != -1) {
        checkCuda(cudaSetDevice(old_device_id));
    }
#endif

}

DScene::~DScene() {
    if (!use_gpu) {
        delete envmap;
    } else {
#ifdef __NVCC__
        int old_device_id = -1;
        checkCuda(cudaGetDevice(&old_device_id));
        if (gpu_index != -1) {
            checkCuda(cudaSetDevice(gpu_index));
        }
        checkCuda(cudaFree(envmap));
        checkCuda(cudaSetDevice(old_device_id));
#else
        assert(false);
#endif
    }
}

FlattenScene get_flatten_scene(const Scene &scene) {
    return FlattenScene{scene.shapes.data,
                        scene.materials.data,
                        scene.area_lights.data,
                        scene.envmap != nullptr ?
                            (int)scene.area_lights.size() + 1 :
                            (int)scene.area_lights.size(),
                        scene.light_pmf.data,
                        scene.light_cdf.data,
                        scene.light_areas.data,
                        scene.area_cdfs.data,
                        scene.envmap,
                        scene.max_generic_texture_dimension};
}

#ifdef __NVCC__
__global__ void to_optix_ray_kernel(
        int N, const int *active_pixels, const Ray *in, OptiXRay *out) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) {
        return;
    }
    out[idx] = OptiXRay(in[active_pixels[idx]]);
}

void to_optix_ray(const BufferView<int> &active_pixels,
                  const BufferView<Ray> &rays,
                  BufferView<OptiXRay> optix_rays) {
    auto block_size = 256;
    auto block_count = idiv_ceil(active_pixels.size(), block_size);
    to_optix_ray_kernel<<<block_count, block_size>>>(
        active_pixels.size(), active_pixels.begin(),
        rays.begin(), optix_rays.begin());
}

__global__ void intersect_shape_kernel(
        int N,
        const Shape *shapes,
        const int *active_pixels,
        const OptiXHit *hits,
        Ray *rays,
        const RayDifferential *ray_differentials,
        Intersection *out_isects,
        SurfacePoint *out_points,
        RayDifferential *new_ray_differentials) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) {
        return;
    }
    auto pixel_id = active_pixels[idx];

    if (hits[idx].t >= 0.f && length_squared(rays[pixel_id].dir) > 1e-3f) {
        auto shape_id = hits[idx].inst_id;
        auto tri_id = hits[idx].tri_id;
        out_isects[pixel_id].shape_id = shape_id;
        out_isects[pixel_id].tri_id = tri_id;
        const auto &shape = shapes[shape_id];
        const auto &ray = rays[pixel_id];
        const auto &ray_differential = ray_differentials[pixel_id];
        out_points[pixel_id] =
            intersect_shape(shape, tri_id, ray, ray_differential,
                new_ray_differentials[pixel_id]);
        rays[pixel_id].tmax = hits[idx].t;
    } else {
        out_isects[pixel_id].shape_id = -1;
        out_isects[pixel_id].tri_id = -1;
        new_ray_differentials[pixel_id] = ray_differentials[pixel_id];
    }
}

void intersect_shape(const Shape *shapes,
                     const BufferView<int> &active_pixels,
                     const BufferView<OptiXHit> &optix_hits,
                     BufferView<Ray> rays,
                     const BufferView<RayDifferential> &ray_differentials,
                     BufferView<Intersection> isects,
                     BufferView<SurfacePoint> points,
                     BufferView<RayDifferential> new_ray_differentials) {
    auto block_size = 64;
    auto block_count = idiv_ceil(active_pixels.size(), block_size);
    intersect_shape_kernel<<<block_count, block_size>>>(
        active_pixels.size(),
        shapes,
        active_pixels.begin(),
        optix_hits.begin(),
        rays.begin(),
        ray_differentials.begin(),
        isects.begin(),
        points.begin(),
        new_ray_differentials.begin());
}
#endif

void intersect(const Scene &scene,
               const BufferView<int> &active_pixels,
               BufferView<Ray> rays,
               const BufferView<RayDifferential> &ray_differentials,
               BufferView<Intersection> intersections,
               BufferView<SurfacePoint> points,
               BufferView<RayDifferential> new_ray_differentials,
               BufferView<OptiXRay> optix_rays,
               BufferView<OptiXHit> optix_hits) {
    if (active_pixels.size() == 0) {
        return;
    }
    if (scene.use_gpu) {
#ifdef __NVCC__
        // OptiX prime query
        // Convert the rays to OptiX format
        to_optix_ray(active_pixels, rays,
                     optix_rays);
        optix::prime::Query query =
            scene.optix_scene->createQuery(RTP_QUERY_TYPE_CLOSEST);
        query->setRays(active_pixels.size(),
                       RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX,
                       RTP_BUFFER_TYPE_CUDA_LINEAR,
                       optix_rays.data);
        query->setHits(active_pixels.size(),
                       RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID,
                       RTP_BUFFER_TYPE_CUDA_LINEAR,
                       optix_hits.data);
        // XXX: should use watertight intersection here?
        query->execute(0);
        intersect_shape(scene.shapes.data,
                        active_pixels,
                        optix_hits,
                        rays,
                        ray_differentials,
                        intersections,
                        points,
                        new_ray_differentials);
#else
        assert(false);
#endif
    } else {
        // Embree query
        auto work_per_thread = 256;
        auto num_threads = idiv_ceil(active_pixels.size(), work_per_thread);
        parallel_for_host([&](int thread_index) {
            auto id_offset = work_per_thread * thread_index;
            auto work_end = std::min(id_offset + work_per_thread,
                                     active_pixels.size());
            for (int work_id = id_offset; work_id < work_end; work_id++) {
                auto id = work_id;
                auto pixel_id = active_pixels[id];
                Ray &ray = rays[pixel_id];
                RTCIntersectContext rtc_context;
                rtcInitIntersectContext(&rtc_context);
                RTCRayHit rtc_ray_hit;
                rtc_ray_hit.ray.org_x = (float)ray.org[0];
                rtc_ray_hit.ray.org_y = (float)ray.org[1];
                rtc_ray_hit.ray.org_z = (float)ray.org[2];
                rtc_ray_hit.ray.dir_x = (float)ray.dir[0];
                rtc_ray_hit.ray.dir_y = (float)ray.dir[1];
                rtc_ray_hit.ray.dir_z = (float)ray.dir[2];
                rtc_ray_hit.ray.tnear = (float)ray.tmin;
                rtc_ray_hit.ray.tfar = (float)ray.tmax;
                rtc_ray_hit.ray.mask = (unsigned int)(-1);
                rtc_ray_hit.ray.time = 0.f;
                rtc_ray_hit.ray.flags = 0;
                rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
                rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
                // TODO: switch to rtcIntersect16
                rtcIntersect1(scene.embree_scene, &rtc_context, &rtc_ray_hit);
                if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID ||
                         length_squared(ray.dir) <= 1e-3f) {
                    intersections[pixel_id] = Intersection{-1, -1};
                    new_ray_differentials[pixel_id] = ray_differentials[pixel_id];
                } else {
                    auto shape_id = (int)rtc_ray_hit.hit.geomID;
                    auto tri_id = (int)rtc_ray_hit.hit.primID;
                    intersections[pixel_id] =
                        Intersection{shape_id, tri_id};
                    const auto &shape = scene.shapes[shape_id];
                    const auto &ray_differential = ray_differentials[pixel_id];
                    points[pixel_id] =
                        intersect_shape(shape,
                                        tri_id,
                                        ray,
                                        ray_differential,
                                        new_ray_differentials[pixel_id]);
                    ray.tmax = rtc_ray_hit.ray.tfar;
                }
            }
        }, num_threads);
    }
}

#ifdef __NVCC__
__global__ void update_occluded_rays_kernel(int N,
                                            const int *active_pixels,
                                            const OptiXHit *optix_hits,
                                            Ray *rays) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) {
        return;
    }

    if (optix_hits[idx].t >= 0.f) {
        // Invalidate ray if occluded
        auto pixel_id = active_pixels[idx];
        rays[pixel_id].tmax = -1;
    }
}

void update_occluded_rays(const BufferView<int> &active_pixels,
                          const BufferView<OptiXHit> &optix_hits,
                          BufferView<Ray> rays) {
    auto block_size = 256;
    auto block_count = idiv_ceil(active_pixels.size(), block_size);
    update_occluded_rays_kernel<<<block_count, block_size>>>(
        active_pixels.size(),
        active_pixels.begin(),
        optix_hits.begin(),
        rays.begin());
}
#endif

void occluded(const Scene &scene,
              const BufferView<int> &active_pixels,
              BufferView<Ray> rays,
              BufferView<OptiXRay> optix_rays,
              BufferView<OptiXHit> optix_hits) {
    if (scene.use_gpu) {
#ifdef __NVCC__
        // OptiX prime query
        // Convert the rays to OptiX format
        to_optix_ray(active_pixels, rays, optix_rays);
        optix::prime::Query query =
            scene.optix_scene->createQuery(RTP_QUERY_TYPE_ANY);
        query->setRays(active_pixels.size(),
                       RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX,
                       RTP_BUFFER_TYPE_CUDA_LINEAR,
                       optix_rays.data);
        query->setHits(active_pixels.size(),
                       RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID,
                       RTP_BUFFER_TYPE_CUDA_LINEAR,
                       optix_hits.data);
        // XXX: should use watertight intersection here?
        query->execute(0);
        update_occluded_rays(active_pixels, optix_hits, rays);
#else
        assert(false);
#endif
    } else {
        // Embree query
        auto work_per_thread = 256;
        auto num_threads = idiv_ceil(active_pixels.size(), work_per_thread);
        parallel_for_host([&](int thread_index) {
            auto id_offset = work_per_thread * thread_index;
            auto work_end = std::min(id_offset + work_per_thread,
                                     active_pixels.size());
            for (int work_id = id_offset; work_id < work_end; work_id++) {
                auto id = work_id;
                auto pixel_id = active_pixels[id];
                const Ray &ray = rays[pixel_id];
                RTCIntersectContext rtc_context;
                rtcInitIntersectContext(&rtc_context);
                RTCRay rtc_ray;
                rtc_ray.org_x = (float)ray.org[0];
                rtc_ray.org_y = (float)ray.org[1];
                rtc_ray.org_z = (float)ray.org[2];
                rtc_ray.dir_x = (float)ray.dir[0];
                rtc_ray.dir_y = (float)ray.dir[1];
                rtc_ray.dir_z = (float)ray.dir[2];
                rtc_ray.tnear = (float)ray.tmin;
                rtc_ray.tfar = (float)ray.tmax;
                rtc_ray.mask = (unsigned int)(-1);
                rtc_ray.time = 0.f;
                rtc_ray.flags = 0;
                // TODO: switch to rtcOccluded16
                rtcOccluded1(scene.embree_scene, &rtc_context, &rtc_ray);
                if (rtc_ray.tfar < 0) {
                    // intersections[pixel_id] = Intersection{-1, -1};
                    rays[pixel_id].tmax = -1;
                }
            }
        }, num_threads); 
    }
}

struct light_point_sampler {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        // Select light source by binary search on light_cdf
        auto sample = samples[pixel_id];
        const Real *light_ptr =
            thrust::upper_bound(thrust::seq,
                scene.light_cdf, scene.light_cdf + scene.num_lights,
                sample.light_sel);
        auto light_id = clamp((int)(light_ptr - scene.light_cdf - 1),
                                    0, scene.num_lights - 1);
        if (scene.envmap != nullptr && light_id == scene.num_lights - 1) {
            // Environment map
            light_isects[pixel_id].shape_id = -1;
            light_isects[pixel_id].tri_id = -1;
            light_points[pixel_id] = SurfacePoint::zero();
            shadow_rays[pixel_id].org = shading_points[pixel_id].position;
            shadow_rays[pixel_id].dir = envmap_sample(*(scene.envmap), sample.uv);
            shadow_rays[pixel_id].tmin = 1e-3f;
            shadow_rays[pixel_id].tmax = infinity<Real>();
        } else {
            // Area light
            const auto &light = scene.area_lights[light_id];
            const auto &shape = scene.shapes[light.shape_id];
            // Select triangle by binary search on area_cdfs
            const Real *area_cdf = scene.area_cdfs[light_id];
            const Real *tri_ptr = thrust::upper_bound(thrust::seq,
                    area_cdf, area_cdf + shape.num_triangles, sample.tri_sel);
            auto tri_id = clamp((int)(tri_ptr - area_cdf - 1), 0, shape.num_triangles - 1);
            light_isects[pixel_id].shape_id = light.shape_id;
            light_isects[pixel_id].tri_id = tri_id;
            light_points[pixel_id] = sample_shape(shape, tri_id, sample.uv);
            shadow_rays[pixel_id].org = shading_points[pixel_id].position;
            shadow_rays[pixel_id].dir = normalize(
                light_points[pixel_id].position - shading_points[pixel_id].position);
            // Shadow epislon. Sorry.
            shadow_rays[pixel_id].tmin = 1e-3f;
            shadow_rays[pixel_id].tmax = (1 - 1e-3f) *
                length(light_points[pixel_id].position - shading_points[pixel_id].position);
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const SurfacePoint *shading_points;
    const LightSample *samples;
    Intersection *light_isects;
    SurfacePoint *light_points;
    Ray *shadow_rays;
};

void sample_point_on_light(const Scene &scene,
                           const BufferView<int> &active_pixels,
                           const BufferView<SurfacePoint> &shading_points,
                           const BufferView<LightSample> &samples,
                           BufferView<Intersection> light_isects,
                           BufferView<SurfacePoint> light_points,
                           BufferView<Ray> shadow_ray) {
    parallel_for(light_point_sampler{
        get_flatten_scene(scene),
        active_pixels.begin(),
        shading_points.begin(),
        samples.begin(),
        light_isects.begin(),
        light_points.begin(),
        shadow_ray.begin()},
        active_pixels.size(), scene.use_gpu);
}

void test_scene_intersect(bool use_gpu) {
    Buffer<Vector3f> vertices(use_gpu, 3);
    vertices[0] = Vector3f{-1.f, 0.f, 1.f};
    vertices[1] = Vector3f{ 1.f, 0.f, 1.f};
    vertices[2] = Vector3f{ 0.f, 1.f, 1.f};
    Buffer<Vector3i> indices(use_gpu, 1);
    indices[0] = Vector3i{0, 1, 2};

    Ray ray0{Vector3{0, 0, 0}, Vector3{0, 0, 1}};
    Ray ray1{Vector3{0, 0, 0}, Vector3{0, 0, -1}};
    RayDifferential ray_diff0{
        Vector3{0, 0, 0}, Vector3{0, 0, 0},
        Vector3{0, 0, 0}, Vector3{0, 0, 0}};
    RayDifferential ray_diff1{
        Vector3{0, 0, 0}, Vector3{0, 0, 0},
        Vector3{0, 0, 0}, Vector3{0, 0, 0}};
    Buffer<Ray> rays(use_gpu, 2);
    rays[0] = ray0;
    rays[1] = ray1;
    Buffer<RayDifferential> ray_diffs(use_gpu, 2);
    ray_diffs[0] = ray_diff0;
    ray_diffs[1] = ray_diff1;
    Shape triangle{(float*)vertices.data,
                   (int*)indices.data,
                   nullptr, // uvs
                   nullptr, // normal
                   nullptr, // uv_indices
                   nullptr, // normal_indices
                   nullptr, // colors
                   3, // num_vertices
                   0, // num_uv_vertices
                   0, // num_normal_vertices
                   1, // num_triangles
                   0,
                   -1};
    auto pos = Vector3f{0, 0, 0};
    auto look = Vector3f{0, 0, 1};
    auto up = Vector3f{0, 1, 0};
    Matrix3x3f n2c = Matrix3x3f::identity();
    Matrix3x3f c2n = Matrix3x3f::identity();
    Camera camera{1, 1,
        &pos[0],
        &look[0],
        &up[0],
        nullptr, // cam_to_world
        nullptr, // world_to_cam
        &n2c.data[0][0],
        &c2n.data[0][0],
        nullptr, // distortion_params
        1e-2f,
        CameraType::Perspective,
        Vector2i{0, 0},
        Vector2i{1, 1}};
    Scene scene{camera, {&triangle}, {}, {}, {}, use_gpu, 0, false, false};
    parallel_init();

    Buffer<int> active_pixels(use_gpu, 2);
    active_pixels[0] = 0;
    active_pixels[1] = 1;
    Buffer<Intersection> isects(use_gpu, 2);
    Buffer<SurfacePoint> surface_points(use_gpu, 2);
    Buffer<OptiXRay> optix_rays(use_gpu, 2);
    Buffer<OptiXHit> optix_hits(use_gpu, 2);
    intersect(scene,
              active_pixels.view(0, active_pixels.size()), 
              rays.view(0, rays.size()),
              ray_diffs.view(0, rays.size()),
              isects.view(0, rays.size()),
              surface_points.view(0, rays.size()),
              ray_diffs.view(0, rays.size()),
              optix_rays.view(0, rays.size()),
              optix_hits.view(0, rays.size()));
    cuda_synchronize();
    equal_or_error(__FILE__, __LINE__, isects[0].shape_id, 0);
    equal_or_error(__FILE__, __LINE__, isects[0].tri_id, 0);
    equal_or_error(__FILE__, __LINE__, isects[1].shape_id, -1);
    equal_or_error(__FILE__, __LINE__, isects[1].tri_id, -1);
    equal_or_error<Real>(__FILE__, __LINE__, surface_points[0].position, Vector3{0, 0, 1});
    equal_or_error<Real>(__FILE__, __LINE__, ray_diffs[0].org_dx, Vector3{0, 0, 0});
    equal_or_error<Real>(__FILE__, __LINE__, ray_diffs[0].org_dy, Vector3{0, 0, 0});
    equal_or_error<Real>(__FILE__, __LINE__, ray_diffs[0].dir_dx, Vector3{0, 0, 0});
    equal_or_error<Real>(__FILE__, __LINE__, ray_diffs[0].dir_dy, Vector3{0, 0, 0});
    equal_or_error<Real>(__FILE__, __LINE__, ray_diffs[1].org_dx, Vector3{0, 0, 0});
    equal_or_error<Real>(__FILE__, __LINE__, ray_diffs[1].org_dy, Vector3{0, 0, 0});
    equal_or_error<Real>(__FILE__, __LINE__, ray_diffs[1].dir_dx, Vector3{0, 0, 0});
    equal_or_error<Real>(__FILE__, __LINE__, ray_diffs[1].dir_dy, Vector3{0, 0, 0});
    parallel_cleanup();
}

void test_sample_point_on_light(bool use_gpu) {
    // Two light sources, one with two triangles with area 1 and 0.5
    // one with one triangle with area 1
    Buffer<Vector3f> vertices0(use_gpu, 6);
    vertices0[0] = Vector3f{-1.f, 0.f, 1.f};
    vertices0[1] = Vector3f{ 1.f, 0.f, 1.f};
    vertices0[2] = Vector3f{ 0.f, 1.f, 1.f};
    vertices0[3] = Vector3f{-1.f, 0.f, 2.f};
    vertices0[4] = Vector3f{ 1.f, 0.f, 2.f};
    vertices0[5] = Vector3f{ 0.f, 0.5f, 2.f};
    Buffer<Vector3i> indices0(use_gpu, 2);
    indices0[0] = Vector3i{0, 1, 2};
    indices0[1] = Vector3i{3, 4, 5};
    Buffer<Vector3f> vertices1(use_gpu, 3);
    vertices1[0] = Vector3f{-1.f, 0.f, 0.f};
    vertices1[1] = Vector3f{ 1.f, 0.f, 0.f};
    vertices1[2] = Vector3f{ 0.f, 1.f, 0.f};
    Buffer<Vector3i> indices1(use_gpu, 1);
    indices1[0] = Vector3i{0, 1, 2};
    Buffer<LightSample> samples(use_gpu, 3);
    samples[0] = LightSample{0.25f, 0.5f, Vector2{0.f, 0.f}};
    samples[1] = LightSample{0.25f, 0.75f, Vector2{0.f, 0.f}};
    samples[2] = LightSample{0.5f, 0.5f, Vector2{0.f, 0.f}};
    Shape shape0{(float*)vertices0.data,
                 (int*)indices0.data,
                 nullptr, // uvs
                 nullptr, // normals
                 nullptr, // uv_indices
                 nullptr, // normal_indices
                 nullptr, // colors
                 6, // num_vertices
                 0, // num_uv_vertices
                 0, // num_normal_vertices
                 2, // num_triangles
                 0,
                 0};
    Shape shape1{(float*)vertices1.data,
                 (int*)indices1.data,
                 nullptr, // uvs
                 nullptr, // normals
                 nullptr, // uv_indices
                 nullptr, // normal_indices
                 nullptr, // colors
                 3, // num_vertices
                 0, // num_uv_vertices
                 0, // num_normal_vertices
                 1, // num_triangles
                 0,
                 0};
    AreaLight light0{0,
                     Vector3f{1.f, 1.f, 1.f},
                     false, // two_sided
                     true}; // directly_visible
    AreaLight light1{1,
                     Vector3f{2.f, 2.f, 2.f},
                     false, // two_sided
                     true}; // directly_visible

    auto shapes = std::make_shared<std::vector<const Shape *>>(
        std::vector<const Shape*>{&shape0, &shape1});
    auto materials = std::make_shared<std::vector<const Material *>>();
    auto lights = std::make_shared<std::vector<const AreaLight *>>(
        std::vector<const AreaLight*>{&light0, &light1});

    auto pos = Vector3f{0, 0, 0};
    auto look = Vector3f{0, 0, 1};
    auto up = Vector3f{0, 1, 0};
    Matrix3x3f n2c = Matrix3x3f::identity();
    Matrix3x3f c2n = Matrix3x3f::identity();
    Camera camera{1, 1,
        &pos[0],
        &look[0],
        &up[0],
        nullptr, // cam_to_world
        nullptr, // world_to_cam
        &n2c.data[0][0],
        &c2n.data[0][0],
        nullptr, // distortion_params
        1e-2f,
        CameraType::Perspective,
        Vector2i{0, 0},
        Vector2i{1, 1}};
    Scene scene{camera, {&shape0, &shape1}, {}, {&light0, &light1}, {}, use_gpu, 0, false, false};
    cuda_synchronize();
    // Power of the first light source: 1.5
    // Power of the second light source: 2
    equal_or_error(__FILE__, __LINE__, scene.light_pmf[0], Real(1.5 / (1.5 + 2)));
    equal_or_error(__FILE__, __LINE__, scene.light_pmf[1], Real(2 / (1.5 + 2)));
    equal_or_error(__FILE__, __LINE__, scene.light_cdf[0], Real(0));
    equal_or_error(__FILE__, __LINE__, scene.light_cdf[1], Real(1.5 / (1.5 + 2)));
    equal_or_error(__FILE__, __LINE__, scene.area_cdfs[0][0], Real(0));
    equal_or_error(__FILE__, __LINE__, scene.area_cdfs[0][1], Real(1.0 / 1.5));
    equal_or_error(__FILE__, __LINE__, scene.area_cdfs[1][0], Real(0));

    Buffer<int> active_pixels(use_gpu, samples.size());
    Buffer<SurfacePoint> shading_points(use_gpu, samples.size());
    Buffer<Intersection> light_isects(use_gpu, samples.size());
    Buffer<SurfacePoint> light_points(use_gpu, samples.size());
    Buffer<Ray> shadow_rays(use_gpu, samples.size());
    for (int i = 0; i < 3; i++) {
        active_pixels[i] = i;
    }
    sample_point_on_light(scene,
                          active_pixels.view(0, active_pixels.size()),
                          shading_points.view(0, samples.size()),
                          samples.view(0, samples.size()),
                          light_isects.view(0, light_isects.size()),
                          light_points.view(0, light_points.size()),
                          shadow_rays.view(0, shadow_rays.size()));
    cuda_synchronize();
    equal_or_error(__FILE__, __LINE__, light_isects[0].shape_id, 0);
    equal_or_error(__FILE__, __LINE__, light_isects[0].tri_id, 0);
    equal_or_error(__FILE__, __LINE__, light_isects[1].shape_id, 0);
    equal_or_error(__FILE__, __LINE__, light_isects[1].tri_id, 1);
    equal_or_error(__FILE__, __LINE__, light_isects[2].shape_id, 1);
    equal_or_error(__FILE__, __LINE__, light_isects[2].tri_id, 0);
    equal_or_error<Real>(__FILE__, __LINE__, light_points[0].position,
                                       Vector3{1.f, 0.f, 1.f});
    equal_or_error<Real>(__FILE__, __LINE__, light_points[1].position,
                                       Vector3{1.f, 0.f, 2.f});
    equal_or_error<Real>(__FILE__, __LINE__, light_points[2].position,
                                       Vector3{1.f, 0.f, 0.f});
}
