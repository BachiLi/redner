#include "pathtracer.h"
#include "scene.h"
#include "pcg_sampler.h"
#include "sobol_sampler.h"
#include "parallel.h"
#include "scene.h"
#include "buffer.h"
#include "camera.h"
#include "intersection.h"
#include "active_pixels.h"
#include "shape.h"
#include "material.h"
#include "area_light.h"
#include "envmap.h"
#include "test_utils.h"
#include "cuda_utils.h"
#include "thrust_utils.h"
#include "primary_intersection.h"
#include "primary_contribution.h"
#include "bsdf_sample.h"
#include "path_contribution.h"

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>

void init_paths(BufferView<Vector3> throughputs,
                BufferView<Real> min_roughness,
                bool use_gpu) {
    DISPATCH(use_gpu, thrust::fill, throughputs.begin(), throughputs.end(), Vector3{1, 1, 1});
    DISPATCH(use_gpu, thrust::fill, min_roughness.begin(), min_roughness.end(), Real(0));
}

struct PathBuffer {
    PathBuffer(int max_bounces,
               int num_pixels,
               bool use_gpu,
               const ChannelInfo &channel_info) :
            num_pixels(num_pixels) {
        assert(max_bounces >= 0);
        // For forward path tracing, we need to allocate memory for
        // all bounces
        // For edge sampling, we need to allocate memory for
        // 2 * num_pixels paths (and 4 * num_pixels for those
        //  shared between two path vertices).
        camera_samples = Buffer<CameraSample>(use_gpu, num_pixels);
        light_samples = Buffer<LightSample>(use_gpu, max_bounces * num_pixels);
        edge_light_samples = Buffer<LightSample>(use_gpu, 2 * num_pixels);
        bsdf_samples = Buffer<BSDFSample>(use_gpu, max_bounces * num_pixels);
        edge_bsdf_samples = Buffer<BSDFSample>(use_gpu, 2 * num_pixels);
        rays = Buffer<Ray>(use_gpu, (max_bounces + 1) * num_pixels);
        nee_rays = Buffer<Ray>(use_gpu, max_bounces * num_pixels);
        primary_ray_differentials = Buffer<RayDifferential>(use_gpu, num_pixels);
        ray_differentials = Buffer<RayDifferential>(use_gpu, (max_bounces + 1) * num_pixels);
        bsdf_ray_differentials = Buffer<RayDifferential>(use_gpu, max_bounces * num_pixels);
        edge_rays = Buffer<Ray>(use_gpu, 4 * num_pixels);
        edge_nee_rays = Buffer<Ray>(use_gpu, 2 * num_pixels);
        edge_ray_differentials = Buffer<RayDifferential>(use_gpu, 2 * num_pixels);
        primary_active_pixels = Buffer<int>(use_gpu, num_pixels);
        active_pixels = Buffer<int>(use_gpu, (max_bounces + 1) * num_pixels);
        edge_active_pixels = Buffer<int>(use_gpu, 4 * num_pixels);
        shading_isects = Buffer<Intersection>(use_gpu, (max_bounces + 1) * num_pixels);
        edge_shading_isects = Buffer<Intersection>(use_gpu, 4 * num_pixels);
        shading_points = Buffer<SurfacePoint>(use_gpu, (max_bounces + 1) * num_pixels);
        edge_shading_points = Buffer<SurfacePoint>(use_gpu, 4 * num_pixels);
        light_isects = Buffer<Intersection>(use_gpu, max_bounces * num_pixels);
        edge_light_isects = Buffer<Intersection>(use_gpu, 2 * num_pixels);
        light_points = Buffer<SurfacePoint>(use_gpu, max_bounces * num_pixels);
        edge_light_points = Buffer<SurfacePoint>(use_gpu, 2 * num_pixels);
        throughputs = Buffer<Vector3>(use_gpu, (max_bounces + 1) * num_pixels);
        edge_throughputs = Buffer<Vector3>(use_gpu, 4 * num_pixels);
        channel_multipliers = Buffer<Real>(use_gpu,
            2 * channel_info.num_total_dimensions * num_pixels);
        min_roughness = Buffer<Real>(use_gpu, (max_bounces + 1) * num_pixels);
        edge_min_roughness = Buffer<Real>(use_gpu, 4 * num_pixels);

        // OptiX buffers
        optix_rays = Buffer<OptiXRay>(use_gpu, 2 * num_pixels);
        optix_hits = Buffer<OptiXHit>(use_gpu, 2 * num_pixels);

        // Derivatives buffers
        d_next_throughputs = Buffer<Vector3>(use_gpu, num_pixels);
        d_next_rays = Buffer<DRay>(use_gpu, num_pixels);
        d_next_ray_differentials = Buffer<RayDifferential>(use_gpu, num_pixels);
        d_next_points = Buffer<SurfacePoint>(use_gpu, num_pixels);
        d_throughputs = Buffer<Vector3>(use_gpu, num_pixels);
        d_rays = Buffer<DRay>(use_gpu, num_pixels);
        d_ray_differentials = Buffer<RayDifferential>(use_gpu, num_pixels);
        d_points = Buffer<SurfacePoint>(use_gpu, num_pixels);

        d_general_vertices = Buffer<DVertex>(use_gpu, 3 * num_pixels);
        d_light_vertices = Buffer<DVertex>(use_gpu, 3 * num_pixels);
        d_bsdf_vertices = Buffer<DVertex>(use_gpu, 3 * num_pixels);
        d_diffuse_texs = Buffer<DTexture3>(use_gpu, num_pixels);
        d_specular_texs = Buffer<DTexture3>(use_gpu, num_pixels);
        d_roughness_texs = Buffer<DTexture1>(use_gpu, num_pixels);
        d_direct_lights = Buffer<DAreaLightInst>(use_gpu, num_pixels);
        d_nee_lights = Buffer<DAreaLightInst>(use_gpu, num_pixels);
        d_bsdf_lights = Buffer<DAreaLightInst>(use_gpu, num_pixels);
        d_envmap_vals = Buffer<DTexture3>(use_gpu, num_pixels);
        d_world_to_envs = Buffer<Matrix4x4>(use_gpu, num_pixels);

        d_vertex_reduce_buffer = Buffer<DVertex>(use_gpu, 3 * num_pixels);
        d_tex3_reduce_buffer = Buffer<DTexture3>(use_gpu, num_pixels);
        d_tex1_reduce_buffer = Buffer<DTexture1>(use_gpu, num_pixels);
        d_lgt_reduce_buffer = Buffer<DAreaLightInst>(use_gpu, num_pixels);
        d_envmap_reduce_buffer = Buffer<DTexture3>(use_gpu, num_pixels);

        d_cameras = Buffer<DCameraInst>(use_gpu, num_pixels);

        primary_edge_samples = Buffer<PrimaryEdgeSample>(use_gpu, num_pixels);
        secondary_edge_samples = Buffer<SecondaryEdgeSample>(use_gpu, num_pixels);
        primary_edge_records = Buffer<PrimaryEdgeRecord>(use_gpu, num_pixels);
        secondary_edge_records = Buffer<SecondaryEdgeRecord>(use_gpu, num_pixels);
        edge_contribs = Buffer<Real>(use_gpu, 2 * num_pixels);
        edge_surface_points = Buffer<Vector3>(use_gpu, 2 * num_pixels);

        tmp_light_samples = Buffer<LightSample>(use_gpu, num_pixels);
        tmp_bsdf_samples = Buffer<BSDFSample>(use_gpu, num_pixels);
    }

    int num_pixels;
    Buffer<CameraSample> camera_samples;
    Buffer<LightSample> light_samples, edge_light_samples;
    Buffer<BSDFSample> bsdf_samples, edge_bsdf_samples;
    Buffer<Ray> rays, nee_rays;
    Buffer<Ray> edge_rays, edge_nee_rays;
    Buffer<RayDifferential> primary_ray_differentials;
    Buffer<RayDifferential> ray_differentials, bsdf_ray_differentials;
    Buffer<RayDifferential> edge_ray_differentials;
    Buffer<int> primary_active_pixels, active_pixels, edge_active_pixels;
    Buffer<Intersection> shading_isects, edge_shading_isects;
    Buffer<SurfacePoint> shading_points, edge_shading_points;
    Buffer<Intersection> light_isects, edge_light_isects;
    Buffer<SurfacePoint> light_points, edge_light_points;
    Buffer<Vector3> throughputs, edge_throughputs;
    Buffer<Real> channel_multipliers;
    Buffer<Real> min_roughness, edge_min_roughness;

    // OptiX related
    Buffer<OptiXRay> optix_rays;
    Buffer<OptiXHit> optix_hits;

    // Derivatives related
    Buffer<Vector3> d_next_throughputs;
    Buffer<DRay> d_next_rays;
    Buffer<RayDifferential> d_next_ray_differentials;
    Buffer<SurfacePoint> d_next_points;
    Buffer<Vector3> d_throughputs;
    Buffer<DRay> d_rays;
    Buffer<RayDifferential> d_ray_differentials;
    Buffer<SurfacePoint> d_points;

    Buffer<DVertex> d_general_vertices;
    Buffer<DVertex> d_light_vertices;
    Buffer<DVertex> d_bsdf_vertices;
    Buffer<DTexture3> d_diffuse_texs;
    Buffer<DTexture3> d_specular_texs;
    Buffer<DTexture1> d_roughness_texs;
    Buffer<DAreaLightInst> d_direct_lights;
    Buffer<DAreaLightInst> d_nee_lights;
    Buffer<DAreaLightInst> d_bsdf_lights;
    Buffer<DTexture3> d_envmap_vals;
    Buffer<Matrix4x4> d_world_to_envs;
    Buffer<DVertex> d_vertex_reduce_buffer;
    Buffer<DTexture3> d_tex3_reduce_buffer;
    Buffer<DTexture1> d_tex1_reduce_buffer;
    Buffer<DAreaLightInst> d_lgt_reduce_buffer;
    Buffer<DTexture3> d_envmap_reduce_buffer;

    Buffer<DCameraInst> d_cameras;

    // Edge sampling related
    Buffer<PrimaryEdgeSample> primary_edge_samples;
    Buffer<SecondaryEdgeSample> secondary_edge_samples;
    Buffer<PrimaryEdgeRecord> primary_edge_records;
    Buffer<SecondaryEdgeRecord> secondary_edge_records;
    Buffer<Real> edge_contribs;
    Buffer<Vector3> edge_surface_points;
    // For sharing RNG between pixels
    Buffer<LightSample> tmp_light_samples;
    Buffer<BSDFSample> tmp_bsdf_samples;
};

void accumulate_vertex(BufferView<DVertex> d_vertices,
                       BufferView<DVertex> reduce_buffer,
                       BufferView<DShape> d_shapes,
                       bool use_gpu,
                       ThrustCachedAllocator &thrust_alloc) {
    if (d_vertices.size() == 0) {
        return;
    }
    // Reduce into unique sequence
    auto beg = d_vertices.begin();
    auto end = d_vertices.end();
    auto buffer_beg = reduce_buffer.begin();
    auto buffer_end = DISPATCH_CACHED(use_gpu, thrust_alloc,
        thrust::remove_copy, beg, end, buffer_beg, DVertex{-1, -1});
    DISPATCH_CACHED(use_gpu, thrust_alloc, thrust::sort, buffer_beg, buffer_end);
    auto new_end = DISPATCH_CACHED(use_gpu, thrust_alloc, thrust::reduce_by_key,
        buffer_beg, buffer_end, // input keys
        buffer_beg, // input values
        beg, // output keys
        beg).first; // output values
    d_vertices.count = new_end - beg;
    // Accumulate to output derivatives
    accumulate_vertex(d_vertices, d_shapes, use_gpu);
}

void accumulate_diffuse(const Scene &scene,
                        BufferView<DTexture3> d_diffuse,
                        BufferView<DTexture3> reduce_buffer,
                        BufferView<DMaterial> d_materials,
                        ThrustCachedAllocator &thrust_alloc) {
    if (d_diffuse.size() == 0) {
        return;
    }
    // Reduce into unique sequence
    auto beg = d_diffuse.begin();
    auto end = d_diffuse.end();
    auto buffer_beg = reduce_buffer.begin();
    auto buffer_end = DISPATCH_CACHED(scene.use_gpu, thrust_alloc,
        thrust::remove_copy, beg, end, buffer_beg, DTexture3{-1, -1, -1, -1});
    DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::sort, buffer_beg, buffer_end);
    auto new_end = DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::reduce_by_key,
        buffer_beg, buffer_end, // input keys
        buffer_beg, // input values
        beg, // output keys
        beg).first; // output values
    d_diffuse.count = new_end - beg;
    // Accumulate to output derivatives
    accumulate_diffuse(scene, d_diffuse, d_materials);
}

void accumulate_specular(const Scene &scene,
                         BufferView<DTexture3> d_specular,
                         BufferView<DTexture3> reduce_buffer,
                         BufferView<DMaterial> d_materials,
                         ThrustCachedAllocator &thrust_alloc) {
    if (d_specular.size() == 0) {
        return;
    }
    // Reduce into unique sequence
    auto beg = d_specular.begin();
    auto end = d_specular.end();
    auto buffer_beg = reduce_buffer.begin();
    auto buffer_end = DISPATCH_CACHED(scene.use_gpu, thrust_alloc,
        thrust::remove_copy, beg, end, buffer_beg, DTexture3{-1, -1, -1, -1});
    DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::sort, buffer_beg, buffer_end);
    auto new_end = DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::reduce_by_key,
        buffer_beg, buffer_end, // input keys
        buffer_beg, // input values
        beg, // output keys
        beg).first; // output values
    d_specular.count = new_end - beg;
    // Accumulate to output derivatives
    accumulate_specular(scene, d_specular, d_materials);
}

void accumulate_roughness(const Scene &scene,
                          BufferView<DTexture1> d_roughness,
                          BufferView<DTexture1> reduce_buffer,
                          BufferView<DMaterial> d_materials,
                          ThrustCachedAllocator &thrust_alloc) {
    if (d_roughness.size() == 0) {
        return;
    }
    // Reduce into unique sequence
    auto beg = d_roughness.begin();
    auto end = d_roughness.end();
    auto buffer_beg = reduce_buffer.begin();
    auto buffer_end = DISPATCH_CACHED(scene.use_gpu, thrust_alloc,
        thrust::remove_copy, beg, end, buffer_beg, DTexture1{-1, -1, -1, -1});
    DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::sort, buffer_beg, buffer_end);
    auto new_end = DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::reduce_by_key,
        buffer_beg, buffer_end, // input keys
        buffer_beg, // input values
        beg, // output keys
        beg).first; // output values
    d_roughness.count = new_end - beg;
    // Accumulate to output derivatives
    accumulate_roughness(scene, d_roughness, d_materials);
}

void accumulate_area_light(BufferView<DAreaLightInst> &d_light_insts,
                           BufferView<DAreaLightInst> reduce_buffer,
                           BufferView<DAreaLight> d_lights,
                           bool use_gpu,
                           ThrustCachedAllocator &thrust_alloc) {
    if (d_light_insts.size() == 0) {
        return;
    }
    // Reduce into unique sequence
    auto beg = d_light_insts.begin();
    auto end = d_light_insts.end();
    auto buffer_beg = reduce_buffer.begin();
    auto buffer_end = DISPATCH_CACHED(use_gpu, thrust_alloc,
        thrust::remove_copy, beg, end, buffer_beg, DAreaLightInst{-1});
    DISPATCH_CACHED(use_gpu, thrust_alloc, thrust::sort, buffer_beg, buffer_end);
    auto new_end = DISPATCH_CACHED(use_gpu, thrust_alloc, thrust::reduce_by_key,
        buffer_beg, buffer_end, // input keys
        buffer_beg, // input values
        beg, // output keys
        beg).first; // output values
    d_light_insts.count = new_end - beg;
    // Accumulate to output derivatives
    accumulate_area_light(d_light_insts, d_lights, use_gpu);
}

void accumulate_envmap(const Scene &scene,
                       BufferView<DTexture3> &d_envmap_vals,
                       BufferView<Matrix4x4> &d_world_to_envs,
                       BufferView<DTexture3> reduce_buffer,
                       DEnvironmentMap *d_envmap,
                       ThrustCachedAllocator &thrust_alloc) {
    if (d_envmap_vals.size() == 0) {
        return;
    }
    {
        // Reduce into unique sequence
        auto beg = d_envmap_vals.begin();
        auto end = d_envmap_vals.end();
        auto buffer_beg = reduce_buffer.begin();
        auto buffer_end = DISPATCH_CACHED(scene.use_gpu, thrust_alloc,
            thrust::remove_copy, beg, end, buffer_beg, DTexture3{-1});
        DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::sort, buffer_beg, buffer_end);
        auto new_end = DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::reduce_by_key,
            buffer_beg, buffer_end, // input keys
            buffer_beg, // input values
            beg, // output keys
            beg).first; // output values
        d_envmap_vals.count = new_end - beg;
    }
    auto d_world_to_env = Matrix4x4();
    {
        // Reduce to a single matrix
        auto beg = d_world_to_envs.begin();
        auto end = d_world_to_envs.begin();
        d_world_to_env = DISPATCH_CACHED(scene.use_gpu, thrust_alloc,
            thrust::reduce, beg, end, Matrix4x4());
    }
    // Accumulate to output derivatives
    accumulate_envmap(scene, d_envmap_vals, d_world_to_env, *d_envmap);
}

// 1 2 3 4 5 -> 1 1 2 2 3 3 4 4 5 5
template <typename T>
struct copy_interleave {
    DEVICE void operator()(int idx) {
        to[2 * idx + 0] = from[idx];
        to[2 * idx + 1] = from[idx];
    }

    const T *from;
    T *to;
};

// Extract the position of a surface point
struct get_position {
    DEVICE void operator()(int idx) {
        p[active_pixels[idx]] = sp[active_pixels[idx]].position;
    }

    const int *active_pixels;
    const SurfacePoint *sp;
    Vector3 *p;
};

void render(const Scene &scene,
            const RenderOptions &options,
            ptr<float> rendered_image,
            ptr<float> d_rendered_image,
            std::shared_ptr<DScene> d_scene,
            ptr<float> debug_image) {
#ifdef __NVCC__
    int old_device_id = -1;
    if (scene.use_gpu) {
        checkCuda(cudaGetDevice(&old_device_id));
        if (scene.gpu_index != -1) {
            checkCuda(cudaSetDevice(scene.gpu_index));
        }
    }
#endif
    parallel_init();
    if (d_rendered_image.get() != nullptr) {
        initialize_ltc_table(scene.use_gpu);
    }
    ChannelInfo channel_info(options.channels, scene.use_gpu);

    // Some common variables
    const auto &camera = scene.camera;
    auto num_pixels = camera.width * camera.height;
    auto max_bounces = options.max_bounces;

    // A main difference between our path tracer and the usual path
    // tracer is that we need to store all the intermediate states
    // for later computation of derivatives.
    // Therefore we allocate a big buffer here for the storage.
    PathBuffer path_buffer(max_bounces, num_pixels, scene.use_gpu, channel_info);
    auto num_active_pixels = std::vector<int>((max_bounces + 1) * num_pixels, 0);
    std::unique_ptr<Sampler> sampler, edge_sampler;
    switch (options.sampler_type) {
        case SamplerType::independent: {
            sampler = std::unique_ptr<Sampler>(new PCGSampler(scene.use_gpu, options.seed, num_pixels));
            edge_sampler = std::unique_ptr<Sampler>(
                new PCGSampler(scene.use_gpu, options.seed + 131071U, num_pixels));
            break;
        } case SamplerType::sobol: {
            sampler = std::unique_ptr<Sampler>(new SobolSampler(scene.use_gpu, options.seed, num_pixels));
            edge_sampler = std::unique_ptr<Sampler>(
                new SobolSampler(scene.use_gpu, options.seed + 131071U, num_pixels));
            break;
        } default: {
            assert(false);
            break;
        }
    }
    auto optix_rays = path_buffer.optix_rays.view(0, 2 * num_pixels);
    auto optix_hits = path_buffer.optix_hits.view(0, 2 * num_pixels);

    ThrustCachedAllocator thrust_alloc(scene.use_gpu, num_pixels * sizeof(DTexture3));

    // For each sample
    for (int sample_id = 0; sample_id < options.num_samples; sample_id++) {
        sampler->begin_sample(sample_id);

        // Buffer view for first intersection
        auto throughputs = path_buffer.throughputs.view(0, num_pixels);
        auto camera_samples = path_buffer.camera_samples.view(0, num_pixels);
        auto rays = path_buffer.rays.view(0, num_pixels);
        auto primary_differentials = path_buffer.primary_ray_differentials.view(0, num_pixels);
        auto ray_differentials = path_buffer.ray_differentials.view(0, num_pixels);
        auto shading_isects = path_buffer.shading_isects.view(0, num_pixels);
        auto shading_points = path_buffer.shading_points.view(0, num_pixels);
        auto primary_active_pixels = path_buffer.primary_active_pixels.view(0, num_pixels);
        auto active_pixels = path_buffer.active_pixels.view(0, num_pixels);
        auto min_roughness = path_buffer.min_roughness.view(0, num_pixels);

        // Initialization
        init_paths(throughputs, min_roughness, scene.use_gpu);
        // Generate primary ray samples
        sampler->next_camera_samples(camera_samples);
        sample_primary_rays(camera, camera_samples, rays, primary_differentials, scene.use_gpu);
        // Initialize pixel id
        init_active_pixels(rays, primary_active_pixels, scene.use_gpu, thrust_alloc);
        auto num_actives_primary = (int)primary_active_pixels.size();
        // Intersect with the scene
        intersect(scene,
                  primary_active_pixels,
                  rays,
                  primary_differentials,
                  shading_isects,
                  shading_points,
                  ray_differentials,
                  optix_rays,
                  optix_hits);
        accumulate_primary_contribs(scene,
                                    primary_active_pixels,
                                    throughputs,
                                    BufferView<Real>(), // channel multipliers
                                    rays,
                                    ray_differentials,
                                    shading_isects,
                                    shading_points,
                                    Real(1) / options.num_samples,
                                    channel_info,
                                    rendered_image.get(),
                                    BufferView<Real>()); // edge_contrib
        // Stream compaction: remove invalid intersection
        update_active_pixels(primary_active_pixels, shading_isects, active_pixels, scene.use_gpu);
        std::fill(num_active_pixels.begin(), num_active_pixels.end(), 0);
        num_active_pixels[0] = active_pixels.size();
        for (int depth = 0; depth < max_bounces &&
                num_active_pixels[depth] > 0; depth++) {
            // Buffer views for this path vertex
            const auto active_pixels =
                path_buffer.active_pixels.view(depth * num_pixels, num_active_pixels[depth]);
            auto light_samples =
                path_buffer.light_samples.view(depth * num_pixels, num_pixels);
            const auto shading_isects = path_buffer.shading_isects.view(
                depth * num_pixels, num_pixels);
            const auto shading_points = path_buffer.shading_points.view(
                depth * num_pixels, num_pixels);
            auto light_isects = path_buffer.light_isects.view(depth * num_pixels, num_pixels);
            auto light_points = path_buffer.light_points.view(depth * num_pixels, num_pixels);
            auto bsdf_samples = path_buffer.bsdf_samples.view(depth * num_pixels, num_pixels);
            auto incoming_rays = path_buffer.rays.view(depth * num_pixels, num_pixels);
            auto incoming_ray_differentials =
                path_buffer.ray_differentials.view(depth * num_pixels, num_pixels);
            auto bsdf_ray_differentials =
                path_buffer.bsdf_ray_differentials.view(depth * num_pixels, num_pixels);
            auto nee_rays = path_buffer.nee_rays.view(depth * num_pixels, num_pixels);
            auto next_rays = path_buffer.rays.view((depth + 1) * num_pixels, num_pixels);
            auto next_ray_differentials = 
                path_buffer.ray_differentials.view((depth + 1) * num_pixels, num_pixels);
            auto bsdf_isects =
                path_buffer.shading_isects.view((depth + 1) * num_pixels, num_pixels);
            auto bsdf_points =
                path_buffer.shading_points.view((depth + 1) * num_pixels, num_pixels);
            const auto throughputs =
                path_buffer.throughputs.view(depth * num_pixels, num_pixels);
            auto next_throughputs =
                path_buffer.throughputs.view((depth + 1) * num_pixels, num_pixels);
            auto next_active_pixels =
                path_buffer.active_pixels.view((depth + 1) * num_pixels, num_pixels);
            auto min_roughness = path_buffer.min_roughness.view(depth * num_pixels, num_pixels);
            auto next_min_roughness =
                path_buffer.min_roughness.view((depth + 1) * num_pixels, num_pixels);

            // Sample points on lights
            sampler->next_light_samples(light_samples);
            sample_point_on_light(scene,
                                  active_pixels,
                                  shading_points,
                                  light_samples,
                                  light_isects,
                                  light_points,
                                  nee_rays);
            occluded(scene, active_pixels, nee_rays, optix_rays, optix_hits);
            
            // Sample directions based on BRDF
            sampler->next_bsdf_samples(bsdf_samples);
            bsdf_sample(scene,
                        active_pixels,
                        incoming_rays,
                        incoming_ray_differentials,
                        shading_isects,
                        shading_points,
                        bsdf_samples,
                        min_roughness,
                        next_rays,
                        bsdf_ray_differentials,
                        next_min_roughness);
            // Intersect with the scene
            intersect(scene,
                      active_pixels,
                      next_rays,
                      bsdf_ray_differentials,
                      bsdf_isects,
                      bsdf_points,
                      next_ray_differentials,
                      optix_rays,
                      optix_hits);

            // Compute path contribution & update throughput
            accumulate_path_contribs(
                scene,
                active_pixels,
                throughputs,
                incoming_rays,
                shading_isects,
                shading_points,
                light_isects,
                light_points,
                nee_rays,
                bsdf_isects,
                bsdf_points,
                next_rays,
                min_roughness,
                Real(1) / options.num_samples,
                channel_info,
                next_throughputs,
                rendered_image.get(),
                BufferView<Real>());
 
            // Stream compaction: remove invalid bsdf intersections
            // active_pixels -> next_active_pixels
            update_active_pixels(active_pixels, bsdf_isects, next_active_pixels, scene.use_gpu); 

            // Record the number of active pixels for next depth
            num_active_pixels[depth + 1] = next_active_pixels.size();
        }

        if (d_rendered_image.get() != nullptr) {
            edge_sampler->begin_sample(sample_id);

            // Traverse the path backward for the derivatives
            bool first = true;
            for (int depth = max_bounces - 1; depth >= 0; depth--) {
                // Buffer views for this path vertex
                auto num_actives = num_active_pixels[depth];
                if (num_actives <= 0) {
                    continue;
                }
                auto active_pixels =
                    path_buffer.active_pixels.view(depth * num_pixels, num_actives);
                auto d_next_throughputs = path_buffer.d_next_throughputs.view(0, num_pixels);
                auto d_next_rays = path_buffer.d_next_rays.view(0, num_pixels);
                auto d_next_ray_differentials =
                    path_buffer.d_next_ray_differentials.view(0, num_pixels);
                auto d_next_points = path_buffer.d_next_points.view(0, num_pixels);
                auto throughputs = path_buffer.throughputs.view(depth * num_pixels, num_pixels);
                auto incoming_rays = path_buffer.rays.view(depth * num_pixels, num_pixels);
                auto incoming_ray_differentials =
                    path_buffer.ray_differentials.view(depth * num_pixels, num_pixels);
                auto next_rays = path_buffer.rays.view((depth + 1) * num_pixels, num_pixels);
                auto bsdf_ray_differentials =
                    path_buffer.bsdf_ray_differentials.view(depth * num_pixels, num_pixels);
                auto nee_rays = path_buffer.nee_rays.view(depth * num_pixels, num_pixels);
                auto light_samples = path_buffer.light_samples.view(
                    depth * num_pixels, num_pixels);
                auto bsdf_samples = path_buffer.bsdf_samples.view(depth * num_pixels, num_pixels);
                auto shading_isects = path_buffer.shading_isects.view(
                    depth * num_pixels, num_pixels);
                auto shading_points = path_buffer.shading_points.view(
                    depth * num_pixels, num_pixels);
                auto light_isects =
                    path_buffer.light_isects.view(depth * num_pixels, num_pixels);
                auto light_points =
                    path_buffer.light_points.view(depth * num_pixels, num_pixels);
                auto bsdf_isects = path_buffer.shading_isects.view(
                    (depth + 1) * num_pixels, num_pixels);
                auto bsdf_points = path_buffer.shading_points.view(
                    (depth + 1) * num_pixels, num_pixels);
                auto min_roughness =
                    path_buffer.min_roughness.view(depth * num_pixels, num_pixels);

                auto d_throughputs = path_buffer.d_throughputs.view(0, num_pixels);
                auto d_rays = path_buffer.d_rays.view(0, num_pixels);
                auto d_ray_differentials = path_buffer.d_ray_differentials.view(0, num_pixels);
                auto d_points = path_buffer.d_points.view(0, num_pixels);

                auto d_light_vertices = path_buffer.d_light_vertices.view(0, 3 * num_actives);
                auto d_bsdf_vertices = path_buffer.d_bsdf_vertices.view(0, 3 * num_actives);
                auto d_diffuse_texs = path_buffer.d_diffuse_texs.view(0, num_actives);
                auto d_specular_texs = path_buffer.d_specular_texs.view(0, num_actives);
                auto d_roughness_texs = path_buffer.d_roughness_texs.view(0, num_actives);
                auto d_nee_lights = path_buffer.d_nee_lights.view(0, num_actives);
                auto d_bsdf_lights = path_buffer.d_bsdf_lights.view(0, num_actives);
                auto d_envmap_vals = path_buffer.d_envmap_vals.view(0, num_actives);
                auto d_world_to_envs = path_buffer.d_world_to_envs.view(0, num_actives);

                if (first) {
                    first = false;
                    // Initialize the derivatives propagated from the next vertex
                    DISPATCH(scene.use_gpu, thrust::fill,
                        d_next_throughputs.begin(), d_next_throughputs.end(),
                        Vector3{0, 0, 0});
                    DISPATCH(scene.use_gpu, thrust::fill,
                        d_next_rays.begin(), d_next_rays.end(), DRay{});
                    DISPATCH(scene.use_gpu, thrust::fill,
                        d_next_ray_differentials.begin(), d_next_ray_differentials.end(),
                        RayDifferential{Vector3{0, 0, 0}, Vector3{0, 0, 0},
                                        Vector3{0, 0, 0}, Vector3{0, 0, 0}});
                    DISPATCH(scene.use_gpu, thrust::fill,
                        d_next_points.begin(), d_next_points.end(),
                        SurfacePoint::zero());
                }

                // Backpropagate path contribution
                d_accumulate_path_contribs(
                    scene,
                    active_pixels,
                    throughputs,
                    incoming_rays,
                    incoming_ray_differentials,
                    light_samples, bsdf_samples,
                    shading_isects, shading_points,
                    light_isects, light_points, nee_rays,
                    bsdf_isects, bsdf_points, next_rays, bsdf_ray_differentials,
                    min_roughness,
                    Real(1) / options.num_samples, // weight
                    channel_info,
                    d_rendered_image.get(),
                    d_next_throughputs,
                    d_next_rays,
                    d_next_ray_differentials,
                    d_next_points,
                    d_light_vertices,
                    d_bsdf_vertices,
                    d_diffuse_texs,
                    d_specular_texs,
                    d_roughness_texs,
                    d_nee_lights,
                    d_bsdf_lights,
                    d_envmap_vals,
                    d_world_to_envs,
                    d_throughputs,
                    d_rays,
                    d_ray_differentials,
                    d_points);

                if (scene.use_secondary_edge_sampling) {
                    ////////////////////////////////////////////////////////////////////////////////
                    // Sample edges for secondary visibility
                    auto num_edge_samples = 2 * num_actives;
                    auto edge_samples = path_buffer.secondary_edge_samples.view(0, num_actives);
                    edge_sampler->next_secondary_edge_samples(edge_samples);
                    auto edge_records = path_buffer.secondary_edge_records.view(0, num_actives);
                    auto edge_rays = path_buffer.edge_rays.view(0, num_edge_samples);
                    auto edge_ray_differentials =
                        path_buffer.edge_ray_differentials.view(0, num_edge_samples);
                    auto edge_throughputs = path_buffer.edge_throughputs.view(0, num_edge_samples);
                    auto edge_shading_isects =
                        path_buffer.edge_shading_isects.view(0, num_edge_samples);
                    auto edge_shading_points =
                        path_buffer.edge_shading_points.view(0, num_edge_samples);
                    auto edge_min_roughness =
                        path_buffer.edge_min_roughness.view(0, num_edge_samples);
                    sample_secondary_edges(
                        scene,
                        active_pixels,
                        edge_samples,
                        incoming_rays,
                        incoming_ray_differentials,
                        shading_isects,
                        shading_points,
                        nee_rays,
                        light_isects,
                        light_points,
                        throughputs,
                        min_roughness,
                        d_rendered_image.get(),
                        channel_info,
                        edge_records,
                        edge_rays,
                        edge_ray_differentials,
                        edge_throughputs,
                        edge_min_roughness);

                    // Now we path trace these edges
                    auto edge_active_pixels = path_buffer.edge_active_pixels.view(0, num_edge_samples);
                    init_active_pixels(edge_rays, edge_active_pixels, scene.use_gpu, thrust_alloc);
                    // Intersect with the scene
                    intersect(scene,
                              edge_active_pixels,
                              edge_rays,
                              edge_ray_differentials,
                              edge_shading_isects,
                              edge_shading_points,
                              edge_ray_differentials,
                              optix_rays,
                              optix_hits);
                    // Update edge throughputs: take geometry terms and Jacobians into account
                    update_secondary_edge_weights(scene,
                                                  active_pixels,
                                                  shading_points,
                                                  edge_shading_isects,
                                                  edge_shading_points,
                                                  edge_records,
                                                  edge_throughputs);
                    // Initialize edge contribution
                    auto edge_contribs = path_buffer.edge_contribs.view(0, num_edge_samples);
                    DISPATCH(scene.use_gpu, thrust::fill,
                        edge_contribs.begin(), edge_contribs.end(), 0);
                    accumulate_primary_contribs(
                        scene,
                        edge_active_pixels,
                        edge_throughputs,
                        BufferView<Real>(), // channel multipliers
                        edge_rays,
                        edge_ray_differentials,
                        edge_shading_isects,
                        edge_shading_points,
                        Real(1) / options.num_samples,
                        channel_info,
                        nullptr,
                        edge_contribs);
                    // Stream compaction: remove invalid intersections
                    update_active_pixels(edge_active_pixels,
                                         edge_shading_isects,
                                         edge_active_pixels,
                                         scene.use_gpu);
                    auto num_active_edge_samples = edge_active_pixels.size();
                    // Record the hit points for derivatives computation later
                    auto edge_surface_points =
                        path_buffer.edge_surface_points.view(0, num_active_edge_samples);
                    parallel_for(get_position{
                        edge_active_pixels.begin(),
                        edge_shading_points.begin(),
                        edge_surface_points.begin()}, num_active_edge_samples, scene.use_gpu);
                    for (int edge_depth = depth + 1; edge_depth < max_bounces &&
                           num_active_edge_samples > 0; edge_depth++) {
                        // Path tracing loop for secondary edges
                        auto edge_depth_ = edge_depth - (depth + 1);
                        auto main_buffer_beg = (edge_depth_ % 2) * (2 * num_pixels);
                        auto next_buffer_beg = ((edge_depth_ + 1) % 2) * (2 * num_pixels);
                        const auto active_pixels = path_buffer.edge_active_pixels.view(
                            main_buffer_beg, num_active_edge_samples);
                        auto light_samples = path_buffer.edge_light_samples.view(0, num_edge_samples);
                        auto bsdf_samples = path_buffer.edge_bsdf_samples.view(0, num_edge_samples);
                        auto tmp_light_samples = path_buffer.tmp_light_samples.view(0, num_actives);
                        auto tmp_bsdf_samples = path_buffer.tmp_bsdf_samples.view(0, num_actives);
                        auto shading_isects =
                            path_buffer.edge_shading_isects.view(main_buffer_beg, num_edge_samples);
                        auto shading_points =
                            path_buffer.edge_shading_points.view(main_buffer_beg, num_edge_samples);
                        auto light_isects = path_buffer.edge_light_isects.view(0, num_edge_samples);
                        auto light_points = path_buffer.edge_light_points.view(0, num_edge_samples);
                        auto incoming_rays =
                            path_buffer.edge_rays.view(main_buffer_beg, num_edge_samples);
                        auto ray_differentials =
                            path_buffer.edge_ray_differentials.view(0, num_edge_samples);
                        auto nee_rays = path_buffer.edge_nee_rays.view(0, num_edge_samples);
                        auto next_rays = path_buffer.edge_rays.view(next_buffer_beg, num_edge_samples);
                        auto bsdf_isects = path_buffer.edge_shading_isects.view(
                            next_buffer_beg, num_edge_samples);
                        auto bsdf_points = path_buffer.edge_shading_points.view(
                            next_buffer_beg, num_edge_samples);
                        const auto throughputs = path_buffer.edge_throughputs.view(
                            main_buffer_beg, num_edge_samples);
                        auto next_throughputs = path_buffer.edge_throughputs.view(
                            next_buffer_beg, num_edge_samples);
                        auto next_active_pixels = path_buffer.edge_active_pixels.view(
                            next_buffer_beg, num_edge_samples);
                        auto edge_min_roughness =
                            path_buffer.edge_min_roughness.view(main_buffer_beg, num_edge_samples);
                        auto edge_next_min_roughness =
                            path_buffer.edge_min_roughness.view(next_buffer_beg, num_edge_samples);

                        // Sample points on lights
                        edge_sampler->next_light_samples(tmp_light_samples);
                        // Copy the samples
                        parallel_for(copy_interleave<LightSample>{
                            tmp_light_samples.begin(), light_samples.begin()},
                            tmp_light_samples.size(), scene.use_gpu);
                        sample_point_on_light(
                            scene, active_pixels, shading_points,
                            light_samples, light_isects, light_points, nee_rays);
                        occluded(scene, active_pixels, nee_rays, optix_rays, optix_hits);

                        // Sample directions based on BRDF
                        edge_sampler->next_bsdf_samples(tmp_bsdf_samples);
                        // Copy the samples
                        parallel_for(copy_interleave<BSDFSample>{
                            tmp_bsdf_samples.begin(), bsdf_samples.begin()},
                            tmp_bsdf_samples.size(), scene.use_gpu);
                        bsdf_sample(scene,
                                    active_pixels,
                                    incoming_rays,
                                    ray_differentials,
                                    shading_isects,
                                    shading_points,
                                    bsdf_samples,
                                    edge_min_roughness,
                                    next_rays,
                                    ray_differentials,
                                    edge_next_min_roughness);
                        // Intersect with the scene
                        intersect(scene,
                                  active_pixels,
                                  next_rays,
                                  ray_differentials,
                                  bsdf_isects,
                                  bsdf_points,
                                  ray_differentials,
                                  optix_rays,
                                  optix_hits);

                        // Compute path contribution & update throughput
                        accumulate_path_contribs(
                            scene,
                            active_pixels,
                            throughputs,
                            incoming_rays,
                            shading_isects,
                            shading_points,
                            light_isects,
                            light_points,
                            nee_rays,
                            bsdf_isects,
                            bsdf_points,
                            next_rays,
                            edge_min_roughness,
                            Real(1) / options.num_samples,
                            channel_info,
                            next_throughputs,
                            nullptr,
                            edge_contribs);

                        // Stream compaction: remove invalid bsdf intersections
                        // active_pixels -> next_active_pixels
                        update_active_pixels(active_pixels, bsdf_isects,
                                             next_active_pixels, scene.use_gpu);
                        num_active_edge_samples = next_active_pixels.size();
                    }
                    // Now the path traced contribution for the edges is stored in edge_contribs
                    // We'll compute the derivatives w.r.t. three points: two on edges and one on
                    // the shading point
                    auto d_edge_vertices = path_buffer.d_general_vertices.view(0, num_edge_samples);
                    accumulate_secondary_edge_derivatives(scene,
                                                          active_pixels,
                                                          shading_points,
                                                          edge_records,
                                                          edge_surface_points,
                                                          edge_contribs,
                                                          d_points,
                                                          d_edge_vertices);
                    // Deposit vertices, texture, light derivatives
                    // sort the derivatives by id & reduce by key
                    accumulate_vertex(
                        d_edge_vertices,
                        path_buffer.d_vertex_reduce_buffer.view(0, 2 * num_actives),
                        d_scene->shapes.view(0, d_scene->shapes.size()),
                        scene.use_gpu,
                        thrust_alloc);
                    ////////////////////////////////////////////////////////////////////////////////
                }
                /*cuda_synchronize();
                for (int i = 0; i < num_actives; i++) {
                    auto pixel_id = active_pixels[i];
                    // auto d_p = d_points[pixel_id].position;
                    // debug_image[3 * pixel_id + 0] += d_p[0];
                    // debug_image[3 * pixel_id + 1] += d_p[0];
                    // debug_image[3 * pixel_id + 2] += d_p[0];
                    auto c = 1;
                    auto d_e_v0 = d_edge_vertices[2 * i + 0];
                    auto d_e_v1 = d_edge_vertices[2 * i + 1];
                    if (d_e_v0.shape_id == 1) {
                        auto d_v0 = d_e_v0.d_v;
                        auto d_v1 = d_e_v1.d_v;
                        debug_image[3 * pixel_id + 0] += d_v0[c] + d_v1[c];
                        debug_image[3 * pixel_id + 1] += d_v0[c] + d_v1[c];
                        debug_image[3 * pixel_id + 2] += d_v0[c] + d_v1[c];
                    }
                    auto d_l_v0 = d_light_vertices[3 * i + 0];
                    auto d_l_v1 = d_light_vertices[3 * i + 1];
                    auto d_l_v2 = d_light_vertices[3 * i + 2];
                    if (d_l_v0.shape_id == 6) {
                        auto d_v0 = d_l_v0.d_v;
                        auto d_v1 = d_l_v1.d_v;
                        auto d_v2 = d_l_v2.d_v;
                        debug_image[3 * pixel_id + 0] += d_v0[c] + d_v1[c] + d_v2[c];
                        debug_image[3 * pixel_id + 1] += d_v0[c] + d_v1[c] + d_v2[c];
                        debug_image[3 * pixel_id + 2] += d_v0[c] + d_v1[c] + d_v2[c];
                    }
                    auto d_b_v0 = d_bsdf_vertices[3 * i + 0];
                    auto d_b_v1 = d_bsdf_vertices[3 * i + 1];
                    auto d_b_v2 = d_bsdf_vertices[3 * i + 2];
                    if (d_b_v0.shape_id == 6) {
                        auto d_v0 = d_b_v0.d_v;
                        auto d_v1 = d_b_v1.d_v;
                        auto d_v2 = d_b_v2.d_v;
                        debug_image[3 * pixel_id + 0] += d_v0[c] + d_v1[c] + d_v2[c];
                        debug_image[3 * pixel_id + 1] += d_v0[c] + d_v1[c] + d_v2[c];
                        debug_image[3 * pixel_id + 2] += d_v0[c] + d_v1[c] + d_v2[c];
                    }
                }*/

                // Deposit vertices, texture, light derivatives
                // sort the derivatives by id & reduce by key
                accumulate_vertex(
                    d_light_vertices, 
                    path_buffer.d_vertex_reduce_buffer.view(0, 3 * num_actives),
                    d_scene->shapes.view(0, d_scene->shapes.size()),
                    scene.use_gpu,
                    thrust_alloc);
                accumulate_vertex(
                    d_bsdf_vertices, 
                    path_buffer.d_vertex_reduce_buffer.view(0, 3 * num_actives),
                    d_scene->shapes.view(0, d_scene->shapes.size()),
                    scene.use_gpu,
                    thrust_alloc);

                // for (int i = 0; i < active_pixels.size(); i++) {
                //     auto pixel_id = active_pixels[i];
                //     auto d_diffuse_tex = d_diffuse_texs[i];
                //     if (d_diffuse_tex.material_id == 4) {
                //         debug_image[3 * pixel_id + 0] += d_diffuse_tex.t00[0];
                //         debug_image[3 * pixel_id + 1] += d_diffuse_tex.t00[1];
                //         debug_image[3 * pixel_id + 2] += d_diffuse_tex.t00[2];
                //     }
                // }
                accumulate_diffuse(
                    scene,
                    d_diffuse_texs,
                    path_buffer.d_tex3_reduce_buffer.view(0, num_actives),
                    d_scene->materials.view(0, d_scene->materials.size()),
                    thrust_alloc);
                accumulate_specular(
                    scene,
                    d_specular_texs,
                    path_buffer.d_tex3_reduce_buffer.view(0, num_actives),
                    d_scene->materials.view(0, d_scene->materials.size()),
                    thrust_alloc);
                // for (int i = 0; i < active_pixels.size(); i++) {
                //     auto pixel_id = active_pixels[i];
                //     auto d_roughness_tex = d_roughness_texs[i];
                //     if (d_roughness_tex.material_id == 4) {
                //         debug_image[3 * pixel_id + 0] += d_roughness_tex.t000;
                //         debug_image[3 * pixel_id + 1] += d_roughness_tex.t000;
                //         debug_image[3 * pixel_id + 2] += d_roughness_tex.t000;
                //     }
                // }
                accumulate_roughness(
                    scene,
                    d_roughness_texs,
                    path_buffer.d_tex1_reduce_buffer.view(0, num_actives),
                    d_scene->materials.view(0, d_scene->materials.size()),
                    thrust_alloc);
                accumulate_area_light(
                    d_nee_lights,
                    path_buffer.d_lgt_reduce_buffer.view(0, num_actives),
                    d_scene->area_lights.view(0, d_scene->area_lights.size()),
                    scene.use_gpu,
                    thrust_alloc);
                accumulate_area_light(
                    d_bsdf_lights,
                    path_buffer.d_lgt_reduce_buffer.view(0, num_actives),
                    d_scene->area_lights.view(0, d_scene->area_lights.size()),
                    scene.use_gpu,
                    thrust_alloc);
                if (scene.envmap != nullptr) {
                    accumulate_envmap(
                        scene,
                        d_envmap_vals,
                        d_world_to_envs,
                        path_buffer.d_envmap_reduce_buffer.view(0, num_actives),
                        d_scene->envmap,
                        thrust_alloc);
                }

                // Previous become next
                std::swap(path_buffer.d_next_throughputs, path_buffer.d_throughputs);
                std::swap(path_buffer.d_next_rays, path_buffer.d_rays);
                std::swap(path_buffer.d_next_ray_differentials, path_buffer.d_ray_differentials);
                std::swap(path_buffer.d_next_points, path_buffer.d_points);
            }
            
            // Backpropagate from first vertex to camera
            // Buffer view for first intersection
            if (num_actives_primary > 0) {
                const auto primary_active_pixels =
                    path_buffer.primary_active_pixels.view(0, num_actives_primary);
                const auto throughputs = path_buffer.throughputs.view(0, num_pixels);
                const auto camera_samples = path_buffer.camera_samples.view(0, num_pixels);
                const auto rays = path_buffer.rays.view(0, num_pixels);
                const auto primary_ray_differentials =
                    path_buffer.primary_ray_differentials.view(0, num_pixels);
                const auto ray_differentials = path_buffer.ray_differentials.view(0, num_pixels);
                const auto shading_isects = path_buffer.shading_isects.view(0, num_pixels);
                const auto shading_points = path_buffer.shading_points.view(0, num_pixels);
                const auto d_rays = path_buffer.d_next_rays.view(0, num_pixels);
                const auto d_ray_differentials =
                    path_buffer.d_next_ray_differentials.view(0, num_pixels);
                auto d_points = path_buffer.d_next_points.view(0, num_pixels);
                auto d_direct_lights = path_buffer.d_direct_lights.view(0, num_actives_primary);
                auto d_envmap_vals = path_buffer.d_envmap_vals.view(0, num_actives_primary);
                auto d_world_to_envs = path_buffer.d_world_to_envs.view(0, num_actives_primary);
                auto d_diffuse_texs = path_buffer.d_diffuse_texs.view(0, num_actives_primary);
                auto d_specular_texs = path_buffer.d_specular_texs.view(0, num_actives_primary);
                auto d_roughness_texs = path_buffer.d_roughness_texs.view(0, num_actives_primary);
                auto d_primary_vertices =
                    path_buffer.d_general_vertices.view(0, 3 * num_actives_primary);
                auto d_cameras = path_buffer.d_cameras.view(0, num_actives_primary);

                d_accumulate_primary_contribs(scene,
                                              primary_active_pixels,
                                              throughputs,
                                              BufferView<Real>(), // channel multiplers
                                              rays,
                                              ray_differentials,
                                              shading_isects,
                                              shading_points,
                                              Real(1) / options.num_samples,
                                              channel_info,
                                              d_rendered_image.get(),
                                              d_direct_lights,
                                              d_envmap_vals,
                                              d_world_to_envs,
                                              d_rays,
                                              d_ray_differentials,
                                              d_points,
                                              d_diffuse_texs,
                                              d_specular_texs,
                                              d_roughness_texs);
                // Propagate to camera
                d_primary_intersection(scene,
                                       primary_active_pixels,
                                       camera_samples,
                                       rays,
                                       primary_ray_differentials,
                                       shading_isects,
                                       d_rays,
                                       d_ray_differentials,
                                       d_points,
                                       d_primary_vertices,
                                       d_cameras);

                /*cuda_synchronize();
                for (int i = 0; i < primary_active_pixels.size(); i++) {
                    auto c = 1;
                    auto pixel_id = primary_active_pixels[i];
                    auto d_v0 = d_primary_vertices[3 * i + 0];
                    auto d_v1 = d_primary_vertices[3 * i + 1];
                    auto d_v2 = d_primary_vertices[3 * i + 2];
                    if (d_v0.shape_id == 6) {
                        debug_image[3 * pixel_id + 0] += (d_v0.d_v[c] + d_v1.d_v[c] + d_v2.d_v[c]);
                        debug_image[3 * pixel_id + 1] += (d_v0.d_v[c] + d_v1.d_v[c] + d_v2.d_v[c]);
                        debug_image[3 * pixel_id + 2] += (d_v0.d_v[c] + d_v1.d_v[c] + d_v2.d_v[c]);
                    }
                }*/

                // for (int i = 0; i < primary_active_pixels.size(); i++) {
                //     auto pixel_id = primary_active_pixels[i];
                //     auto d_pos = d_cameras[i].position;
                //     debug_image[3 * pixel_id + 0] += d_pos[0];
                //     debug_image[3 * pixel_id + 1] += d_pos[0];
                //     debug_image[3 * pixel_id + 2] += d_pos[0];
                // }

                // Deposit derivatives
                accumulate_area_light(
                    d_direct_lights,
                    path_buffer.d_lgt_reduce_buffer.view(0, num_actives_primary),
                    d_scene->area_lights.view(0, d_scene->area_lights.size()),
                    scene.use_gpu,
                    thrust_alloc);
                if (scene.envmap != nullptr) {
                    accumulate_envmap(
                        scene,
                        d_envmap_vals,
                        d_world_to_envs,
                        path_buffer.d_envmap_reduce_buffer.view(0, num_actives_primary),
                        d_scene->envmap,
                        thrust_alloc);
                }
                accumulate_vertex(
                    d_primary_vertices,
                    path_buffer.d_vertex_reduce_buffer.view(0, num_actives_primary),
                    d_scene->shapes.view(0, d_scene->shapes.size()),
                    scene.use_gpu,
                    thrust_alloc);
                accumulate_diffuse(
                    scene,
                    d_diffuse_texs,
                    path_buffer.d_tex3_reduce_buffer.view(0, num_actives_primary),
                    d_scene->materials.view(0, d_scene->materials.size()),
                    thrust_alloc);
                accumulate_specular(
                    scene,
                    d_specular_texs,
                    path_buffer.d_tex3_reduce_buffer.view(0, num_actives_primary),
                    d_scene->materials.view(0, d_scene->materials.size()),
                    thrust_alloc);
                accumulate_roughness(
                    scene,
                    d_roughness_texs,
                    path_buffer.d_tex1_reduce_buffer.view(0, num_actives_primary),
                    d_scene->materials.view(0, d_scene->materials.size()),
                    thrust_alloc);

                // Reduce the camera array
                DCameraInst d_camera = DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::reduce,
                    d_cameras.begin(), d_cameras.end(), DCameraInst{});
                accumulate_camera(camera, d_camera, d_scene->camera, scene.use_gpu);
            }

            /////////////////////////////////////////////////////////////////////////////////
            // Sample primary edges for geometric derivatives
            if (scene.use_primary_edge_sampling && scene.edge_sampler.edges.size() > 0) {
                auto primary_edge_samples = path_buffer.primary_edge_samples.view(0, num_pixels);
                auto edge_records = path_buffer.primary_edge_records.view(0, num_pixels);
                auto rays = path_buffer.edge_rays.view(0, 2 * num_pixels);
                auto ray_differentials =
                    path_buffer.edge_ray_differentials.view(0, 2 * num_pixels);
                auto throughputs = path_buffer.edge_throughputs.view(0, 2 * num_pixels);
                auto channel_multipliers = path_buffer.channel_multipliers.view(
                    0, 2 * channel_info.num_total_dimensions * num_pixels);
                auto shading_isects = path_buffer.edge_shading_isects.view(0, 2 * num_pixels);
                auto shading_points = path_buffer.edge_shading_points.view(0, 2 * num_pixels);
                auto active_pixels = path_buffer.edge_active_pixels.view(0, 2 * num_pixels);
                auto edge_contribs = path_buffer.edge_contribs.view(0, 2 * num_pixels);
                auto edge_min_roughness = path_buffer.edge_min_roughness.view(0, 2 * num_pixels);
                // Initialize edge contribution
                DISPATCH(scene.use_gpu, thrust::fill,
                         edge_contribs.begin(), edge_contribs.end(), 0);
                // Initialize max roughness
                DISPATCH(scene.use_gpu, thrust::fill,
                         edge_min_roughness.begin(), edge_min_roughness.end(), 0);

                // Generate rays & weights for edge sampling
                edge_sampler->next_primary_edge_samples(primary_edge_samples);
                sample_primary_edges(scene,
                                     primary_edge_samples,
                                     d_rendered_image.get(),
                                     channel_info,
                                     edge_records,
                                     rays,
                                     ray_differentials,
                                     throughputs,
                                     channel_multipliers);
                // Initialize pixel id
                init_active_pixels(rays, active_pixels, scene.use_gpu, thrust_alloc);

                // Intersect with the scene
                intersect(scene,
                          active_pixels,
                          rays,
                          ray_differentials,
                          shading_isects,
                          shading_points,
                          ray_differentials,
                          optix_rays,
                          optix_hits);
                update_primary_edge_weights(scene,
                                            edge_records,
                                            shading_isects,
                                            channel_info,
                                            throughputs,
                                            channel_multipliers);
                accumulate_primary_contribs(scene,
                                            active_pixels,
                                            throughputs,
                                            channel_multipliers,
                                            rays,
                                            ray_differentials,
                                            shading_isects,
                                            shading_points,
                                            Real(1) / options.num_samples,
                                            channel_info,
                                            nullptr,
                                            edge_contribs);
                // Stream compaction: remove invalid intersections
                update_active_pixels(active_pixels, shading_isects, active_pixels, scene.use_gpu);
                auto active_pixels_size = active_pixels.size();
                for (int depth = 0; depth < max_bounces && active_pixels_size > 0; depth++) {
                    // Buffer views for this path vertex
                    auto main_buffer_beg = (depth % 2) * (2 * num_pixels);
                    auto next_buffer_beg = ((depth + 1) % 2) * (2 * num_pixels);
                    const auto active_pixels =
                        path_buffer.edge_active_pixels.view(main_buffer_beg, active_pixels_size);
                    auto light_samples = path_buffer.edge_light_samples.view(0, 2 * num_pixels);
                    auto bsdf_samples = path_buffer.edge_bsdf_samples.view(0, 2 * num_pixels);
                    auto tmp_light_samples = path_buffer.tmp_light_samples.view(0, num_pixels);
                    auto tmp_bsdf_samples = path_buffer.tmp_bsdf_samples.view(0, num_pixels);
                    auto shading_isects =
                        path_buffer.edge_shading_isects.view(main_buffer_beg, 2 * num_pixels);
                    auto shading_points =
                        path_buffer.edge_shading_points.view(main_buffer_beg, 2 * num_pixels);
                    auto light_isects = path_buffer.edge_light_isects.view(0, 2 * num_pixels);
                    auto light_points = path_buffer.edge_light_points.view(0, 2 * num_pixels);
                    auto nee_rays = path_buffer.edge_nee_rays.view(0, 2 * num_pixels);
                    auto incoming_rays =
                        path_buffer.edge_rays.view(main_buffer_beg, 2 * num_pixels);
                    auto ray_differentials =
                        path_buffer.edge_ray_differentials.view(0, 2 * num_pixels);
                    auto next_rays = path_buffer.edge_rays.view(next_buffer_beg, 2 * num_pixels);
                    auto bsdf_isects = path_buffer.edge_shading_isects.view(
                        next_buffer_beg, 2 * num_pixels);
                    auto bsdf_points = path_buffer.edge_shading_points.view(
                        next_buffer_beg, 2 * num_pixels);
                    const auto throughputs = path_buffer.edge_throughputs.view(
                        main_buffer_beg, 2 * num_pixels);
                    auto next_throughputs = path_buffer.edge_throughputs.view(
                        next_buffer_beg, 2 * num_pixels);
                    auto next_active_pixels = path_buffer.edge_active_pixels.view(
                        next_buffer_beg, 2 * num_pixels);
                    auto edge_min_roughness =
                        path_buffer.edge_min_roughness.view(main_buffer_beg, 2 * num_pixels);
                    auto edge_next_min_roughness =
                        path_buffer.edge_min_roughness.view(next_buffer_beg, 2 * num_pixels);

                    // Sample points on lights
                    edge_sampler->next_light_samples(tmp_light_samples);
                    // Copy the samples
                    parallel_for(copy_interleave<LightSample>{
                        tmp_light_samples.begin(), light_samples.begin()},
                        tmp_light_samples.size(), scene.use_gpu);
                    sample_point_on_light(
                        scene, active_pixels, shading_points,
                        light_samples, light_isects, light_points, nee_rays);
                    occluded(scene, active_pixels, nee_rays, optix_rays, optix_hits);

                    // Sample directions based on BRDF
                    edge_sampler->next_bsdf_samples(tmp_bsdf_samples);
                    // Copy the samples
                    parallel_for(copy_interleave<BSDFSample>{
                        tmp_bsdf_samples.begin(), bsdf_samples.begin()},
                        tmp_bsdf_samples.size(), scene.use_gpu);
                    bsdf_sample(scene,
                                active_pixels,
                                incoming_rays,
                                ray_differentials,
                                shading_isects,
                                shading_points,
                                bsdf_samples,
                                edge_min_roughness,
                                next_rays,
                                ray_differentials,
                                edge_next_min_roughness);
                    // Intersect with the scene
                    intersect(scene,
                              active_pixels,
                              next_rays,
                              ray_differentials,
                              bsdf_isects,
                              bsdf_points,
                              ray_differentials,
                              optix_rays,
                              optix_hits);
                    // Compute path contribution & update throughput
                    accumulate_path_contribs(
                        scene,
                        active_pixels,
                        throughputs,
                        incoming_rays,
                        shading_isects,
                        shading_points,
                        light_isects,
                        light_points,
                        nee_rays,
                        bsdf_isects,
                        bsdf_points,
                        next_rays,
                        edge_min_roughness,
                        Real(1) / options.num_samples,
                        channel_info,
                        next_throughputs,
                        nullptr,
                        edge_contribs);

                    // Stream compaction: remove invalid bsdf intersections
                    // active_pixels -> next_active_pixels
                    update_active_pixels(active_pixels, bsdf_isects,
                        next_active_pixels, scene.use_gpu);
                    active_pixels_size = next_active_pixels.size();
                }

                // Convert edge contributions to vertex derivatives
                auto d_vertices = path_buffer.d_general_vertices.view(0, 2 * num_pixels);
                auto d_cameras = path_buffer.d_cameras.view(0, num_pixels);
                compute_primary_edge_derivatives(
                    scene, edge_records, edge_contribs, d_vertices, d_cameras);

                // for (int i = 0; i < edge_records.size(); i++) {
                //     auto rec = edge_records[i];
                //     if (rec.edge.shape_id >= 0) {
                //         auto edge_pt = rec.edge_pt;
                //         auto xi = int(edge_pt[0] * camera.width);
                //         auto yi = int(edge_pt[1] * camera.height);
                //         auto d_v0 = d_vertices[2 * i + 0].d_v;
                //         auto d_v1 = d_vertices[2 * i + 1].d_v;
                //         debug_image[3 * (yi * camera.width + xi) + 0] += d_v0[0] + d_v1[0];
                //         debug_image[3 * (yi * camera.width + xi) + 1] += d_v0[0] + d_v1[0];
                //         debug_image[3 * (yi * camera.width + xi) + 2] += d_v0[0] + d_v1[0];
                //     }
                // }

                // Deposit vertices
                accumulate_vertex(
                    d_vertices,
                    path_buffer.d_vertex_reduce_buffer.view(0, d_vertices.size()),
                    d_scene->shapes.view(0, d_scene->shapes.size()),
                    scene.use_gpu,
                    thrust_alloc);

                // Reduce the camera array
                DCameraInst d_camera = DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::reduce,
                    d_cameras.begin(), d_cameras.end(), DCameraInst{});
                accumulate_camera(camera, d_camera, d_scene->camera, scene.use_gpu);

                // for (int i = 0; i < edge_records.size(); i++) {
                //     auto rec = edge_records[i];
                //     auto edge_pt = rec.edge_pt;
                //     auto xi = int(edge_pt[0] * camera.width);
                //     auto yi = int(edge_pt[1] * camera.height);
                //     auto d_pos = d_cameras[i].position;
                //     debug_image[3 * (yi * camera.width + xi) + 0] += d_pos[0];
                //     debug_image[3 * (yi * camera.width + xi) + 1] += d_pos[0];
                //     debug_image[3 * (yi * camera.width + xi) + 2] += d_pos[0];
                // }
            }
            /////////////////////////////////////////////////////////////////////////////////
        }
    }

    if (scene.use_gpu) {
        cuda_synchronize();
    }
    channel_info.free();
    parallel_cleanup();

#ifdef __NVCC__
    if (old_device_id != -1) {
        checkCuda(cudaSetDevice(old_device_id));
    }
#endif
}
