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

        primary_edge_samples = Buffer<PrimaryEdgeSample>(use_gpu, num_pixels);
        secondary_edge_samples = Buffer<SecondaryEdgeSample>(use_gpu, num_pixels);
        primary_edge_records = Buffer<PrimaryEdgeRecord>(use_gpu, num_pixels);
        secondary_edge_records = Buffer<SecondaryEdgeRecord>(use_gpu, num_pixels);
        edge_contribs = Buffer<Real>(use_gpu, 2 * num_pixels);
        edge_surface_points = Buffer<Vector3>(use_gpu, 2 * num_pixels);

        tmp_light_samples = Buffer<LightSample>(use_gpu, num_pixels);
        tmp_bsdf_samples = Buffer<BSDFSample>(use_gpu, num_pixels);

        generic_texture_buffer = Buffer<Real>(use_gpu,
            channel_info.max_generic_texture_dimension * num_pixels);
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

    // For temporary storing generic texture values per thread
    Buffer<Real> generic_texture_buffer;
};

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
            ptr<float> screen_gradient_image,
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
    ChannelInfo channel_info(options.channels,
                             scene.use_gpu,
                             scene.max_generic_texture_dimension);

    // Some common variables
    const auto &camera = scene.camera;
    auto num_pixels = camera.width * camera.height;
    auto max_bounces = options.max_bounces;

    // A main difference between our path tracer and the usual path
    // tracer is that we need to store all the intermediate states
    // for later computation of derivatives.
    // Therefore we allocate a big buffer here for the storage.
    PathBuffer path_buffer(max_bounces,
                           num_pixels,
                           scene.use_gpu,
                           channel_info);
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

    ThrustCachedAllocator thrust_alloc(scene.use_gpu, num_pixels * sizeof(int));

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
        auto generic_texture_buffer =
            path_buffer.generic_texture_buffer.view(0, path_buffer.generic_texture_buffer.size());

        // Initialization
        init_paths(throughputs, min_roughness, scene.use_gpu);
        // Generate primary ray samples
        sampler->next_camera_samples(camera_samples, options.sample_pixel_center);
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
                                    BufferView<Real>(), // edge_contrib
                                    generic_texture_buffer);
        // Stream compaction: remove invalid intersection
        update_active_pixels(primary_active_pixels, shading_isects, active_pixels, scene.use_gpu);
        std::fill(num_active_pixels.begin(), num_active_pixels.end(), 0);
        num_active_pixels[0] = active_pixels.size();
        for (int depth = 0; depth < max_bounces && num_active_pixels[depth] > 0 && has_lights(scene); depth++) {
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

            // Initialize the derivatives for path vertices
            auto d_throughputs = path_buffer.d_throughputs.view(0, num_pixels);
            auto d_rays = path_buffer.d_rays.view(0, num_pixels);
            auto d_ray_differentials = path_buffer.d_ray_differentials.view(0, num_pixels);
            auto d_points = path_buffer.d_points.view(0, num_pixels);
            auto d_next_throughputs = path_buffer.d_next_throughputs.view(0, num_pixels);
            auto d_next_rays = path_buffer.d_next_rays.view(0, num_pixels);
            auto d_next_ray_differentials =
                path_buffer.d_next_ray_differentials.view(0, num_pixels);
            auto d_next_points = path_buffer.d_next_points.view(0, num_pixels);
            DISPATCH(scene.use_gpu, thrust::fill,
                d_throughputs.begin(), d_throughputs.end(),
                Vector3{0, 0, 0});
            DISPATCH(scene.use_gpu, thrust::fill,
                d_rays.begin(), d_rays.end(), DRay{});
            DISPATCH(scene.use_gpu, thrust::fill,
                d_ray_differentials.begin(), d_ray_differentials.end(),
                RayDifferential{Vector3{0, 0, 0}, Vector3{0, 0, 0},
                                Vector3{0, 0, 0}, Vector3{0, 0, 0}});
            DISPATCH(scene.use_gpu, thrust::fill,
                d_points.begin(), d_points.end(),
                SurfacePoint::zero());
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

            // Traverse the path backward for the derivatives
            for (int depth = max_bounces - 1; depth >= 0 && has_lights(scene); depth--) {
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
                    d_scene.get(),
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
                        edge_contribs,
                        generic_texture_buffer);
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
                    accumulate_secondary_edge_derivatives(scene,
                                                          active_pixels,
                                                          shading_points,
                                                          edge_records,
                                                          edge_surface_points,
                                                          edge_contribs,
                                                          d_points,
                                                          d_scene->shapes.view(0, d_scene->shapes.size()));
                    ////////////////////////////////////////////////////////////////////////////////
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
                                              generic_texture_buffer,
                                              d_scene.get(),
                                              d_rays,
                                              d_ray_differentials,
                                              d_points);
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
                                       d_scene.get(),
                                       screen_gradient_image.get());
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
                                            nullptr, // rendered_image
                                            edge_contribs,
                                            generic_texture_buffer);
                // Stream compaction: remove invalid intersections
                update_active_pixels(active_pixels, shading_isects, active_pixels, scene.use_gpu);
                auto active_pixels_size = active_pixels.size();
                for (int depth = 0; depth < max_bounces && active_pixels_size > 0 && has_lights(scene); depth++) {
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
                compute_primary_edge_derivatives(
                    scene, edge_records, edge_contribs,
                    d_scene->shapes.view(0, d_scene->shapes.size()),
                    d_scene->camera,
                    screen_gradient_image.get());
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
