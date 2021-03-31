#include "pathtracer_was.h"
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
#include "shape_adjacency.h"
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
#include "warp_field.h"

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <fstream>

namespace vfield {
    void init_paths(BufferView<Vector3> throughputs,
                    BufferView<Real> min_roughness,
                    BufferView<Vector3> path_contribs,
                    BufferView<Vector3> nee_contribs,
                    bool use_gpu) {
        DISPATCH(use_gpu, thrust::fill, throughputs.begin(), throughputs.end(), Vector3{1, 1, 1});
        DISPATCH(use_gpu, thrust::fill, min_roughness.begin(), min_roughness.end(), Real(0));
        DISPATCH(use_gpu, thrust::fill, path_contribs.begin(), path_contribs.end(), Vector3{0, 0, 0});
        DISPATCH(use_gpu, thrust::fill, nee_contribs.begin(), nee_contribs.end(), Vector3{0, 0, 0});
    }

    void init_control_variables(BufferView<Vector3> control_mean_grad_contrib,
                    BufferView<Real> control_mean_contrib,
                    BufferView<Matrix3x3> control_sample_covariance,
                    bool use_gpu) {
        DISPATCH(use_gpu, thrust::fill, control_mean_grad_contrib.begin(), control_mean_grad_contrib.end(), Vector3{0, 0, 0});
        DISPATCH(use_gpu, thrust::fill, control_mean_contrib.begin(), control_mean_contrib.end(), Real(0));
        DISPATCH(use_gpu, thrust::fill, control_sample_covariance.begin(), control_sample_covariance.end(), Matrix3x3::zeros());
    }

    struct PathBuffer {
        PathBuffer(int max_bounces,
                int num_pixels,
                int max_aux_samples,
                int num_control_samples,
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
            
            // Vector Field Rendering buffers.
            aux_bsdf_counts = Buffer<uint>(use_gpu, num_pixels);
            aux_bsdf_count_samples = Buffer<AuxCountSample>(use_gpu, num_pixels);
            aux_bsdf_samples = Buffer<AuxSample>(use_gpu, max_aux_samples * num_pixels);
            aux_bsdf_rays = Buffer<Ray>(use_gpu, max_aux_samples * num_pixels);
            aux_bsdf_isects = Buffer<Intersection>(use_gpu, max_aux_samples * num_pixels);
            aux_bsdf_points = Buffer<SurfacePoint>(use_gpu, max_aux_samples * num_pixels);

            aux_nee_counts = Buffer<uint>(use_gpu, num_pixels);
            aux_nee_count_samples = Buffer<AuxCountSample>(use_gpu, num_pixels);
            aux_nee_samples = Buffer<AuxSample>(use_gpu, max_aux_samples * num_pixels);
            aux_nee_rays = Buffer<Ray>(use_gpu, max_aux_samples * num_pixels);
            aux_nee_isects = Buffer<Intersection>(use_gpu, max_aux_samples * num_pixels);
            aux_nee_points = Buffer<SurfacePoint>(use_gpu, max_aux_samples * num_pixels);

            aux_ray_differentials = Buffer<RayDifferential>(use_gpu, max_aux_samples * num_pixels);

            aux_active_pixels = Buffer<int>(use_gpu, max_aux_samples * num_pixels);

            // For RR only..
            aux_bsdf_active_pixels = Buffer<int>(use_gpu, max_aux_samples * num_pixels);
            aux_nee_active_pixels = Buffer<int>(use_gpu, max_aux_samples * num_pixels);

            control_rays = Buffer<Ray>(use_gpu, num_control_samples * num_pixels);
            control_samples = Buffer<CameraSample>(use_gpu, num_control_samples * num_pixels);
            control_isects = Buffer<Intersection>(use_gpu, num_control_samples * num_pixels);
            control_points = Buffer<SurfacePoint>(use_gpu, num_control_samples * num_pixels);
            control_mean_grad_contrib = Buffer<Vector3>(use_gpu, num_pixels);
            control_mean_contrib = Buffer<Real>(use_gpu, num_pixels);
            control_sample_covariance = Buffer<Matrix3x3>(use_gpu, num_pixels);

            // Per-path contribs required to compute second warp component.
            // Note that only the primary-ray contrib is required. The auxillary rays
            // don't have to be path traced.
            primary_contribs = Buffer<Vector3>(use_gpu, (max_bounces + 1) * num_pixels);
            nee_contribs = Buffer<Vector3>(use_gpu, (max_bounces + 1) * num_pixels);
            next_and_scatter_contribs = Buffer<Vector3>(use_gpu, num_pixels); 

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

            d_light_wos = Buffer<Vector3>(use_gpu, num_pixels);
            d_bsdf_wos = Buffer<Vector3>(use_gpu, num_pixels);
            d_camera_samples = Buffer<Vector2>(use_gpu, num_pixels);

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

        // Intermediate derivatives requried for
        // warp derivative computation.
        Buffer<Vector3> d_bsdf_wos;
        Buffer<Vector3> d_light_wos;
        Buffer<Vector2> d_camera_samples;

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

        // Warp Field rendering
        Buffer<uint> aux_bsdf_counts;
        Buffer<uint> aux_nee_counts;

        Buffer<AuxCountSample> aux_bsdf_count_samples;
        Buffer<AuxCountSample> aux_nee_count_samples;

        Buffer<AuxSample> aux_bsdf_samples;
        Buffer<AuxSample> aux_nee_samples;

        Buffer<Ray> aux_bsdf_rays;
        Buffer<Intersection> aux_bsdf_isects;
        Buffer<SurfacePoint> aux_bsdf_points;

        Buffer<Ray> aux_nee_rays;
        Buffer<Intersection> aux_nee_isects;
        Buffer<SurfacePoint> aux_nee_points;

        Buffer<RayDifferential> aux_ray_differentials; // Just a buffer to collect and ignore..

        Buffer<int> aux_active_pixels;

        // For Russian Roulette only..
        Buffer<int> aux_bsdf_active_pixels;
        Buffer<int> aux_nee_active_pixels;

        // Control variate requirements.
        Buffer<Ray> control_rays; // Store control rays for this 
        Buffer<CameraSample> control_samples;
        Buffer<Intersection> control_isects;
        Buffer<SurfacePoint> control_points;
        
        // Accumulate per-pixel mean gradient of contribution to
        // use for the linear-contrib control variate estimation.
        Buffer<Vector3> control_mean_grad_contrib;

        // Accumulate per-pixel mean contribution for use with
        // the linear-velocity control variate estimation.
        Buffer<Real> control_mean_contrib;

        // Accumulate per-pixel sample covariance for use with
        // the linear-velocity control variate estimation.
        Buffer<Matrix3x3> control_sample_covariance;

        Buffer<Vector3> primary_contribs; // Stores per-depth contribs.
        Buffer<Vector3> nee_contribs;     // Stores nee_contrib at each depth.
        Buffer<Vector3> next_and_scatter_contribs; // Used as temp to hold (Sum_(i=d to max_d)(primary_contribs_i)) - nee_contribs_d

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

    template <typename T>
    struct aux_expand_x {
        DEVICE void operator()(int idx) {
            for(int i = 0; i < n; i++)
                to[n * idx + i] = from[idx] * n + i;
        }

        const T *from;
        T *to;
        const int n;
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

    template <typename T>
    struct sum_buffers {
        DEVICE void operator()(int idx) {
            t[active_pixels[idx]] = a[active_pixels[idx]] + b[active_pixels[idx]];
        }

        const int *active_pixels;
        const T *a;
        const T *b;
        T *t;
    };

    template <typename T>
    struct diff_buffers {
        DEVICE void operator()(int idx) {
            t[active_pixels[idx]] = a[active_pixels[idx]] - b[active_pixels[idx]];
        }

        const int *active_pixels;
        const T *a;
        const T *b;
        T *t;
    };

    struct overwrite_ray_dir {
        DEVICE void operator()(int idx) {
            a[idx].dir = b;
        }
        Ray *a;
        const Vector3 b;
    };

    struct is_aux_ray_valid {
        DEVICE bool operator()(int idx) {
            int rr_count = active_counts[(idx / max_rays)];
            return ((idx % max_rays) < rr_count);
        }

        const int max_rays;
        const uint* active_counts;
    };


    struct inverted_bsdf_copier_gaussian {
        DEVICE void operator()(int idx) {

            samples[idx] = BSDFSample{
                Vector2(
                    samples[idx].uv[0], 
                        (samples[idx].uv[1] + 0.5) > 1 ?
                        (samples[idx].uv[1] - 0.5) :
                        (samples[idx].uv[1] + 0.5)
                ),
                samples[idx].w
            };
        }

        BSDFSample *samples;
    };

    void invert_copy_bsdf_samples(BufferView<BSDFSample> &samples,
                                bool use_gpu) {
        // Turn off correlated pair sampling if filter type is not gaussian 
        parallel_for(inverted_bsdf_copier_gaussian{
            samples.begin()},
            samples.size(), use_gpu);
    }

    void update_aux_active_pixels( const BufferView<int> &active_pixels,
                          const BufferView<uint> &active_counts,
                          const int max_aux_rays,
                          BufferView<int> &new_active_pixels,
                          bool use_gpu) {
        auto op = is_aux_ray_valid{max_aux_rays, active_counts.begin()};
        auto new_end = DISPATCH(use_gpu, thrust::copy_if,
            active_pixels.begin(), active_pixels.end(),
            new_active_pixels.begin(), op);
        new_active_pixels.count = new_end - new_active_pixels.begin();
    }

    void render(Scene &scene,
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

        KernelParameters kernel_parameters = options.kernel_parameters;
        const int numAuxillaryRays = kernel_parameters.numAuxillaryRays;
        const int numControlRays = options.variance_reduction_mode.num_control_samples;

        // Some common variables
        const auto &camera = scene.camera;
        auto num_pixels = camera.width * camera.height;
        auto max_bounces = options.max_bounces;

        // Maintain an edge adjacency list requried for weight computation.
        std::vector<ShapeAdjacency> adjacencies(scene.shapes.size());
        for(int i = 0; i < adjacencies.size(); i++) {
            compute_adjacency(&scene.shapes[i], &adjacencies.at(i));
        }

        // A main difference between our path tracer and the sual path
        // tracer is that we need to store all the intermediate states
        // for later computation of derivatives.
        // Therefore we allocate a big buffer here for the storage.
        PathBuffer path_buffer(max_bounces, num_pixels, numAuxillaryRays, numControlRays, scene.use_gpu, channel_info);
        auto num_active_pixels = std::vector<int>((max_bounces + 1) * num_pixels, 0);
        std::unique_ptr<Sampler> sampler, edge_sampler, aux_sampler;
        switch (options.sampler_type) {
            case SamplerType::independent: {
                sampler = std::unique_ptr<Sampler>(new PCGSampler(scene.use_gpu, options.seed, num_pixels));
                edge_sampler = std::unique_ptr<Sampler>(
                    new PCGSampler(scene.use_gpu, options.seed + 131071U, num_pixels));
                aux_sampler = std::unique_ptr<Sampler>(
                    new PCGSampler(scene.use_gpu, options.seed + 131071U, num_pixels * numAuxillaryRays));
                break;
            } case SamplerType::sobol: {
                sampler = std::unique_ptr<Sampler>(new SobolSampler(scene.use_gpu, options.seed, num_pixels));
                edge_sampler = std::unique_ptr<Sampler>(
                    new SobolSampler(scene.use_gpu, options.seed + 131071U, num_pixels));
                aux_sampler = std::unique_ptr<Sampler>(
                    new SobolSampler(scene.use_gpu, options.seed + 131071U, num_pixels * numAuxillaryRays));
                break;
            } default: {
                assert(false);
                break;
            }
        }
        auto optix_rays = path_buffer.optix_rays.view(0, 2 * num_pixels);
        auto optix_hits = path_buffer.optix_hits.view(0, 2 * num_pixels);

        ThrustCachedAllocator thrust_alloc(scene.use_gpu, num_pixels * sizeof(int));

        auto control_mean_grad_contrib = path_buffer.control_mean_grad_contrib.view(0, num_pixels);
        auto control_mean_contrib = path_buffer.control_mean_contrib.view(0, num_pixels);
        auto control_sample_covariance = path_buffer.control_sample_covariance.view(0, num_pixels);

        init_control_variables(control_mean_grad_contrib, control_mean_contrib, control_sample_covariance, scene.use_gpu);

        std::vector<bool> depths_used(max_bounces, false);
        depths_used = std::vector<bool>(max_bounces, false);
        // For each sample
        for (int sample_id = 0; sample_id < options.num_samples; sample_id++) {
            
            sampler->begin_sample(sample_id);

            //std::cout << "Sample " << sample_id << std::endl;

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

            auto _path_contribs = path_buffer.primary_contribs.view(0, num_pixels * (max_bounces + 1));
            auto _nee_contribs = path_buffer.nee_contribs.view(0, num_pixels * max_bounces);

            // Initialization
            init_paths(throughputs, min_roughness, _path_contribs, _nee_contribs, scene.use_gpu);

            // Buffer for primary contribution.
            auto path_contribs = path_buffer.primary_contribs.view(0, num_pixels);

            // Generate primary ray samples
            if (!options.variance_reduction_mode.secondary_antithetic_variates){
                if (options.variance_reduction_mode.primary_antithetic_variates) {
                    if(sample_id % 2 == 0)
                        sampler->next_camera_samples(camera_samples, options.sample_pixel_center);
                    else
                        invert_copy_camera_samples(camera, camera_samples, scene.use_gpu);
                } else {
                    sampler->next_camera_samples(camera_samples, options.sample_pixel_center);
                }
            } else {
                if (options.variance_reduction_mode.primary_antithetic_variates) {
                    if((sample_id / 2) % 2 == 0) {
                        if(sample_id % 2 == 0)
                            sampler->next_camera_samples(camera_samples, options.sample_pixel_center);
                    } else {
                        // Do nothing..
                        if(sample_id % 2 == 0)
                            invert_copy_camera_samples(camera, camera_samples, scene.use_gpu);
                    }
                } else {
                    sampler->next_camera_samples(camera_samples, options.sample_pixel_center);
                }
            }

            
            sample_primary_rays(camera, camera_samples, rays, primary_differentials, scene.use_gpu);
            // Initialize pixel id
            init_active_pixels(rays, primary_active_pixels, scene.use_gpu, thrust_alloc);
            auto num_actives_primary = (int)primary_active_pixels.size();
            //std::cout << "Num actives primary: " << num_actives_primary << std::endl;
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
                                        BufferView<Real>(),
                                        path_contribs,
                                        generic_texture_buffer); // edge_contrib

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
                auto primary_contribs = path_buffer.primary_contribs.view((depth + 1) * num_pixels, num_pixels);
                auto nee_contribs = path_buffer.nee_contribs.view((depth) * num_pixels, num_pixels);

                // Sample points on lights
                if (!options.variance_reduction_mode.secondary_antithetic_variates) {
                    if (options.variance_reduction_mode.primary_antithetic_variates) {
                        // Don't sample a new set of Light samples on odd numbered samples.
                        if(sample_id % 2 == 0 || !depths_used[depth]) {
                            sampler->next_light_samples(light_samples);
                        } else {
                            // do nothing.
                            // This enables an (almost) correlated path to be sampled.. 
                            // leading to sharply reduced variance in the kernel sampling..
                        }
                    } else {
                        sampler->next_light_samples(light_samples);
                    }
                } else {
                    if (options.variance_reduction_mode.primary_antithetic_variates) {
                        // Don't sample a new set of Light samples on odd numbered samples.
                        if(((sample_id % 2 == 0) && ((sample_id/2) % 2 == 0)) || !depths_used[depth]) {
                            sampler->next_light_samples(light_samples);
                        } else {
                            // do nothing.
                            // This enables an (almost) correlated path to be sampled.. 
                            // leading to sharply reduced variance in the kernel sampling..
                        }
                    } else {
                        sampler->next_light_samples(light_samples);
                    }
                }

                sample_point_on_light(scene,
                                    active_pixels,
                                    shading_points,
                                    light_samples,
                                    light_isects,
                                    light_points,
                                    nee_rays);
                occluded(scene, active_pixels, nee_rays, optix_rays, optix_hits);

                if (!options.variance_reduction_mode.secondary_antithetic_variates) {
                    if (options.variance_reduction_mode.primary_antithetic_variates) {
                        // Don't sample a new set of BSDF samples on odd numbered samples.
                        if(sample_id % 2 == 0 || !depths_used[depth]){
                            sampler->next_bsdf_samples(bsdf_samples); 
                        } else {
                            // do nothing.
                            // This enables an (almost) correlated path to be sampled.. 
                            // leading to sharply reduced variance in the kernel sampling..
                        }
                    } else {
                        sampler->next_bsdf_samples(bsdf_samples);
                    }
                } else {
                    if (options.variance_reduction_mode.primary_antithetic_variates) {
                        // Don't sample a new set of BSDF samples on odd numbered samples.
                        if( (((sample_id / 2) % 2 == 0) && ((sample_id) % 2 == 0)) ) {// || !depths_used[depth]) {
                            sampler->next_bsdf_samples(bsdf_samples); 
                        } else {
                            invert_copy_bsdf_samples(bsdf_samples, scene.use_gpu); 
                        }
                    } else {
                        sampler->next_bsdf_samples(bsdf_samples);
                    }
                }

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
                    BufferView<Real>(),
                    primary_contribs,
                    nee_contribs);

                // Stream compaction: remove invalid bsdf intersections
                // active_pixels -> next_active_pixels
                update_active_pixels(active_pixels, bsdf_isects, next_active_pixels, scene.use_gpu); 

                // Record the number of active pixels for next depth
                num_active_pixels[depth + 1] = next_active_pixels.size();
                
            }

            if (d_rendered_image.get() != nullptr) {

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
                    const auto prev_active_pixels = (depth != 0) ?
                        path_buffer.active_pixels.view((depth - 1) * num_pixels, num_active_pixels[depth]) :
                        path_buffer.primary_active_pixels.view(0, num_actives_primary);
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
                    // Holds the total contrib from each depth.
                    auto primary_contribs = 
                        path_buffer.primary_contribs.view((depth + 1) * num_pixels, num_pixels);
                    
                    auto nee_contribs = path_buffer.nee_contribs.view((depth) * num_pixels, num_pixels);
                    
                    auto next_and_scatter_contribs = path_buffer.next_and_scatter_contribs.view(0, num_pixels);
                    
                    auto prev_primary_contribs = 
                        path_buffer.primary_contribs.view((depth) * num_pixels, num_pixels);

                    auto aux_nee_counts = path_buffer.aux_nee_counts.view(0, num_pixels);
                    auto aux_bsdf_counts = path_buffer.aux_bsdf_counts.view(0, num_pixels);
                    auto aux_nee_count_samples = path_buffer.aux_nee_count_samples.view(0, num_pixels);
                    auto aux_bsdf_count_samples = path_buffer.aux_bsdf_count_samples.view(0, num_pixels);
                    auto aux_nee_samples = path_buffer.aux_nee_samples.view(0, numAuxillaryRays * num_pixels);
                    auto aux_bsdf_samples = path_buffer.aux_bsdf_samples.view(0, numAuxillaryRays * num_pixels);
                    auto aux_nee_rays = path_buffer.aux_nee_rays.view(0, numAuxillaryRays * num_pixels);
                    auto aux_bsdf_rays = path_buffer.aux_bsdf_rays.view(0, numAuxillaryRays * num_pixels);
                    auto aux_nee_isects = path_buffer.aux_nee_isects.view(0, numAuxillaryRays * num_pixels);
                    auto aux_bsdf_isects = path_buffer.aux_bsdf_isects.view(0, numAuxillaryRays * num_pixels);
                    auto aux_nee_points = path_buffer.aux_nee_points.view(0, numAuxillaryRays * num_pixels);
                    auto aux_bsdf_points = path_buffer.aux_bsdf_points.view(0, numAuxillaryRays * num_pixels);
                    auto aux_ray_differentials = path_buffer.aux_ray_differentials.view(0, numAuxillaryRays * num_pixels);
                    auto aux_active_pixels = path_buffer.aux_active_pixels.view(0, numAuxillaryRays * num_pixels);
                    
                    // RR-specific buffers.
                    auto aux_bsdf_active_pixels = path_buffer.aux_bsdf_active_pixels.view(0, numAuxillaryRays * num_pixels);
                    auto aux_nee_active_pixels = path_buffer.aux_nee_active_pixels.view(0, numAuxillaryRays * num_pixels);

                    auto d_throughputs = path_buffer.d_throughputs.view(0, num_pixels);
                    auto d_rays = path_buffer.d_rays.view(0, num_pixels);
                    auto d_ray_differentials = path_buffer.d_ray_differentials.view(0, num_pixels);
                    auto d_points = path_buffer.d_points.view(0, num_pixels);
                    auto d_bsdf_wos = path_buffer.d_bsdf_wos.view(0, num_pixels);
                    auto d_light_wos = path_buffer.d_light_wos.view(0, num_pixels);

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
                        
                        DISPATCH(scene.use_gpu, thrust::fill,
                            d_ray_differentials.begin(), d_ray_differentials.end(),
                            RayDifferential{Vector3{0, 0, 0}, Vector3{0, 0, 0},
                                            Vector3{0, 0, 0}, Vector3{0, 0, 0}});

                        DISPATCH(scene.use_gpu, thrust::fill,
                            d_points.begin(), d_points.end(), SurfacePoint::zero());

                        DISPATCH(scene.use_gpu, thrust::fill,
                            d_rays.begin(), d_rays.end(), DRay{});                        
                    }
                    

                    // Backpropagate central derivative
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
                        d_points,
                        d_bsdf_wos,
                        d_light_wos);

                    if (options.enable_secondary_warp_field) {
                        ////////////////////////////////////////////////////////////////////////////////
                        // Subtract nee_contrib from primary_contrib to form
                        // next_and_scatter_contrib. This quantity tracks the contribution just 
                        // from bsdf_ray.dir. This separation allows the computation of warp
                        // derivatives efficiently.
                        parallel_for(diff_buffers<Vector3>{
                            active_pixels.begin(),
                            primary_contribs.begin(),
                            nee_contribs.begin(),
                            next_and_scatter_contribs.begin()}, active_pixels.size(), scene.use_gpu);

                        // Sample points to compute the local field around bsdf rays
                        aux_sampler->next_aux_samples(aux_bsdf_samples);
                        // Sample points to compute the local field around nee rays
                        aux_sampler->next_aux_samples(aux_nee_samples);

                        if (options.variance_reduction_mode.aux_antithetic_variates) {
                            // Overwrite one half of the samples with the inverse of the other half.
                            aux_generate_correlated_pairs(kernel_parameters, 
                                active_pixels, aux_bsdf_samples, scene.use_gpu);
                            aux_generate_correlated_pairs(kernel_parameters,
                                active_pixels, aux_nee_samples, scene.use_gpu);
                        }

                        if ( kernel_parameters.rr_enabled ) {
                            // Sample the number of samples for Russian roulette.
                            // More info on the generalized Russian roulette de-biasing method here:
                            // https://arxiv.org/abs/1005.2228
                            aux_sampler->next_aux_count_samples(aux_bsdf_count_samples);
                            aux_sampler->next_aux_count_samples(aux_nee_count_samples);

                            aux_sample_sample_counts( kernel_parameters,
                                    scene,
                                    active_pixels,
                                    aux_bsdf_count_samples,
                                    aux_bsdf_counts
                            );

                            aux_sample_sample_counts( kernel_parameters,
                                    scene,
                                    active_pixels,
                                    aux_nee_count_samples,
                                    aux_nee_counts
                            );

                        } else {
                            DISPATCH(scene.use_gpu, 
                                    thrust::fill, 
                                    aux_nee_counts.begin(),
                                    aux_nee_counts.end(),
                                    static_cast<uint>(numAuxillaryRays));
                            DISPATCH(scene.use_gpu, 
                                    thrust::fill, 
                                    aux_bsdf_counts.begin(),
                                    aux_bsdf_counts.end(),
                                    static_cast<uint>(numAuxillaryRays));
                        }

                        // Compute perturbed ray bundle around the bsdf ray
                        aux_bundle_sample( kernel_parameters,
                                    scene,
                                    active_pixels,
                                    shading_points,
                                    incoming_rays,
                                    shading_isects,
                                    next_rays,
                                    aux_bsdf_counts,
                                    aux_bsdf_samples,
                                    aux_bsdf_rays);

                        // Compute perturbed ray bundle around the nee ray
                        aux_bundle_sample( kernel_parameters,
                                    scene,
                                    active_pixels,
                                    shading_points,
                                    incoming_rays,
                                    shading_isects,
                                    nee_rays,
                                    aux_nee_counts,
                                    aux_nee_samples,
                                    aux_nee_rays);

                        // Populate aux_active_pixels by tiling active_pixels.
                        parallel_for(aux_expand_x<int>{
                            active_pixels.begin(),
                            aux_active_pixels.begin(),
                            kernel_parameters.numAuxillaryRays}, active_pixels.size(), scene.use_gpu);
    
                        aux_active_pixels.count = 
                            active_pixels.count * kernel_parameters.numAuxillaryRays;
                        
                        if( kernel_parameters.rr_enabled ){
                            // Updates active-flag for auxiliary rays based on
                            // Russian roulette stopping condition.
                            update_aux_active_pixels(aux_active_pixels,
                                                    aux_bsdf_counts,
                                                    kernel_parameters.numAuxillaryRays,
                                                    aux_bsdf_active_pixels,
                                                    scene.use_gpu);
                            update_aux_active_pixels(aux_active_pixels,
                                                    aux_nee_counts,
                                                    kernel_parameters.numAuxillaryRays,
                                                    aux_nee_active_pixels,
                                                    scene.use_gpu);
                        }

                        // Intersect the two bundles.
                        intersect(scene, // in
                            (kernel_parameters.rr_enabled) ? aux_bsdf_active_pixels : aux_active_pixels,
                            aux_bsdf_rays,
                            aux_ray_differentials, // out
                            aux_bsdf_isects,
                            aux_bsdf_points,
                            aux_ray_differentials,
                            optix_rays,
                            optix_hits);

                        intersect(scene, // in
                            (kernel_parameters.rr_enabled) ? aux_nee_active_pixels : aux_active_pixels,
                            aux_nee_rays,
                            aux_ray_differentials, // out
                            aux_nee_isects,
                            aux_nee_points,
                            aux_ray_differentials,
                            optix_rays,
                            optix_hits);

                        // Compute and accumulate the 
                        // harmonic warp derivative for
                        accumulate_warp_derivatives(scene,
                            active_pixels,
                            shading_points,
                            shading_isects,

                            next_rays,   // primary_rays
                            bsdf_isects, // primary_isects
                            bsdf_points, // primary_points

                            aux_bsdf_counts,
                            aux_bsdf_rays,
                            aux_bsdf_isects,
                            aux_bsdf_points,
                            aux_bsdf_samples,

                            next_and_scatter_contribs,

                            d_bsdf_wos, // d_out (in)
                            Real(1) / options.num_samples,
                            channel_info,
                            kernel_parameters,
                            d_rendered_image.get(), // per-pixel loss weights.
                            options.variance_reduction_mode.aux_control_variates,
                            d_points, // d_in (out)
                            d_scene->shapes.view(0, d_scene->shapes.size()), // d_in (out)

                            BufferView<CameraSample>(), // No bounds on spherical domain.
                            BufferView<Vector2>(),      // No camera sample derivatives.
                            &adjacencies.at(0),
                            d_scene.get(),
                            debug_image.get(),
                            nullptr,

                            BufferView<Vector3>(),
                            BufferView<Real>(),
                            BufferView<Matrix3x3>()
                        );

                        accumulate_warp_derivatives(scene,
                            active_pixels,
                            shading_points,
                            shading_isects,

                            nee_rays,   // primary_rays
                            light_isects, // primary_isects
                            light_points, // primary_points

                            aux_nee_counts,
                            aux_nee_rays,
                            aux_nee_isects,
                            aux_nee_points,
                            aux_nee_samples,

                            nee_contribs,

                            d_light_wos, // d_out (in)
                            Real(1) / options.num_samples,
                            channel_info,
                            kernel_parameters,
                            d_rendered_image.get(), // per-pixel loss weights.
                            options.variance_reduction_mode.aux_control_variates,
                            d_points, // d_in (out)
                            d_scene->shapes.view(0, d_scene->shapes.size()), // d_in (out)

                            BufferView<CameraSample>(), // No bounds on spherical domain.
                            BufferView<Vector2>(),      // No camera sample derivatives.
                            &adjacencies.at(0),
                            d_scene.get(),
                            debug_image.get(),
                            nullptr,

                            BufferView<Vector3>(),
                            BufferView<Real>(),
                            BufferView<Matrix3x3>()
                        );
                        //}

                        // Sum this and previous primary_contrib buffers to accumulate
                        // contributions from the deepest till this depth.
                        // This quantity tracks the contribution that is affected by changes to
                        // bsdf_ray.dir AND nee_ray.dir
                        parallel_for(sum_buffers<Vector3>{
                            prev_active_pixels.begin(),
                            prev_primary_contribs.begin(), // A
                            primary_contribs.begin(),      // B
                            prev_primary_contribs.begin()}, // A + B
                            prev_active_pixels.size(), scene.use_gpu);

                        // Problem with regions outside active-pixels..

                    }
    
                    // Previous become next
                    std::swap(path_buffer.d_next_throughputs, path_buffer.d_throughputs);
                    std::swap(path_buffer.d_next_rays, path_buffer.d_rays);
                    std::swap(path_buffer.d_next_ray_differentials, path_buffer.d_ray_differentials);
                    std::swap(path_buffer.d_next_points, path_buffer.d_points);
                    depths_used[depth] = true;
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
                    auto d_bsdf_wos = path_buffer.d_bsdf_wos.view(0, num_pixels);
                    auto d_camera_samples = path_buffer.d_camera_samples.view(0, num_pixels);

                    auto aux_bsdf_counts = path_buffer.aux_bsdf_counts.view(0, num_pixels);
                    auto aux_bsdf_count_samples = path_buffer.aux_bsdf_count_samples.view(0, num_pixels);
                    auto aux_bsdf_samples = path_buffer.aux_bsdf_samples.view(0, numAuxillaryRays * num_pixels);
                    auto aux_bsdf_points = path_buffer.aux_bsdf_points.view(0, numAuxillaryRays * num_pixels);
                    auto aux_bsdf_rays = path_buffer.aux_bsdf_rays.view(0, numAuxillaryRays * num_pixels);
                    auto aux_bsdf_isects = path_buffer.aux_bsdf_isects.view(0, numAuxillaryRays * num_pixels);
                    auto aux_ray_differentials = path_buffer.aux_ray_differentials.view(0, numAuxillaryRays * num_pixels);
                    auto aux_active_pixels = path_buffer.aux_active_pixels.view(0, numAuxillaryRays * num_pixels);
                    auto aux_bsdf_active_pixels = path_buffer.aux_bsdf_active_pixels.view(0, numAuxillaryRays * num_pixels);

                    auto control_mean_grad_contrib = path_buffer.control_mean_grad_contrib.view(0, num_pixels);
                    auto control_mean_contrib = path_buffer.control_mean_contrib.view(0, num_pixels);
                    auto control_sample_covariance = path_buffer.control_sample_covariance.view(0, num_pixels);

                    auto bsdf_samples = path_buffer.bsdf_samples.view(0, num_pixels);

                    auto all_contribs = path_buffer.primary_contribs.view(0, num_pixels);

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
                                                d_points,
                                                d_bsdf_wos);

                    // Propagate to camera.
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
                                        d_camera_samples,
                                        debug_image.get(),
                                        screen_gradient_image.get());

                    // Compute primary edge contribution using warp fields.
                    if( options.enable_primary_warp_field ) {
                        // Sample points to compute the local field around bsdf rays
                        aux_sampler->next_aux_samples(aux_bsdf_samples);
                        if (options.variance_reduction_mode.aux_antithetic_variates) {
                            // Overwrite one half of the samples with the inverse of the other half.
                            aux_generate_correlated_pairs(kernel_parameters,
                                primary_active_pixels, aux_bsdf_samples, scene.use_gpu);
                        }

                        if ( kernel_parameters.rr_enabled ) {
                            // Sample RR stopping points.
                            aux_sampler->next_aux_count_samples(aux_bsdf_count_samples);

                            aux_sample_sample_counts( kernel_parameters,
                                    scene,
                                    primary_active_pixels,
                                    aux_bsdf_count_samples,
                                    aux_bsdf_counts
                            );

                        } else {
                            DISPATCH(scene.use_gpu, 
                                    thrust::fill, 
                                    aux_bsdf_counts.begin(),
                                    aux_bsdf_counts.end(),
                                    static_cast<uint>(numAuxillaryRays));
                        }

                        // Compute perturbed ray bundle around the bsdf ray
                        // TODO: Add an option for partial sampling..
                        aux_bundle_sample_primary(kernel_parameters,
                                    scene,
                                    primary_active_pixels,
                                    aux_bsdf_counts,
                                    rays,
                                    camera_samples,
                                    aux_bsdf_samples,
                                    aux_bsdf_rays);

                        // Populate aux_active_pixels by tiling active_pixels.
                        parallel_for(aux_expand_x<int>{
                            primary_active_pixels.begin(),
                            aux_active_pixels.begin(),
                            numAuxillaryRays}, primary_active_pixels.size(), scene.use_gpu);
                        aux_active_pixels.count = primary_active_pixels.count * numAuxillaryRays;

                        if( kernel_parameters.rr_enabled ){
                            update_aux_active_pixels(aux_active_pixels, 
                                                    aux_bsdf_counts, 
                                                    kernel_parameters.numAuxillaryRays,
                                                    aux_bsdf_active_pixels,
                                                    scene.use_gpu);
                        }

                        //  Intersect the bundle.
                        intersect(scene, // in
                            kernel_parameters.rr_enabled ? aux_bsdf_active_pixels : aux_active_pixels,
                            aux_bsdf_rays,
                            aux_ray_differentials, // out
                            aux_bsdf_isects,
                            aux_bsdf_points,
                            aux_ray_differentials,
                            optix_rays,
                            optix_hits);

                        accumulate_warp_derivatives(scene,
                                primary_active_pixels,
                                BufferView<SurfacePoint>(),
                                BufferView<Intersection>(),

                                rays,   // primary_rays
                                shading_isects, // primary_isects
                                shading_points, // primary_points

                                aux_bsdf_counts,
                                aux_bsdf_rays,
                                aux_bsdf_isects,
                                aux_bsdf_points,
                                aux_bsdf_samples,

                                all_contribs, // All accumulated path contribs.

                                BufferView<Vector3>(), // d_out (in)
                                Real(1) / options.num_samples,
                                channel_info,
                                kernel_parameters,
                                d_rendered_image.get(), // per-pixel loss weights.
                                options.variance_reduction_mode.aux_control_variates,
                                BufferView<SurfacePoint>(), // d_in (out) [No points for primary isect.]
                                d_scene->shapes.view(0, d_scene->shapes.size()), // d_in (out)

                                camera_samples, // Used to account for pixel boundary terms.
                                d_camera_samples, // Derivatives propagated from the previous step.
                                &adjacencies.at(0),
                                d_scene.get(),
                                debug_image.get(),
                                screen_gradient_image.get(),
                                control_mean_grad_contrib,
                                control_mean_contrib,
                                control_sample_covariance
                        );

                    }
                }

            }
        }

        auto primary_active_pixels = path_buffer.primary_active_pixels.view(0, num_pixels);
        auto primary_differentials = path_buffer.primary_ray_differentials.view(0, num_pixels);

        // Add control variate sampling.
        if (options.enable_primary_warp_field && d_scene.get() != nullptr) {
            for (int i = 0; options.variance_reduction_mode.primary_control_variates && i < numControlRays; i++) {
                auto control_rays = path_buffer.control_rays.view(i * num_pixels, num_pixels);
                auto control_samples = path_buffer.control_samples.view(i * num_pixels, num_pixels);
                auto control_isects = path_buffer.control_isects.view(i * num_pixels, num_pixels);
                auto control_points = path_buffer.control_points.view(i * num_pixels, num_pixels);

                if (numControlRays > 1) {
                    sampler->next_camera_samples(control_samples, options.sample_pixel_center);
                } else {
                    // Single sample mode.
                    DISPATCH(scene.use_gpu, 
                            thrust::fill, 
                            control_samples.begin(), 
                            control_samples.end(), 
                            CameraSample{Vector2{1.0, 0.0}});
                }

                sample_primary_rays(camera, control_samples, control_rays, primary_differentials, scene.use_gpu);

                init_active_pixels(control_rays, primary_active_pixels, scene.use_gpu, thrust_alloc);

                auto num_active_control_pixels = (int)primary_active_pixels.size();

                intersect(scene,
                        primary_active_pixels,
                        control_rays,
                        primary_differentials,
                        control_isects,
                        control_points,
                        primary_differentials,
                        optix_rays,
                        optix_hits);

                accumulate_control_variates(scene,
                    options.kernel_parameters,
                    primary_active_pixels,
                    control_points,
                    control_isects,
                    control_rays,
                    control_samples,

                    control_mean_grad_contrib,
                    control_mean_contrib,
                    control_sample_covariance,

                    d_scene->shapes.view(0, d_scene->shapes.size()),

                    Real(1) / options.num_samples,
                    debug_image.get()
                );
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

}
