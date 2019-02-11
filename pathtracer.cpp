#include "pathtracer.h"
#include "scene.h"
#include "sampler.h"
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

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

void init_paths(BufferView<Vector3> throughputs,
                BufferView<Real> min_roughness,
                bool use_gpu) {
    DISPATCH(use_gpu, thrust::fill,
             throughputs.begin(),
             throughputs.end(),
             Vector3{1, 1, 1});
    DISPATCH(use_gpu, thrust::fill,
             min_roughness.begin(),
             min_roughness.end(),
             Real(0));
}

struct d_primary_intersector {
    DEVICE void operator()(int idx) {
        // Initialize derivatives
        auto d_primary_v = d_vertices + 3 * idx;
        d_primary_v[0] = DVertex{};
        d_primary_v[1] = DVertex{};
        d_primary_v[2] = DVertex{};
        auto &d_camera = d_cameras[idx];
        d_camera = DCameraInst{};

        auto pixel_idx = active_pixels[idx];
        const auto &isect = isects[pixel_idx];
        auto d_primary_ray_differential = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}
        };
        auto d_ray = d_rays[pixel_idx];

        if (isect.valid()) {
            auto shape_id = isect.shape_id;
            auto tri_id = isect.tri_id;
            const auto &shape = shapes[shape_id];
            auto ind = get_indices(shape, tri_id);
            d_primary_v[0].shape_id = shape_id;
            d_primary_v[0].vertex_id = ind[0];
            d_primary_v[1].shape_id = shape_id;
            d_primary_v[1].vertex_id = ind[1];
            d_primary_v[2].shape_id = shape_id;
            d_primary_v[2].vertex_id = ind[2];
            d_intersect_shape(shape,
                              tri_id,
                              rays[pixel_idx],
                              primary_ray_differentials[pixel_idx],
                              d_points[pixel_idx],
                              d_ray_differentials[pixel_idx],
                              d_ray,
                              d_primary_ray_differential,
                              d_primary_v);
        }

        auto pixel_x = pixel_idx % camera.width;
        auto pixel_y = pixel_idx / camera.width;
        auto sample = samples[pixel_idx].xy;
        auto screen_pos = Vector2{
            (pixel_x + sample[0]) / Real(camera.width),
            (pixel_y + sample[1]) / Real(camera.height)
        };

        // Ray differential computation
        auto delta = Real(1e-3);
        auto screen_pos_dx = screen_pos + Vector2{delta, Real(0)};
        // auto ray_dx = sample_primary(camera, screen_pos_dx);
        auto screen_pos_dy = screen_pos + Vector2{Real(0), delta};
        // auto ray_dy = sample_primary(camera, screen_pos_dx);
        auto pixel_size_x = Real(0.5) / camera.width;
        auto pixel_size_y = Real(0.5) / camera.height;
        // auto org_dx = pixel_size_x * (ray_dx.org - ray.org) / delta;
        // auto org_dy = pixel_size_y * (ray_dy.org - ray.org) / delta;
        // auto dir_dx = pixel_size_x * (ray_dx.dir - ray.dir) / delta;
        // auto dir_dy = pixel_size_y * (ray_dy.dir - ray.dir) / delta;
        // ray_differentials[idx] = RayDifferential{org_dx, org_dy, dir_dx, dir_dy}
        auto d_ray_dx = DRay{d_primary_ray_differential.org_dx * pixel_size_x / delta,
                             d_primary_ray_differential.dir_dx * pixel_size_x / delta};
        auto d_ray_dy = DRay{d_primary_ray_differential.org_dy * pixel_size_y / delta,
                             d_primary_ray_differential.dir_dy * pixel_size_y / delta};
        d_ray.org += (d_primary_ray_differential.org_dx * -pixel_size_x +
                      d_primary_ray_differential.org_dx * -pixel_size_y) / delta;
        d_ray.dir += (d_primary_ray_differential.dir_dx * -pixel_size_x +
                      d_primary_ray_differential.dir_dx * -pixel_size_y) / delta;

        d_sample_primary_ray(camera, screen_pos, d_ray, d_camera);
        d_sample_primary_ray(camera, screen_pos_dx, d_ray_dx, d_camera);
        d_sample_primary_ray(camera, screen_pos_dy, d_ray_dy, d_camera);
    }

    const Camera camera;
    const Shape *shapes;
    const int *active_pixels;
    const CameraSample *samples;
    const Ray *rays;
    const RayDifferential *primary_ray_differentials;
    const Intersection *isects;
    const DRay *d_rays;
    const RayDifferential *d_ray_differentials;
    const SurfacePoint *d_points;
    DVertex *d_vertices;
    DCameraInst *d_cameras;
};

void d_primary_intersection(const Scene &scene,
                            const BufferView<int> &active_pixels,
                            const BufferView<CameraSample> &samples,
                            const BufferView<Ray> &rays,
                            const BufferView<RayDifferential> &primary_ray_differentials,
                            const BufferView<Intersection> &intersections,
                            const BufferView<DRay> &d_rays,
                            const BufferView<RayDifferential> &d_ray_differentials,
                            const BufferView<SurfacePoint> &d_surface_points,
                            BufferView<DVertex> d_vertices,
                            BufferView<DCameraInst> d_cameras) {
    parallel_for(d_primary_intersector{
        scene.camera,
        scene.shapes.data,
        active_pixels.begin(),
        samples.begin(),
        rays.begin(),
        primary_ray_differentials.begin(),
        intersections.begin(),
        d_rays.begin(),
        d_ray_differentials.begin(),
        d_surface_points.begin(),
        d_vertices.begin(),
        d_cameras.begin()}, active_pixels.size(), scene.use_gpu);
}

struct primary_contribs_accumulator {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &throughput = throughputs[pixel_id];
        const auto &shading_isect = shading_isects[pixel_id];
        const auto &incoming_ray = incoming_rays[pixel_id];
        Vector3 emission = Vector3{0, 0, 0};
        if (shading_isect.valid()) {
            const auto &shading_point = shading_points[pixel_id];
            const auto &shading_shape = scene.shapes[shading_isect.shape_id];
            auto wi = -incoming_ray.dir;
            if (shading_shape.light_id >= 0) {
                const auto &light = scene.area_lights[shading_shape.light_id];
                if (light.two_sided || dot(wi, shading_point.shading_frame.n) > 0) {
                    emission += light.intensity;
                }
            }
        } else if (scene.envmap != nullptr) {
            auto dir = incoming_rays[pixel_id].dir;
            emission = envmap_eval(*(scene.envmap), dir, incoming_ray_differentials[pixel_id]);
        }
        auto contrib = weight * throughput * emission;
        if (rendered_image != nullptr) {
            auto nc = channel_info.num_channels;
            auto nd = channel_info.num_total_dimensions;
            auto d = 0;
            for (int c = 0; c < nc; c++) {
                switch (channel_info.channels[c]) {
                    case Channels::radiance: {
                        rendered_image[nd * pixel_id + d] += contrib[0];
                        d++;
                        rendered_image[nd * pixel_id + d] += contrib[1];
                        d++;
                        rendered_image[nd * pixel_id + d] += contrib[2];
                        d++;
                    } break;
                    case Channels::alpha: {
                        if (shading_isect.valid()) {
                            auto alpha = weight;
                            if (channel_multipliers != nullptr) {
                                alpha *= channel_multipliers[nd * pixel_id + d];
                            }
                            rendered_image[nd * pixel_id + d] += alpha;
                        }
                        d++;
                    } break;
                    case Channels::depth: {
                        if (shading_isect.valid()) {
                            const auto &shading_point = shading_points[pixel_id];
                            auto depth = distance(incoming_ray.org,
                                                  shading_point.position) * weight;
                            if (channel_multipliers != nullptr) {
                                depth *= channel_multipliers[nd * pixel_id + d];
                            }
                            rendered_image[nd * pixel_id + d] += depth;
                        }
                        d++;
                    } break;
                    case Channels::position: {
                        if (shading_isect.valid()) {
                            const auto &shading_point = shading_points[pixel_id];
                            auto position = shading_point.position * weight;
                            if (channel_multipliers != nullptr) {
                                position[0] *= channel_multipliers[nd * pixel_id + d];
                                position[1] *= channel_multipliers[nd * pixel_id + d + 1];
                                position[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            }
                            rendered_image[nd * pixel_id + d    ] += position[0];
                            rendered_image[nd * pixel_id + d + 1] += position[1];
                            rendered_image[nd * pixel_id + d + 2] += position[2];
                        }
                        d += 3;
                    } break;
                    case Channels::geometry_normal: {
                        if (shading_isect.valid()) {
                            const auto &shading_point = shading_points[pixel_id];
                            auto geom_normal = shading_point.geom_normal * weight;
                            if (channel_multipliers != nullptr) {
                                geom_normal[0] *= channel_multipliers[nd * pixel_id + d];
                                geom_normal[1] *= channel_multipliers[nd * pixel_id + d + 1];
                                geom_normal[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            }
                            rendered_image[nd * pixel_id + d    ] += geom_normal[0];
                            rendered_image[nd * pixel_id + d + 1] += geom_normal[1];
                            rendered_image[nd * pixel_id + d + 2] += geom_normal[2];
                        }
                        d += 3;
                    } break;
                    case Channels::shading_normal: {
                        if (shading_isect.valid()) {
                            const auto &shading_point = shading_points[pixel_id];
                            auto shading_normal = shading_point.shading_frame[2] * weight;
                            if (channel_multipliers != nullptr) {
                                shading_normal[0] *= channel_multipliers[nd * pixel_id + d];
                                shading_normal[1] *= channel_multipliers[nd * pixel_id + d + 1];
                                shading_normal[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            }
                            rendered_image[nd * pixel_id + d    ] += shading_normal[0];
                            rendered_image[nd * pixel_id + d + 1] += shading_normal[1];
                            rendered_image[nd * pixel_id + d + 2] += shading_normal[2];
                        }
                        d += 3;
                    } break;
                    case Channels::uv: {
                        if (shading_isect.valid()) {
                            const auto &shading_point = shading_points[pixel_id];
                            auto uv = shading_point.uv * weight;
                            if (channel_multipliers != nullptr) {
                                uv[0] *= channel_multipliers[nd * pixel_id + d];
                                uv[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            }
                            rendered_image[nd * pixel_id + d    ] += uv[0];
                            rendered_image[nd * pixel_id + d + 1] += uv[1];
                        }
                        d += 2;
                    } break;
                    case Channels::diffuse_reflectance: {
                        if (shading_isect.valid()) {
                            const auto &shading_point = shading_points[pixel_id];
                            const auto &shading_shape = scene.shapes[shading_isect.shape_id];
                            const auto &material = scene.materials[shading_shape.material_id];
                            auto refl =
                                get_diffuse_reflectance(material, shading_point) * weight;
                            if (channel_multipliers != nullptr) {
                                refl[0] *= channel_multipliers[nd * pixel_id + d];
                                refl[1] *= channel_multipliers[nd * pixel_id + d + 1];
                                refl[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            }
                            rendered_image[nd * pixel_id + d    ] += refl[0];
                            rendered_image[nd * pixel_id + d + 1] += refl[1];
                            rendered_image[nd * pixel_id + d + 2] += refl[2];
                        }
                        d += 3;
                    } break;
                    case Channels::specular_reflectance: {
                        if (shading_isect.valid()) {
                            const auto &shading_point = shading_points[pixel_id];
                            const auto &shading_shape = scene.shapes[shading_isect.shape_id];
                            const auto &material = scene.materials[shading_shape.material_id];
                            auto refl =
                                get_specular_reflectance(material, shading_point) * weight;
                            if (channel_multipliers != nullptr) {
                                refl[0] *= channel_multipliers[nd * pixel_id + d];
                                refl[1] *= channel_multipliers[nd * pixel_id + d + 1];
                                refl[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            }
                            rendered_image[nd * pixel_id + d    ] += refl[0];
                            rendered_image[nd * pixel_id + d + 1] += refl[1];
                            rendered_image[nd * pixel_id + d + 2] += refl[2];
                        }
                        d += 3;
                    } break;
                    case Channels::roughness: {
                        if (shading_isect.valid()) {
                            const auto &shading_point = shading_points[pixel_id];
                            const auto &shading_shape = scene.shapes[shading_isect.shape_id];
                            const auto &material = scene.materials[shading_shape.material_id];
                            auto r = get_roughness(material, shading_point) * weight;
                            if (channel_multipliers != nullptr) {
                                r *= channel_multipliers[nd * pixel_id + d];
                            }
                            rendered_image[nd * pixel_id + d    ] += r;
                        }
                        d++;
                    } break;
                    default: {
                        assert(false);
                    }
                }
            }
        }
        if (edge_contribs != nullptr) {
            auto nc = channel_info.num_channels;
            auto nd = channel_info.num_total_dimensions;
            auto d = 0;
            for (int c = 0; c < nc; c++) {
                switch (channel_info.channels[c]) {
                    case Channels::radiance: {
                        edge_contribs[pixel_id] += sum(contrib);
                        d += 3;
                    } break;
                    case Channels::alpha: {
                        if (shading_isect.valid() && channel_multipliers != nullptr) {
                            auto alpha = weight;
                            alpha *= channel_multipliers[nd * pixel_id + d];
                            edge_contribs[pixel_id] += alpha;
                        }
                        d++;
                    } break;
                    case Channels::depth: {
                        if (shading_isect.valid() && channel_multipliers != nullptr) {
                            const auto &shading_point = shading_points[pixel_id];
                            auto depth = distance(incoming_ray.org,
                                                  shading_point.position) * weight;
                            depth *= channel_multipliers[nd * pixel_id + d];
                            edge_contribs[pixel_id] += depth;
                        }
                        d++;
                    } break;
                    case Channels::position: {
                        if (shading_isect.valid() && channel_multipliers != nullptr) {
                            const auto &shading_point = shading_points[pixel_id];
                            auto position = shading_point.position * weight;
                            position[0] *= channel_multipliers[nd * pixel_id + d];
                            position[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            position[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            edge_contribs[pixel_id] += sum(position);
                        }
                        d += 3;
                    } break;
                    case Channels::geometry_normal: {
                        if (shading_isect.valid() && channel_multipliers != nullptr) {
                            const auto &shading_point = shading_points[pixel_id];
                            auto geom_normal = shading_point.geom_normal * weight;
                            geom_normal[0] *= channel_multipliers[nd * pixel_id + d];
                            geom_normal[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            geom_normal[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            edge_contribs[pixel_id] += sum(geom_normal);
                        }
                        d += 3;
                    } break;
                    case Channels::shading_normal: {
                        if (shading_isect.valid() && channel_multipliers != nullptr) {
                            const auto &shading_point = shading_points[pixel_id];
                            auto shading_normal = shading_point.shading_frame[2] * weight;
                            shading_normal[0] *= channel_multipliers[nd * pixel_id + d];
                            shading_normal[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            shading_normal[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            edge_contribs[pixel_id] += sum(shading_normal);
                        }
                        d += 3;
                    } break;
                    case Channels::uv: {
                        if (shading_isect.valid() && channel_multipliers != nullptr) {
                            const auto &shading_point = shading_points[pixel_id];
                            auto uv = shading_point.uv * weight;
                            uv[0] *= channel_multipliers[nd * pixel_id + d];
                            uv[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            edge_contribs[pixel_id] += sum(uv);
                        }
                        d += 2;
                    } break;
                    case Channels::diffuse_reflectance: {
                        if (shading_isect.valid() && channel_multipliers != nullptr) {
                            const auto &shading_point = shading_points[pixel_id];
                            const auto &shading_shape = scene.shapes[shading_isect.shape_id];
                            const auto &material = scene.materials[shading_shape.material_id];
                            auto refl =
                                get_diffuse_reflectance(material, shading_point) * weight;
                            refl[0] *= channel_multipliers[nd * pixel_id + d];
                            refl[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            refl[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            edge_contribs[pixel_id] += sum(refl);
                        }
                        d += 3;
                    } break;
                    case Channels::specular_reflectance: {
                        if (shading_isect.valid() && channel_multipliers != nullptr) {
                            const auto &shading_point = shading_points[pixel_id];
                            const auto &shading_shape = scene.shapes[shading_isect.shape_id];
                            const auto &material = scene.materials[shading_shape.material_id];
                            auto refl =
                                get_specular_reflectance(material, shading_point) * weight;
                            refl[0] *= channel_multipliers[nd * pixel_id + d];
                            refl[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            refl[2] *= channel_multipliers[nd * pixel_id + d + 2];
                            edge_contribs[pixel_id] += sum(refl);
                        }
                        d += 3;
                    } break;
                    case Channels::roughness: {
                        if (shading_isect.valid() && channel_multipliers != nullptr) {
                            const auto &shading_point = shading_points[pixel_id];
                            const auto &shading_shape = scene.shapes[shading_isect.shape_id];
                            const auto &material = scene.materials[shading_shape.material_id];
                            auto r = get_roughness(material, shading_point) * weight;
                            r *= channel_multipliers[nd * pixel_id + d];
                            edge_contribs[pixel_id] += r;
                        }
                        d++;
                    } break;
                    default: {
                        assert(false);
                    }
                }
            }
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Vector3 *throughputs;
    const Real *channel_multipliers;
    const Ray *incoming_rays;
    const RayDifferential *incoming_ray_differentials;
    const Intersection *shading_isects;
    const SurfacePoint *shading_points;
    const Real weight;
    const ChannelInfo channel_info;
    float *rendered_image;
    Real *edge_contribs;
};

struct d_primary_contribs_accumulator {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &throughput = throughputs[pixel_id];
        const auto &shading_isect = shading_isects[pixel_id];
        const auto &incoming_ray = incoming_rays[pixel_id];
        d_direct_lights[idx] = DAreaLightInst{};
        if (scene.envmap != nullptr) {
            d_envmap_vals[idx] = DTexture3{};
            d_world_to_envs[idx] = Matrix4x4{};
        }
        d_diffuse_texs[idx] = DTexture3{};
        d_specular_texs[idx] = DTexture3{};
        d_roughness_texs[idx] = DTexture1{};
        auto nc = channel_info.num_channels;
        auto nd = channel_info.num_total_dimensions;
        auto d = 0;
        for (int c = 0; c < nc; c++) {
            switch (channel_info.channels[c]) {
                case Channels::radiance: {
                    // contrib = weight * throughput * emission
                    auto d_emission = weight * throughput *
                            Vector3{d_rendered_image[nd * pixel_id + d],
                                    d_rendered_image[nd * pixel_id + d + 1],
                                    d_rendered_image[nd * pixel_id + d + 2]};
                    if (shading_isect.valid()) {
                        const auto &shading_point = shading_points[pixel_id];
                        const auto &shading_shape = scene.shapes[shading_isect.shape_id];
                        auto wi = -incoming_rays[pixel_id].dir;

                        if (shading_shape.light_id >= 0 &&
                                dot(wi, shading_point.shading_frame.n) > 0) {
                            const auto &light = scene.area_lights[shading_shape.light_id];
                            if (light.two_sided || dot(wi, shading_point.shading_frame.n) > 0) {
                                d_direct_lights[idx].light_id = shading_shape.light_id;
                                d_direct_lights[idx].intensity = d_emission;
                            }
                        }
                    } else if (scene.envmap != nullptr) {
                        // Environment map
                        auto dir = incoming_rays[pixel_id].dir;
                        // emission = envmap_eval(*(scene.envmap),
                        //                        dir,
                        //                        incoming_ray_differentials[pixel_id])
                        d_envmap_eval(*(scene.envmap),
                                      dir,
                                      incoming_ray_differentials[pixel_id],
                                      d_emission,
                                      d_envmap_vals[idx],
                                      d_world_to_envs[idx],
                                      d_incoming_rays[pixel_id].dir,
                                      d_incoming_ray_differentials[pixel_id]);
                    }
                    d += 3;
                } break;
                case Channels::alpha: {
                    // Nothing to backprop to
                    d += 1;
                } break;
                case Channels::depth: {
                    if (shading_isect.valid()) {
                        const auto &shading_point = shading_points[pixel_id];
                        // auto depth = distance(incoming_ray.org,
                        //                       shading_point.position) * weight;
                        // depth *= channel_multipliers[nd * pixel_id + d];
                        auto d_depth = d_rendered_image[nd * pixel_id + d];
                        auto d_dist = d_depth * weight;
                        if (channel_multipliers != nullptr) {
                            d_dist *= channel_multipliers[nd * pixel_id + d];
                        }
                        auto d_org = Vector3{0, 0, 0};
                        auto d_position = Vector3{0, 0, 0};
                        d_distance(incoming_ray.org, shading_point.position, d_dist,
                                   d_org, d_position);
                        d_incoming_rays[pixel_id].org += d_org;
                        d_shading_points[pixel_id].position += d_position;
                    }
                    d += 1;
                } break;
                case Channels::position: {
                    if (shading_isect.valid()) {
                        auto d_position = weight *
                                Vector3{d_rendered_image[nd * pixel_id + d],
                                        d_rendered_image[nd * pixel_id + d + 1],
                                        d_rendered_image[nd * pixel_id + d + 2]};
                        if (channel_multipliers != nullptr) {
                            d_position[0] *= channel_multipliers[nd * pixel_id + d];
                            d_position[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            d_position[2] *= channel_multipliers[nd * pixel_id + d + 2];
                        }
                        d_shading_points[pixel_id].position += d_position;
                    }
                    d += 3;
                } break;
                case Channels::geometry_normal: {
                    if (shading_isect.valid()) {
                        auto d_geom_normal = weight *
                                Vector3{d_rendered_image[nd * pixel_id + d],
                                        d_rendered_image[nd * pixel_id + d + 1],
                                        d_rendered_image[nd * pixel_id + d + 2]};
                        if (channel_multipliers != nullptr) {
                            d_geom_normal[0] *= channel_multipliers[nd * pixel_id + d];
                            d_geom_normal[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            d_geom_normal[2] *= channel_multipliers[nd * pixel_id + d + 2];
                        }
                        d_shading_points[pixel_id].geom_normal += d_geom_normal;
                    }
                    d += 3;
                } break;
                case Channels::shading_normal: {
                    if (shading_isect.valid()) {
                        auto d_shading_normal = weight *
                                Vector3{d_rendered_image[nd * pixel_id + d],
                                        d_rendered_image[nd * pixel_id + d + 1],
                                        d_rendered_image[nd * pixel_id + d + 2]};
                        if (channel_multipliers != nullptr) {
                            d_shading_normal[0] *= channel_multipliers[nd * pixel_id + d];
                            d_shading_normal[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            d_shading_normal[2] *= channel_multipliers[nd * pixel_id + d + 2];
                        }
                        d_shading_points[pixel_id].shading_frame[2] += d_shading_normal;
                    }
                    d += 3;
                } break;
                case Channels::uv: {
                    if (shading_isect.valid()) {
                        auto d_uv = weight *
                                Vector2{d_rendered_image[nd * pixel_id + d],
                                        d_rendered_image[nd * pixel_id + d + 1]};
                        if (channel_multipliers != nullptr) {
                            d_uv[0] *= channel_multipliers[nd * pixel_id + d];
                            d_uv[1] *= channel_multipliers[nd * pixel_id + d + 1];
                        }
                        d_shading_points[pixel_id].uv += d_uv;
                    }
                    d += 2;
                } break;
                case Channels::diffuse_reflectance: {
                    if (shading_isect.valid()) {
                        const auto &shading_point = shading_points[pixel_id];
                        const auto &shape = scene.shapes[shading_isect.shape_id];
                        const auto &material = scene.materials[shape.material_id];
                        auto d_refl = weight *
                                Vector3{d_rendered_image[nd * pixel_id + d],
                                        d_rendered_image[nd * pixel_id + d + 1],
                                        d_rendered_image[nd * pixel_id + d + 2]};
                        if (channel_multipliers != nullptr) {
                            d_refl[0] *= channel_multipliers[nd * pixel_id + d];
                            d_refl[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            d_refl[2] *= channel_multipliers[nd * pixel_id + d + 2];
                        }
                        d_diffuse_texs[idx].material_id = shape.material_id;
                        d_get_diffuse_reflectance(material, shading_point, d_refl,
                            d_diffuse_texs[idx],
                            d_shading_points[pixel_id]);
                    }
                    d += 3;
                } break;
                case Channels::specular_reflectance: {
                    if (shading_isect.valid()) {
                        const auto &shading_point = shading_points[pixel_id];
                        const auto &shape = scene.shapes[shading_isect.shape_id];
                        const auto &material = scene.materials[shape.material_id];
                        auto d_refl = weight *
                                Vector3{d_rendered_image[nd * pixel_id + d],
                                        d_rendered_image[nd * pixel_id + d + 1],
                                        d_rendered_image[nd * pixel_id + d + 2]};
                        if (channel_multipliers != nullptr) {
                            d_refl[0] *= channel_multipliers[nd * pixel_id + d];
                            d_refl[1] *= channel_multipliers[nd * pixel_id + d + 1];
                            d_refl[2] *= channel_multipliers[nd * pixel_id + d + 2];
                        }
                        d_specular_texs[idx].material_id = shape.material_id;
                        d_get_specular_reflectance(material, shading_point, d_refl,
                            d_specular_texs[idx],
                            d_shading_points[pixel_id]);
                    }
                    d += 3;
                } break;
                case Channels::roughness: {
                    if (shading_isect.valid()) {
                        const auto &shading_point = shading_points[pixel_id];
                        const auto &shape = scene.shapes[shading_isect.shape_id];
                        const auto &material = scene.materials[shape.material_id];
                        auto d_roughness = weight *
                                d_rendered_image[nd * pixel_id + d];
                        if (channel_multipliers != nullptr) {
                            d_roughness *= channel_multipliers[nd * pixel_id + d];
                        }
                        d_roughness_texs[idx].material_id = shape.material_id;
                        d_get_roughness(material, shading_point, d_roughness,
                            d_roughness_texs[idx],
                            d_shading_points[pixel_id]);
                    }
                    d += 1;
                } break;
                default: {
                    assert(false);
                }
            }
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Vector3 *throughputs;
    const Real *channel_multipliers;
    const Ray *incoming_rays;
    const RayDifferential *incoming_ray_differentials;
    const Intersection *shading_isects;
    const SurfacePoint *shading_points;
    const Real weight;
    const ChannelInfo channel_info;
    const float *d_rendered_image;
    DAreaLightInst *d_direct_lights;
    DTexture3 *d_envmap_vals;
    Matrix4x4 *d_world_to_envs;
    DRay *d_incoming_rays;
    RayDifferential *d_incoming_ray_differentials;
    SurfacePoint *d_shading_points;
    DTexture3 *d_diffuse_texs;
    DTexture3 *d_specular_texs;
    DTexture1 *d_roughness_texs;
};

void accumulate_primary_contribs(
        const Scene &scene,
        const BufferView<int> &active_pixels,
        const BufferView<Vector3> &throughputs,
        const BufferView<Real> &channel_multipliers,
        const BufferView<Ray> &incoming_rays,
        const BufferView<RayDifferential> &incoming_ray_differentials,
        const BufferView<Intersection> &shading_isects,
        const BufferView<SurfacePoint> &shading_points,
        const Real weight,
        const ChannelInfo &channel_info,
        float *rendered_image,
        BufferView<Real> edge_contribs) {
    parallel_for(primary_contribs_accumulator{
        get_flatten_scene(scene),
        active_pixels.begin(),
        throughputs.begin(),
        channel_multipliers.begin(),
        incoming_rays.begin(),
        incoming_ray_differentials.begin(),
        shading_isects.begin(),
        shading_points.begin(),
        weight,
        channel_info,
        rendered_image,
        edge_contribs.begin()
    }, active_pixels.size(), scene.use_gpu);
}

void d_accumulate_primary_contribs(
        const Scene &scene,
        const BufferView<int> &active_pixels,
        const BufferView<Vector3> &throughputs,
        const BufferView<Real> &channel_multipliers,
        const BufferView<Ray> &incoming_rays,
        const BufferView<RayDifferential> &incoming_ray_differentials,
        const BufferView<Intersection> &shading_isects,
        const BufferView<SurfacePoint> &shading_points,
        const Real weight,
        const ChannelInfo &channel_info,
        const float *d_rendered_image,
        BufferView<DAreaLightInst> d_direct_lights,
        BufferView<DTexture3> d_envmap_vals,
        BufferView<Matrix4x4> d_world_to_envs,
        BufferView<DRay> d_incoming_rays,
        BufferView<RayDifferential> d_incoming_ray_differentials,
        BufferView<SurfacePoint> d_shading_points,
        BufferView<DTexture3> d_diffuse_texs,
        BufferView<DTexture3> d_specular_texs,
        BufferView<DTexture1> d_roughness_texs) {
    parallel_for(d_primary_contribs_accumulator{
        get_flatten_scene(scene),
        active_pixels.begin(),
        throughputs.begin(),
        channel_multipliers.begin(),
        incoming_rays.begin(),
        incoming_ray_differentials.begin(),
        shading_isects.begin(),
        shading_points.begin(),
        weight,
        channel_info,
        d_rendered_image,
        d_direct_lights.begin(),
        d_envmap_vals.begin(),
        d_world_to_envs.begin(),
        d_incoming_rays.begin(),
        d_incoming_ray_differentials.begin(),
        d_shading_points.begin(),
        d_diffuse_texs.begin(),
        d_specular_texs.begin(),
        d_roughness_texs.begin()
    }, active_pixels.size(), scene.use_gpu);
}

struct bsdf_sampler {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &isect = shading_isects[pixel_id];
        const auto &shape = scene.shapes[isect.shape_id];
        const auto &material = scene.materials[shape.material_id];
        const auto &incoming_ray = incoming_rays[pixel_id];
        const auto &shading_point = shading_points[pixel_id];

        next_rays[pixel_id] = Ray{
            shading_points[pixel_id].position,
            bsdf_sample(
                material,
                shading_point,
                -incoming_ray.dir,
                bsdf_samples[pixel_id],
                min_roughness[pixel_id],
                incoming_ray_differentials[pixel_id],
                bsdf_ray_differentials[pixel_id],
                &next_min_roughness[pixel_id])};
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Ray *incoming_rays;
    const RayDifferential *incoming_ray_differentials;
    const Intersection *shading_isects;
    const SurfacePoint *shading_points;
    const BSDFSample *bsdf_samples;
    const Real *min_roughness;
    Ray *next_rays;
    RayDifferential *bsdf_ray_differentials;
    Real *next_min_roughness;
};

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
                 BufferView<Real> next_min_roughness) {
    parallel_for(
        bsdf_sampler{get_flatten_scene(scene),
                     active_pixels.begin(),
                     incoming_rays.begin(),
                     incoming_ray_differentials.begin(),
                     shading_isects.begin(),
                     shading_points.begin(),
                     bsdf_samples.begin(),
                     min_roughness.begin(),
                     next_rays.begin(),
                     bsdf_ray_differentials.begin(),
                     next_min_roughness.begin()},
                     active_pixels.size(), scene.use_gpu);
}

struct path_contribs_accumulator {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &throughput = throughputs[pixel_id];
        assert(isfinite(throughput));
        const auto &incoming_ray = incoming_rays[pixel_id];
        const auto &shading_isect = shading_isects[pixel_id];
        const auto &shading_point = shading_points[pixel_id];
        const auto &light_isect = light_isects[pixel_id];
        const auto &light_point = light_points[pixel_id];
        const auto &light_ray = light_rays[pixel_id];
        const auto &bsdf_isect = bsdf_isects[pixel_id];
        const auto &bsdf_point = bsdf_points[pixel_id];
        const auto &bsdf_ray = bsdf_rays[pixel_id];
        const auto &min_rough = min_roughness[pixel_id];
        auto &next_throughput = next_throughputs[pixel_id];

        auto wi = -incoming_ray.dir;
        auto p = shading_point.position;
        const auto &shading_shape = scene.shapes[shading_isect.shape_id];
        const auto &material = scene.materials[shading_shape.material_id];

        // Next event estimation
        auto nee_contrib = Vector3{0, 0, 0};
        if (light_ray.tmax >= 0) { // tmax < 0 means the ray is blocked
            if (light_isect.valid()) {
                // area light
                const auto &light_shape = scene.shapes[light_isect.shape_id];
                auto dir = light_point.position - p;
                auto dist_sq = length_squared(dir);
                auto wo = dir / sqrt(dist_sq);
                if (light_shape.light_id >= 0) {
                    const auto &light = scene.area_lights[light_shape.light_id];
                    if (light.two_sided || dot(-wo, light_point.shading_frame.n) > 0) {
                        auto bsdf_val = bsdf(material, shading_point, wi, wo, min_rough);
                        auto geometry_term = fabs(dot(wo, light_point.geom_normal)) / dist_sq;
                        auto light_contrib = light.intensity;
                        auto light_pmf = scene.light_pmf[light_shape.light_id];
                        auto light_area = scene.light_areas[light_shape.light_id];
                        auto pdf_nee = light_pmf / light_area;
                        auto pdf_bsdf =
                            bsdf_pdf(material, shading_point, wi, wo, min_rough) * geometry_term;
                        auto mis_weight = square(pdf_nee) / (square(pdf_nee) + square(pdf_bsdf));
                        nee_contrib =
                            (mis_weight * geometry_term / pdf_nee) * bsdf_val * light_contrib;
                    }
                }
            } else if (scene.envmap != nullptr) {
                // Environment light
                auto wo = light_ray.dir;
                auto pdf_nee = envmap_pdf(*scene.envmap, wo);
                if (pdf_nee > 0) {
                    auto bsdf_val = bsdf(material, shading_point, wi, wo, min_rough);
                    // XXX: For now we don't use ray differentials for envmap
                    //      A proper approach might be to use a filter radius based on sampling density?
                    RayDifferential ray_diff{Vector3{0, 0, 0}, Vector3{0, 0, 0},
                                             Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                    auto light_contrib = envmap_eval(*scene.envmap, wo, ray_diff);
                    auto pdf_bsdf = bsdf_pdf(material, shading_point, wi, wo, min_rough);
                    auto mis_weight = square(pdf_nee) / (square(pdf_nee) + square(pdf_bsdf));
                    nee_contrib = (mis_weight / pdf_nee) * bsdf_val * light_contrib;
                }
            }
        }
        // BSDF importance sampling
        auto scatter_contrib = Vector3{0, 0, 0};
        auto scatter_bsdf = Vector3{0, 0, 0};
        if (bsdf_isect.valid()) {
            const auto &bsdf_shape = scene.shapes[bsdf_isect.shape_id];
            auto dir = bsdf_point.position - p;
            auto dist_sq = length_squared(dir);
            auto wo = dir / sqrt(dist_sq);
            auto pdf_bsdf = bsdf_pdf(material, shading_point, wi, wo, min_rough);
            if (pdf_bsdf > 1e-20f) {
                auto bsdf_val = bsdf(material, shading_point, wi, wo, min_rough);
                if (bsdf_shape.light_id >= 0) {
                    const auto &light = scene.area_lights[bsdf_shape.light_id];
                    if (light.two_sided || dot(-wo, bsdf_point.shading_frame.n) > 0) {
                        auto light_contrib = light.intensity;
                        auto light_pmf = scene.light_pmf[bsdf_shape.light_id];
                        auto light_area = scene.light_areas[bsdf_shape.light_id];
                        auto geometry_term = fabs(dot(wo, bsdf_point.geom_normal)) / dist_sq;
                        auto pdf_nee = (light_pmf / light_area) / geometry_term;
                        auto mis_weight = square(pdf_bsdf) / (square(pdf_nee) + square(pdf_bsdf));
                        scatter_contrib = (mis_weight / pdf_bsdf) * bsdf_val * light_contrib;
                    }
                }
                scatter_bsdf = bsdf_val / pdf_bsdf;
                next_throughput = throughput * scatter_bsdf;
            } else {
                next_throughput = Vector3{0, 0, 0};
            }
        } else if (scene.envmap != nullptr) {
            // Hit environment map
            auto wo = bsdf_ray.dir;
            auto pdf_bsdf = bsdf_pdf(material, shading_point, wi, wo, min_rough);
            if (pdf_bsdf > 1e-20f) {
                // XXX: For now we don't use ray differentials for envmap
                //      A proper approach might be to use a filter radius based on sampling density?
                RayDifferential ray_diff{Vector3{0, 0, 0}, Vector3{0, 0, 0},
                                         Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                auto bsdf_val = bsdf(material, shading_point, wi, wo, min_rough);
                auto light_contrib = envmap_eval(*scene.envmap, wo, ray_diff);
                auto pdf_nee = envmap_pdf(*scene.envmap, wo);
                auto mis_weight = square(pdf_bsdf) / (square(pdf_nee) + square(pdf_bsdf));
                scatter_contrib = (mis_weight / pdf_bsdf) * bsdf_val * light_contrib;
            } else {
                next_throughput = Vector3{0, 0, 0};
            }
        }

        auto path_contrib = throughput * (nee_contrib + scatter_contrib);
        assert(isfinite(nee_contrib));
        assert(isfinite(scatter_contrib));
        if (rendered_image != nullptr) {
            auto nd = channel_info.num_total_dimensions;
            auto d = channel_info.radiance_dimension;
            rendered_image[nd * pixel_id + d] += weight * path_contrib[0];
            rendered_image[nd * pixel_id + d + 1] += weight * path_contrib[1];
            rendered_image[nd * pixel_id + d + 2] += weight * path_contrib[2];
        }
        if (edge_contribs != nullptr) {
            edge_contribs[pixel_id] += sum(weight * path_contrib);
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Vector3 *throughputs;
    const Ray *incoming_rays;
    const Intersection *shading_isects;
    const SurfacePoint *shading_points;
    const Intersection *light_isects;
    const SurfacePoint *light_points;
    const Ray *light_rays;
    const Intersection *bsdf_isects;
    const SurfacePoint *bsdf_points;
    const Ray *bsdf_rays;
    const Real *min_roughness;
    const Real weight;
    const ChannelInfo channel_info;
    Vector3 *next_throughputs;
    float *rendered_image;
    Real *edge_contribs;
};

struct d_path_contribs_accumulator {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &throughput = throughputs[pixel_id];
        const auto &incoming_ray = incoming_rays[pixel_id];
        const auto &incoming_ray_differential = incoming_ray_differentials[pixel_id];
        const auto &bsdf_ray_differential = bsdf_ray_differentials[pixel_id];
        const auto &shading_isect = shading_isects[pixel_id];
        const auto &shading_point = shading_points[pixel_id];
        const auto &light_isect = light_isects[pixel_id];
        const auto &light_ray = light_rays[pixel_id];
        const auto &min_rough = min_roughness[pixel_id];

        auto d_light_v = d_light_vertices + 3 * idx;
        auto d_bsdf_v = d_bsdf_vertices + 3 * idx;
        auto &d_diffuse_tex = d_diffuse_texs[idx];
        auto &d_specular_tex = d_specular_texs[idx];
        auto &d_roughness_tex = d_roughness_texs[idx];
        auto &d_nee_light = d_nee_lights[idx];
        auto &d_bsdf_light = d_bsdf_lights[idx];
        auto &d_envmap_val = d_envmap_vals[idx];
        auto &d_world_to_env = d_world_to_envs[idx];

        auto &d_throughput = d_throughputs[pixel_id];
        auto &d_incoming_ray = d_incoming_rays[pixel_id];
        auto &d_incoming_ray_differential = d_incoming_ray_differentials[pixel_id];
        auto &d_shading_point = d_shading_points[pixel_id];
    
        auto wi = -incoming_ray.dir;
        auto p = shading_point.position;
        const auto &shading_shape = scene.shapes[shading_isect.shape_id];
        const auto &material = scene.materials[shading_shape.material_id];

        auto nd = channel_info.num_total_dimensions;
        auto d = channel_info.radiance_dimension;
        // rendered_image[nd * pixel_id + d    ] += weight * path_contrib[0];
        // rendered_image[nd * pixel_id + d + 1] += weight * path_contrib[1];
        // rendered_image[nd * pixel_id + d + 2] += weight * path_contrib[2];
        auto d_path_contrib = weight *
            Vector3{d_rendered_image[nd * pixel_id + d    ],
                    d_rendered_image[nd * pixel_id + d + 1],
                    d_rendered_image[nd * pixel_id + d + 2]};

        // Initialize derivatives
        d_light_v[0] = DVertex{};
        d_light_v[1] = DVertex{};
        d_light_v[2] = DVertex{};
        d_bsdf_v[0] = DVertex{};
        d_bsdf_v[1] = DVertex{};
        d_bsdf_v[2] = DVertex{};
        d_diffuse_tex = DTexture3{};
        d_specular_tex = DTexture3{};
        d_roughness_tex = DTexture1{};
        d_nee_light = DAreaLightInst{};
        d_bsdf_light = DAreaLightInst{};
        d_envmap_val = DTexture3{};
        d_world_to_env = Matrix4x4{};
        d_throughput = Vector3{0, 0, 0};
        d_incoming_ray = DRay{};
        d_incoming_ray_differential = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}};
        d_shading_point = SurfacePoint::zero();

        // Next event estimation
        if (light_ray.tmax >= 0) { // tmax < 0 means the ray is blocked
            if (light_isect.valid()) {
                // Area light
                const auto &light_shape = scene.shapes[light_isect.shape_id];
                const auto &light_sample = light_samples[pixel_id];
                const auto &light_point = light_points[pixel_id];

                auto dir = light_point.position - p;
                auto dist_sq = length_squared(dir);
                auto wo = dir / sqrt(dist_sq);
                if (light_shape.light_id >= 0) {
                    const auto &light = scene.area_lights[light_shape.light_id];
                    if (light.two_sided || dot(-wo, light_point.shading_frame.n) > 0) {
                        d_diffuse_tex.material_id = shading_shape.material_id;
                        d_specular_tex.material_id = shading_shape.material_id;
                        d_roughness_tex.material_id = shading_shape.material_id;

                        auto light_tri_index = get_indices(light_shape, light_isect.tri_id);
                        d_light_v[0].shape_id = light_isect.shape_id;
                        d_light_v[0].vertex_id = light_tri_index[0];
                        d_light_v[1].shape_id = light_isect.shape_id;
                        d_light_v[1].vertex_id = light_tri_index[1];
                        d_light_v[2].shape_id = light_isect.shape_id;
                        d_light_v[2].vertex_id = light_tri_index[2];
                        d_nee_light.light_id = light_shape.light_id;

                        auto bsdf_val = bsdf(material, shading_point, wi, wo, min_rough);
                        auto cos_light = dot(wo, light_point.geom_normal);
                        auto geometry_term = fabs(cos_light) / dist_sq;
                        const auto &light = scene.area_lights[light_shape.light_id];
                        auto light_contrib = light.intensity;
                        auto light_pmf = scene.light_pmf[light_shape.light_id];
                        auto light_area = scene.light_areas[light_shape.light_id];
                        auto pdf_nee = light_pmf / light_area;
                        auto pdf_bsdf =
                            bsdf_pdf(material, shading_point, wi, wo, min_rough) * geometry_term;
                        auto mis_weight = square(pdf_nee) / (square(pdf_nee) + square(pdf_bsdf));

                        auto nee_contrib = (mis_weight * geometry_term / pdf_nee) *
                                            bsdf_val * light_contrib;

                        // path_contrib = throughput * (nee_contrib + scatter_contrib)
                        auto d_nee_contrib = d_path_contrib * throughput;
                        d_throughput += d_path_contrib * nee_contrib;

                        auto weight = mis_weight / pdf_nee;
                        // nee_contrib = (weight * geometry_term) *
                        //                bsdf_val * light_contrib
                        // Ignore derivatives of MIS weight & PMF
                        auto d_weight = geometry_term *
                            sum(d_nee_contrib * bsdf_val * light_contrib);
                        // weight = mis_weight / pdf_nee
                        auto d_pdf_nee = -d_weight * weight / pdf_nee;
                        // nee_contrib = (weight * geometry_term) *
                        //                bsdf_val * light_contrib
                        auto d_geometry_term = weight * sum(d_nee_contrib * bsdf_val * light_contrib);
                        auto d_bsdf_val = weight * d_nee_contrib * geometry_term * light_contrib;
                        auto d_light_contrib = weight * d_nee_contrib * geometry_term * bsdf_val;
                        // pdf_nee = light_pmf / light_area
                        //         = light_pmf * tri_pmf / tri_area
                        auto d_area =
                            -d_pdf_nee * pdf_nee / get_area(light_shape, light_isect.tri_id);
                        d_get_area(light_shape, light_isect.tri_id, d_area, d_light_v);
                        // light_contrib = light.intensity
                        d_nee_light.intensity += d_light_contrib;
                        // geometry_term = fabs(cos_light) / dist_sq
                        auto d_cos_light = cos_light > 0 ?
                            d_geometry_term / dist_sq : -d_geometry_term / dist_sq;
                        auto d_dist_sq = -d_geometry_term * geometry_term / dist_sq;
                        // cos_light = dot(wo, light_point.geom_normal)
                        auto d_wo = d_cos_light * light_point.geom_normal;
                        auto d_light_point = SurfacePoint::zero();
                        d_light_point.geom_normal = d_cos_light * wo;
                        // bsdf_val = bsdf(material, shading_point, wi, wo)
                        auto d_wi = Vector3{0, 0, 0};
                        d_bsdf(material, shading_point, wi, wo, min_rough, d_bsdf_val,
                               d_diffuse_tex, d_specular_tex, d_roughness_tex,
                               d_shading_point, d_wi, d_wo);
                        // wo = dir / sqrt(dist_sq)
                        auto d_dir = d_wo / sqrt(dist_sq);
                        // sqrt(dist_sq)
                        auto d_sqrt_dist_sq = -sum(d_wo * dir) / dist_sq;
                        d_dist_sq += (0.5f * d_sqrt_dist_sq / sqrt(dist_sq));
                        // dist_sq = length_squared(dir)
                        d_dir += d_length_squared(dir, d_dist_sq);
                        // dir = light_point.position - p
                        d_light_point.position += d_dir;
                        d_shading_point.position -= d_dir;
                        // wi = -incoming_ray.dir
                        d_incoming_ray.dir -= d_wi;

                        // sample point on light
                        d_sample_shape(light_shape, light_isect.tri_id,
                            light_sample.uv, d_light_point, d_light_v);
                    }
                }
            } else if (scene.envmap != nullptr) {
                // Environment light
                auto wo = light_ray.dir;
                auto pdf_nee = envmap_pdf(*scene.envmap, wo);
                if (pdf_nee > 0) {
                    d_diffuse_tex.material_id = shading_shape.material_id;
                    d_specular_tex.material_id = shading_shape.material_id;
                    d_roughness_tex.material_id = shading_shape.material_id;

                    auto bsdf_val = bsdf(material, shading_point, wi, wo, min_rough);
                    // XXX: For now we don't use ray differentials for next event estimation.
                    //      A proper approach might be to use a filter radius based on sampling density?
                    auto ray_diff = RayDifferential{
                        Vector3{0, 0, 0}, Vector3{0, 0, 0},
                        Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                    auto light_contrib = envmap_eval(*scene.envmap, wo, ray_diff);
                    auto pdf_bsdf = bsdf_pdf(material, shading_point, wi, wo, min_rough);
                    auto mis_weight = square(pdf_nee) / (square(pdf_nee) + square(pdf_bsdf));
                    auto nee_contrib = (mis_weight / pdf_nee) * bsdf_val * light_contrib;

                    // path_contrib = throughput * (nee_contrib + scatter_contrib)
                    auto d_nee_contrib = d_path_contrib * throughput;
                    d_throughput += d_path_contrib * nee_contrib;

                    auto weight = mis_weight / pdf_nee;
                    // nee_contrib = weight * bsdf_val * light_contrib
                    // Ignore derivatives of MIS weight & pdf
                    auto d_bsdf_val = weight * d_nee_contrib * light_contrib;
                    auto d_light_contrib = weight * d_nee_contrib * bsdf_val;
                    auto d_wo = Vector3{0, 0, 0};
                    auto d_ray_diff = RayDifferential{
                        Vector3{0, 0, 0}, Vector3{0, 0, 0},
                        Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                    // light_contrib = eval_envmap(*scene.envmap, wo, ray_diff)
                    d_envmap_eval(*scene.envmap, wo, ray_diff, d_light_contrib,
                        d_envmap_val, d_world_to_env, d_wo, d_ray_diff);
                    // bsdf_val = bsdf(material, shading_point, wi, wo, min_rough)
                    auto d_wi = Vector3{0, 0, 0};
                    d_bsdf(material, shading_point, wi, wo, min_rough, d_bsdf_val,
                        d_diffuse_tex, d_specular_tex, d_roughness_tex,
                        d_shading_point, d_wi, d_wo);
                    // wi = -incoming_ray.dir
                    d_incoming_ray.dir -= d_wi;
                }
            }
        }

        // BSDF importance sampling
        const auto &bsdf_isect = bsdf_isects[pixel_id];
        if (bsdf_isect.valid()) {
            const auto &bsdf_shape = scene.shapes[bsdf_isect.shape_id];
            const auto &bsdf_sample = bsdf_samples[pixel_id];
            const auto &bsdf_point = bsdf_points[pixel_id];
            const auto &d_next_ray = d_next_rays[pixel_id];
            const auto &d_next_ray_differential = d_next_ray_differentials[pixel_id];
            const auto &d_next_point = d_next_points[pixel_id];
            const auto &d_next_throughput = d_next_throughputs[pixel_id];

            // Initialize bsdf vertex derivatives
            auto bsdf_tri_index = get_indices(bsdf_shape, bsdf_isect.tri_id);
            d_bsdf_v[0].shape_id = bsdf_isect.shape_id;
            d_bsdf_v[0].vertex_id = bsdf_tri_index[0];
            d_bsdf_v[1].shape_id = bsdf_isect.shape_id;
            d_bsdf_v[1].vertex_id = bsdf_tri_index[1];
            d_bsdf_v[2].shape_id = bsdf_isect.shape_id;
            d_bsdf_v[2].vertex_id = bsdf_tri_index[2];
            d_bsdf_light.light_id = bsdf_shape.light_id;

            auto dir = bsdf_point.position - p;
            auto dist_sq = length_squared(dir);
            auto wo = dir / sqrt(dist_sq);
            auto pdf_bsdf = bsdf_pdf(material, shading_point, wi, wo, min_rough);
            if (pdf_bsdf > 0) {
                d_diffuse_tex.material_id = shading_shape.material_id;
                d_specular_tex.material_id = shading_shape.material_id;
                d_roughness_tex.material_id = shading_shape.material_id;

                auto bsdf_val = bsdf(material, shading_point, wi, wo, min_rough);
                auto scatter_bsdf = bsdf_val / pdf_bsdf;

                // next_throughput = throughput * scatter_bsdf
                d_throughput += d_next_throughput * scatter_bsdf;
                auto d_scatter_bsdf = d_next_throughput * throughput;
                // scatter_bsdf = bsdf_val / pdf_bsdf
                auto d_bsdf_val = d_scatter_bsdf;
                auto d_pdf_bsdf = -sum(d_scatter_bsdf * scatter_bsdf) / pdf_bsdf;

                if (bsdf_shape.light_id >= 0) {
                    const auto &light = scene.area_lights[bsdf_shape.light_id];
                    if (light.two_sided || dot(-wo, bsdf_point.shading_frame.n) > 0) {
                        auto geometry_term = fabs(dot(wo, bsdf_point.geom_normal)) / dist_sq;
                        const auto &light = scene.area_lights[bsdf_shape.light_id];
                        auto light_contrib = light.intensity;
                        auto light_pmf = scene.light_pmf[bsdf_shape.light_id];
                        auto light_area = scene.light_areas[bsdf_shape.light_id];
                        auto pdf_nee = (light_pmf / light_area) / geometry_term;
                        auto mis_weight = square(pdf_bsdf) / (square(pdf_nee) + square(pdf_bsdf));
                        auto scatter_contrib = (mis_weight / pdf_bsdf) * bsdf_val * light_contrib;

                        // path_contrib = throughput * (nee_contrib + scatter_contrib)
                        auto d_scatter_contrib = d_path_contrib * throughput;
                        d_throughput += d_path_contrib * scatter_contrib;
                        auto weight = mis_weight / pdf_bsdf;
                        // scatter_contrib = weight * bsdf_val * light_contrib
                        
                        auto d_weight = sum(d_scatter_contrib * bsdf_val * light_contrib);
                        // Ignore derivatives of MIS weight
                        // weight = mis_weight / pdf_bsdf
                        d_pdf_bsdf += -d_weight * weight / pdf_bsdf;
                        d_bsdf_val += weight * d_scatter_contrib * light_contrib;
                        auto d_light_contrib = weight * d_scatter_contrib * bsdf_val;
                        // light_contrib = light.intensity
                        d_bsdf_light.intensity += d_light_contrib;
                    }
                }

                auto d_wi = Vector3{0, 0, 0};
                auto d_wo = d_next_ray.dir;
                // pdf_bsdf = bsdf_pdf(material, shading_point, wi, wo, min_rough)
                d_bsdf_pdf(material, shading_point, wi, wo, min_rough, d_pdf_bsdf,
                           d_roughness_tex, d_shading_point, d_wi, d_wo);
                // bsdf_val = bsdf(material, shading_point, wi, wo)
                d_bsdf(material, shading_point, wi, wo, min_rough, d_bsdf_val,
                       d_diffuse_tex, d_specular_tex, d_roughness_tex,
                       d_shading_point, d_wi, d_wo);

                // wo = dir / sqrt(dist_sq)
                auto d_dir = d_wo / sqrt(dist_sq);
                auto d_sqrt_dist_sq = -sum(d_wo * dir) / dist_sq;
                auto d_dist_sq = 0.5f * d_sqrt_dist_sq / sqrt(dist_sq);
                // dist_sq = length_squared(dir)
                d_dir += d_length_squared(dir, d_dist_sq);

                auto d_bsdf_point = d_next_point;
                // dir = bsdf_point.position - p
                d_bsdf_point.position += d_dir;
                // d_shading_point.position -= d_dir; (see below)

                // bsdf intersection
                DRay d_ray;
                RayDifferential d_bsdf_ray_differential{
                    Vector3{0, 0, 0}, Vector3{0, 0, 0},
                    Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                d_intersect_shape(bsdf_shape,
                                  bsdf_isect.tri_id,
                                  Ray{shading_point.position, wo},
                                  bsdf_ray_differential,
                                  d_bsdf_point,
                                  d_next_ray_differential,
                                  d_ray,
                                  d_bsdf_ray_differential,
                                  d_bsdf_v);

                // XXX HACK: diffuse interreflection causes a lot of noise
                // on position derivatives but has small impact on the final derivatives,
                // so we ignore them here.
                // A properer way is to come up with an importance sampling distribution,
                // or use a lot more samples
                if (min_rough > 0.01f) {
                    d_shading_point.position -= d_dir;
                    d_shading_point.position += d_ray.org;
                }
                d_wo += d_ray.dir;

                // sample bsdf direction
                d_bsdf_sample(material,
                              shading_point,
                              wi,
                              bsdf_sample,
                              min_rough,
                              incoming_ray_differential,
                              d_wo,
                              d_bsdf_ray_differential,
                              d_roughness_tex,
                              d_shading_point,
                              d_wi,
                              d_incoming_ray_differential);

                // wi = -incoming_ray.dir
                d_incoming_ray.dir -= d_wi;
            }
        } else if (scene.envmap != nullptr) {
            // Hit environment map
            const auto &bsdf_ray = bsdf_rays[pixel_id];
            
            auto wo = bsdf_ray.dir;
            auto pdf_bsdf = bsdf_pdf(material, shading_point, wi, wo, min_rough);
            if (pdf_bsdf > 0) {
                d_diffuse_tex.material_id = shading_shape.material_id;
                d_specular_tex.material_id = shading_shape.material_id;
                d_roughness_tex.material_id = shading_shape.material_id;
                d_envmap_val.material_id = 0;

                auto bsdf_val = bsdf(material, shading_point, wi, wo, min_rough);
                auto ray_diff = RayDifferential{
                    Vector3{0, 0, 0}, Vector3{0, 0, 0},
                    Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                auto light_contrib = envmap_eval(*scene.envmap, wo, ray_diff);
                auto pdf_nee = envmap_pdf(*scene.envmap, wo);
                auto mis_weight = square(pdf_bsdf) / (square(pdf_nee) + square(pdf_bsdf));
                auto scatter_contrib = (mis_weight / pdf_bsdf) * bsdf_val * light_contrib;

                // path_contrib = throughput * (nee_contrib + scatter_contrib)
                auto d_scatter_contrib = d_path_contrib * throughput;
                d_throughput += d_path_contrib * scatter_contrib;
                auto weight = mis_weight / pdf_bsdf;

                // scatter_contrib = weight * bsdf_val * light_contrib                
                // auto d_weight = sum(d_scatter_contrib * bsdf_val * light_contrib);
                // Ignore derivatives of MIS weight and pdf
                // XXX: unlike the case of mesh light sources, we don't propagate to
                //      the sampling procedure, since it causes higher variance when
                //      there is a huge gradients in d_envmap_eval()
                // weight = mis_weight / pdf_bsdf
                // auto d_pdf_bsdf = -d_weight * weight / pdf_bsdf;
                auto d_bsdf_val = weight * d_scatter_contrib * light_contrib;
                auto d_light_contrib = weight * d_scatter_contrib * bsdf_val;

                auto d_wo = Vector3{0, 0, 0};
                auto d_ray_diff = RayDifferential{
                    Vector3{0, 0, 0}, Vector3{0, 0, 0},
                    Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                // light_contrib = eval_envmap(*scene.envmap, wo, ray_diff)
                d_envmap_eval(*scene.envmap, wo, ray_diff, d_light_contrib,
                    d_envmap_val, d_world_to_env, d_wo, d_ray_diff);
                auto d_wi = Vector3{0, 0, 0};
                // bsdf_val = bsdf(material, shading_point, wi, wo)
                d_bsdf(material, shading_point, wi, wo, min_rough, d_bsdf_val,
                       d_diffuse_tex, d_specular_tex, d_roughness_tex,
                       d_shading_point, d_wi, d_wo);

                // pdf_bsdf = bsdf_pdf(material, shading_point, wi, wo, min_rough)
                // d_bsdf_pdf(material, shading_point, wi, wo, min_rough, d_pdf_bsdf,
                //            d_roughness_tex, d_shading_point, d_wi, d_wo);

                // sample bsdf direction
                // auto d_bsdf_ray_differential = RayDifferential{
                //     Vector3{0, 0, 0}, Vector3{0, 0, 0},
                //     Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                // d_bsdf_sample(material,
                //               shading_point,
                //               wi,
                //               bsdf_sample,
                //               min_rough,
                //               incoming_ray_differential,
                //               d_wo,
                //               d_bsdf_ray_differential,
                //               d_roughness_tex,
                //               d_shading_point,
                //               d_wi,
                //               d_incoming_ray_differential);

                // wi = -incoming_ray.dir
                d_incoming_ray.dir -= d_wi;
            }
        }
    }

    const FlattenScene scene;
    const int *active_pixels;
    const Vector3 *throughputs;
    const Ray *incoming_rays;
    const RayDifferential *incoming_ray_differentials;
    const LightSample *light_samples;
    const BSDFSample *bsdf_samples;
    const Intersection *shading_isects;
    const SurfacePoint *shading_points;
    const Intersection *light_isects;
    const SurfacePoint *light_points;
    const Ray *light_rays;
    const Intersection *bsdf_isects;
    const SurfacePoint *bsdf_points;
    const Ray *bsdf_rays;
    const RayDifferential *bsdf_ray_differentials;
    const Real *min_roughness;
    const Real weight;
    const ChannelInfo channel_info;
    const float *d_rendered_image;
    const Vector3 *d_next_throughputs;
    const DRay *d_next_rays;
    const RayDifferential *d_next_ray_differentials;
    const SurfacePoint *d_next_points;
    DVertex *d_light_vertices;
    DVertex *d_bsdf_vertices;
    DTexture3 *d_diffuse_texs;
    DTexture3 *d_specular_texs;
    DTexture1 *d_roughness_texs;
    DAreaLightInst *d_nee_lights;
    DAreaLightInst *d_bsdf_lights;
    DTexture3 *d_envmap_vals;
    Matrix4x4 *d_world_to_envs;
    Vector3 *d_throughputs;
    DRay *d_incoming_rays;
    RayDifferential *d_incoming_ray_differentials;
    SurfacePoint *d_shading_points;
};

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
                              BufferView<Real> edge_contribs) {
    parallel_for(path_contribs_accumulator{
        get_flatten_scene(scene),
        active_pixels.begin(),
        throughputs.begin(),
        incoming_rays.begin(),
        shading_isects.begin(),
        shading_points.begin(),
        light_isects.begin(),
        light_points.begin(),
        light_rays.begin(),
        bsdf_isects.begin(),
        bsdf_points.begin(),
        bsdf_rays.begin(),
        min_roughness.begin(),
        weight,
        channel_info,
        next_throughputs.begin(),
        rendered_image,
        edge_contribs.begin()}, active_pixels.size(), scene.use_gpu);
}

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
                                BufferView<SurfacePoint> d_shading_points) {
    parallel_for(d_path_contribs_accumulator{
        get_flatten_scene(scene),
        active_pixels.begin(),
        throughputs.begin(),
        incoming_rays.begin(),
        ray_differentials.begin(),
        light_samples.begin(),
        bsdf_samples.begin(),
        shading_isects.begin(),
        shading_points.begin(),
        light_isects.begin(),
        light_points.begin(),
        light_rays.begin(),
        bsdf_isects.begin(),
        bsdf_points.begin(),
        bsdf_rays.begin(),
        bsdf_ray_differentials.begin(),
        min_roughness.begin(),
        weight,
        channel_info,
        d_rendered_image,
        d_next_throughputs.begin(),
        d_next_rays.begin(),
        d_next_ray_differentials.begin(),
        d_next_points.begin(),
        d_light_vertices.begin(),
        d_bsdf_vertices.begin(),
        d_diffuse_texs.begin(),
        d_specular_texs.begin(),
        d_roughness_texs.begin(),
        d_nee_lights.begin(),
        d_bsdf_lights.begin(),
        d_envmap_vals.begin(),
        d_world_to_envs.begin(),
        d_throughputs.begin(),
        d_incoming_rays.begin(),
        d_incoming_ray_differentials.begin(),
        d_shading_points.begin()},
        active_pixels.size(), scene.use_gpu);
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
    end = DISPATCH_CACHED(use_gpu, thrust_alloc, thrust::remove, beg, end, DVertex{-1, -1});
    DISPATCH_CACHED(use_gpu, thrust_alloc, thrust::sort, beg, end);
    auto buffer_beg = reduce_buffer.begin();
    auto new_end = DISPATCH_CACHED(use_gpu, thrust_alloc, thrust::reduce_by_key,
        beg, end, // input keysi
        thrust::make_constant_iterator(0), // input values
        buffer_beg, // output keys
        thrust::make_discard_iterator()).first; // output values
    DISPATCH(use_gpu, thrust::copy, buffer_beg, new_end, beg);
    d_vertices.count = new_end - buffer_beg;
    // Accumulate to output derivatives
    accumulate_vertex(d_vertices, d_shapes, use_gpu);
}

void accumulate_diffuse(const Scene &scene,
                        BufferView<DTexture3> &d_diffuse,
                        BufferView<DTexture3> reduce_buffer,
                        BufferView<DMaterial> d_materials,
                        ThrustCachedAllocator &thrust_alloc) {
    if (d_diffuse.size() == 0) {
        return;
    }
    // Reduce into unique sequence
    auto beg = d_diffuse.begin();
    auto end = d_diffuse.end();
    end = DISPATCH_CACHED(scene.use_gpu,
        thrust_alloc, thrust::remove, beg, end, DTexture3{-1, -1, -1, -1});
    DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::sort, beg, end);
    auto buffer_beg = reduce_buffer.begin();
    auto new_end = DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::reduce_by_key,
        beg, end, // input keys
        thrust::make_constant_iterator(0), // input values
        buffer_beg, // output keys
        thrust::make_discard_iterator()).first; // output values
    DISPATCH(scene.use_gpu, thrust::copy, buffer_beg, new_end, beg);
    d_diffuse.count = new_end - buffer_beg;
    // Accumulate to output derivatives
    accumulate_diffuse(scene, d_diffuse, d_materials);
}

void accumulate_specular(const Scene &scene,
                         BufferView<DTexture3> &d_specular,
                         BufferView<DTexture3> reduce_buffer,
                         BufferView<DMaterial> d_materials,
                         ThrustCachedAllocator &thrust_alloc) {
    if (d_specular.size() == 0) {
        return;
    }
    // Reduce into unique sequence
    auto beg = d_specular.begin();
    auto end = d_specular.end();
    end = DISPATCH_CACHED(scene.use_gpu,
        thrust_alloc, thrust::remove, beg, end, DTexture3{-1, -1, -1, -1});
    DISPATCH_CACHED(scene.use_gpu,
        thrust_alloc, thrust::sort, beg, end);
    auto buffer_beg = reduce_buffer.begin();
    auto new_end = DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::reduce_by_key,
        beg, end, // input keys
        thrust::make_constant_iterator(0), // input values
        buffer_beg, // output keys
        thrust::make_discard_iterator()).first; // output values
    DISPATCH(scene.use_gpu, thrust::copy, buffer_beg, new_end, beg);
    d_specular.count = new_end - buffer_beg;
    // Accumulate to output derivatives
    accumulate_specular(scene, d_specular, d_materials);
}

void accumulate_roughness(const Scene &scene,
                          BufferView<DTexture1> &d_roughness,
                          BufferView<DTexture1> reduce_buffer,
                          BufferView<DMaterial> d_materials,
                          ThrustCachedAllocator &thrust_alloc) {
    if (d_roughness.size() == 0) {
        return;
    }
    // Reduce into unique sequence
    auto beg = d_roughness.begin();
    auto end = d_roughness.end();
    end = DISPATCH_CACHED(scene.use_gpu,
        thrust_alloc, thrust::remove, beg, end, DTexture1{-1, -1, -1, -1});
    DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::sort, beg, end);
    auto buffer_beg = reduce_buffer.begin();
    auto new_end = DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::reduce_by_key,
        beg, end, // input keys
        thrust::make_constant_iterator(0), // input values
        buffer_beg, // output keys
        thrust::make_discard_iterator()).first; // output values
    DISPATCH(scene.use_gpu, thrust::copy, buffer_beg, new_end, beg);
    d_roughness.count = new_end - buffer_beg;
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
    end = DISPATCH_CACHED(use_gpu, thrust_alloc, thrust::remove, beg, end, DAreaLightInst{-1});
    DISPATCH_CACHED(use_gpu, thrust_alloc, thrust::sort, beg, end);
    auto buffer_beg = reduce_buffer.begin();
    auto new_end = DISPATCH_CACHED(use_gpu, thrust_alloc, thrust::reduce_by_key,
        beg, end, // input keys
        thrust::make_constant_iterator(0), // input values
        buffer_beg, // output keys
        thrust::make_discard_iterator()).first; // output values
    DISPATCH(use_gpu, thrust::copy, buffer_beg, new_end, beg);
    d_light_insts.count = new_end - buffer_beg;
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
        end = DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::remove, beg, end, DTexture3{-1});
        DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::sort, beg, end);
        auto buffer_beg = reduce_buffer.begin();
        auto new_end = DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::reduce_by_key,
            beg, end, // input keys
            thrust::make_constant_iterator(0), // input values
            buffer_beg, // output keys
            thrust::make_discard_iterator()).first; // output values
        DISPATCH(scene.use_gpu, thrust::copy, buffer_beg, new_end, beg);
       
        d_envmap_vals.count = new_end - buffer_beg;
    }
    auto d_world_to_env = Matrix4x4();
    {
        // Reduce to a single matrix
        auto beg = d_world_to_envs.begin();
        auto end = d_world_to_envs.begin();
        d_world_to_env = DISPATCH_CACHED(scene.use_gpu, thrust_alloc, thrust::reduce, beg, end, Matrix4x4());
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
    parallel_init();
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
    auto sampler = Sampler(scene.use_gpu, options.seed, num_pixels);
    auto edge_sampler = Sampler(scene.use_gpu, options.seed + 131071U, num_pixels);
    auto optix_rays = path_buffer.optix_rays.view(0, 2 * num_pixels);
    auto optix_hits = path_buffer.optix_hits.view(0, 2 * num_pixels);

    ThrustCachedAllocator thrust_alloc(scene.use_gpu, num_pixels * sizeof(DTexture3));

    // For each sample
    for (int sample_id = 0; sample_id < options.num_samples; sample_id++) {
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
        sampler.next_camera_samples(camera_samples);
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
            sampler.next_light_samples(light_samples);
            sample_point_on_light(scene,
                                  active_pixels,
                                  shading_points,
                                  light_samples,
                                  light_isects,
                                  light_points,
                                  nee_rays);
            occluded(scene, active_pixels, nee_rays, optix_rays, optix_hits);
            
            // Sample directions based on BRDF
            sampler.next_bsdf_samples(bsdf_samples);
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
            bool first = true;
             // Traverse the path backward for the derivatives
            for (int depth = max_bounces - 1; depth >= 0; depth--) {
                // Buffer views for this path vertex
                auto num_actives = num_active_pixels[depth];
                if (num_actives <= 0) {
                    break;
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

                ////////////////////////////////////////////////////////////////////////////////
                // Sample edges for secondary visibility
                auto num_edge_samples = 2 * num_actives;
                auto edge_samples = path_buffer.secondary_edge_samples.view(0, num_actives);
                edge_sampler.next_secondary_edge_samples(edge_samples);
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
                    edge_sampler.next_light_samples(tmp_light_samples);
                    // Copy the samples
                    parallel_for(copy_interleave<LightSample>{
                        tmp_light_samples.begin(), light_samples.begin()},
                        tmp_light_samples.size(), scene.use_gpu);
                    sample_point_on_light(
                        scene, active_pixels, shading_points,
                        light_samples, light_isects, light_points, nee_rays);
                    occluded(scene, active_pixels, nee_rays, optix_rays, optix_hits);

                    // Sample directions based on BRDF
                    edge_sampler.next_bsdf_samples(tmp_bsdf_samples);
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
                ////////////////////////////////////////////////////////////////////////////////
                
                // for (int i = 0; i < active_pixels.size(); i++) {
                //     auto pixel_id = active_pixels[i];
                //     // auto d_p = d_points[pixel_id].position;
                //     // debug_image[3 * pixel_id + 0] += d_p[0];
                //     // debug_image[3 * pixel_id + 1] += d_p[0];
                //     // debug_image[3 * pixel_id + 2] += d_p[0];
                //     auto edge_record = edge_records[i];
                //     if (edge_record.shape_id == 6) {
                //         auto d_v0 = d_edge_vertices[2 * i + 0].d_v;
                //         auto d_v1 = d_edge_vertices[2 * i + 1].d_v;
                //         debug_image[3 * pixel_id + 0] += d_v0[0] + d_v1[0];
                //         debug_image[3 * pixel_id + 1] += d_v0[0] + d_v1[0];
                //         debug_image[3 * pixel_id + 2] += d_v0[0] + d_v1[0];
                //         // auto ec0 = edge_contribs[2 * i + 0];
                //         // auto ec1 = edge_contribs[2 * i + 1];
                //         // debug_image[3 * pixel_id + 0] += ec0 + ec1;
                //         // debug_image[3 * pixel_id + 1] += ec0 + ec1;
                //         // debug_image[3 * pixel_id + 2] += ec0 + ec1;
                //     }
                // }

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
                accumulate_vertex(
                    d_edge_vertices,
                    path_buffer.d_vertex_reduce_buffer.view(0, 2 * num_actives),
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

                // for (int i = 0; i < primary_active_pixels.size(); i++) {
                //     auto pixel_id = primary_active_pixels[i];
                //     auto d_v0 = d_primary_vertices[3 * i + 0].d_v;
                //     auto d_v1 = d_primary_vertices[3 * i + 1].d_v;
                //     auto d_v2 = d_primary_vertices[3 * i + 2].d_v;
                //     debug_image[3 * pixel_id + 0] += (d_v0[0] + d_v1[0] + d_v2[0]) / 3.f;
                //     debug_image[3 * pixel_id + 1] += (d_v0[0] + d_v1[0] + d_v2[0]) / 3.f;
                //     debug_image[3 * pixel_id + 2] += (d_v0[0] + d_v1[0] + d_v2[0]) / 3.f;
                // }

                // for (int i = 0; i < primary_active_pixels.size(); i++) {
                //     auto pixel_id = primary_active_pixels[i];
                //     auto d_c2w = d_cameras[i].cam_to_world;
                //     debug_image[3 * pixel_id + 0] += d_c2w(0, 3);
                //     debug_image[3 * pixel_id + 1] += d_c2w(0, 3);
                //     debug_image[3 * pixel_id + 2] += d_c2w(0, 3);
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
                accumulate_camera(d_camera, d_scene->camera, scene.use_gpu);
            }

            /////////////////////////////////////////////////////////////////////////////////
            // Sample primary edges for geometric derivatives
            if (scene.edge_sampler.edges.size() > 0) {
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
                edge_sampler.next_primary_edge_samples(primary_edge_samples);
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
                    edge_sampler.next_light_samples(tmp_light_samples);
                    // Copy the samples
                    parallel_for(copy_interleave<LightSample>{
                        tmp_light_samples.begin(), light_samples.begin()},
                        tmp_light_samples.size(), scene.use_gpu);
                    sample_point_on_light(
                        scene, active_pixels, shading_points,
                        light_samples, light_isects, light_points, nee_rays);
                    occluded(scene, active_pixels, nee_rays, optix_rays, optix_hits);

                    // Sample directions based on BRDF
                    edge_sampler.next_bsdf_samples(tmp_bsdf_samples);
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
                accumulate_camera(d_camera, d_scene->camera, scene.use_gpu);

                // for (int i = 0; i < primary_edge_records.size(); i++) {
                //     auto rec = primary_edge_records[i];
                //     auto edge_pt = rec.edge_pt;
                //     auto xi = int(edge_pt[0] * camera.width);
                //     auto yi = int(edge_pt[1] * camera.height);
                //     auto d_cam = d_cameras[i].cam_to_world;
                //     debug_image[3 * (yi * camera.width + xi) + 0] += d_cam(0, 2);
                //     debug_image[3 * (yi * camera.width + xi) + 1] += d_cam(0, 2);
                //     debug_image[3 * (yi * camera.width + xi) + 2] += d_cam(0, 2);
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
}
