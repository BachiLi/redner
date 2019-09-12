#include "edge.h"
#include "line_clip.h"
#include "scene.h"
#include "parallel.h"
#include "thrust_utils.h"
#include "ltc.inc"
#include <memory>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/remove.h>

struct edge_collector {
    DEVICE inline void operator()(int idx) {
        const auto &shape = *shape_ptr;
        // For each triangle
        auto ind = get_indices(shape, idx / 3);
        if ((idx % 3) == 0) {
            edges[idx] = Edge{shape_id,
                              min(ind[0], ind[1]),
                              max(ind[0], ind[1]),
                              idx / 3, -1};
        } else if ((idx % 3) == 1) {
            edges[idx] = Edge{shape_id,
                              min(ind[1], ind[2]),
                              max(ind[1], ind[2]),
                              idx / 3, -1};
        } else {
            edges[idx] = Edge{shape_id,
                              min(ind[2], ind[0]),
                              max(ind[2], ind[0]),
                              idx / 3, -1};
        }
    }

    int shape_id;
    const Shape *shape_ptr;
    Edge *edges;
};

struct edge_less_comparer {
    DEVICE inline bool operator()(const Edge &e0, const Edge &e1) {
        if (e0.v0 == e1.v0) {
            return e0.v1 < e1.v1;
        }
        return e0.v0 < e1.v0;
    }
};

struct edge_equal_comparer {
    DEVICE inline bool operator()(const Edge &e0, const Edge &e1) {
        return e0.v0 == e1.v0 && e0.v1 == e1.v1;
    }
};

struct edge_merger {
    DEVICE inline Edge operator()(const Edge &e0, const Edge &e1) {
        return Edge{e0.shape_id, e0.v0, e0.v1, e0.f0, e1.f0};
    }
};

DEVICE inline bool less_than(const Vector3f &v0, const Vector3f &v1) {
    if (v0.x != v1.x) {
        return v0.x < v1.x;
    } else if (v0.y != v1.y) {
        return v0.y < v1.y;
    } else if (v0.z != v1.z) {
        return v0.z < v1.z;
    }
    return true;
}

struct edge_vertex_comparer {
    DEVICE inline bool operator()(const Edge &e0, const Edge &e1) {
        // First, locally sort v0 & v1 within e0 & e1
        auto v00 = get_vertex(*shape_ptr, e0.v0);
        auto v01 = get_vertex(*shape_ptr, e0.v1);
        if (less_than(v01, v00)) {
            swap_(v00, v01);
        }
        auto v10 = get_vertex(*shape_ptr, e1.v0);
        auto v11 = get_vertex(*shape_ptr, e1.v1);
        if (less_than(v11, v10)) {
            swap_(v10, v11);
        }
        // Next, compare and return
        if (v00 != v10) {
            return less_than(v00, v10);
        } else if (v01 != v11) {
            return less_than(v01, v11);
        }
        return true;
    }

    const Shape *shape_ptr;
};

struct edge_face_assigner {
    DEVICE void operator()(int idx) {
        auto &edge = edges[idx];
        if (edge.f1 != -1) {
            return;
        }
        auto v0 = get_vertex(*shape_ptr, edge.v0);
        auto v1 = get_vertex(*shape_ptr, edge.v1);
        if (less_than(v1, v0)) {
            swap_(v0, v1);
        }
        if (idx > 0) {
            const auto &cmp_edge = edges[idx - 1];
            auto cmp_v0 = get_vertex(*shape_ptr, cmp_edge.v0);
            auto cmp_v1 = get_vertex(*shape_ptr, cmp_edge.v1);
            if (less_than(cmp_v1, cmp_v0)) {
                swap_(cmp_v0, cmp_v1);
            }
            if (v0 == cmp_v0 && v1 == cmp_v1) {
                edge.f1 = cmp_edge.f0;
            }
        }
        if (idx < num_edges - 1) {
            const auto &cmp_edge = edges[idx + 1];
            auto cmp_v0 = get_vertex(*shape_ptr, cmp_edge.v0);
            auto cmp_v1 = get_vertex(*shape_ptr, cmp_edge.v1);
            if (less_than(cmp_v1, cmp_v0)) {
                swap_(cmp_v0, cmp_v1);
            }
            if (v0 == cmp_v0 && v1 == cmp_v1) {
                edge.f1 = cmp_edge.f0;
            }
        }
    }

    const Shape *shape_ptr;
    Edge *edges;
    int num_edges;
};

struct edge_remover {
    DEVICE inline bool operator()(const Edge &e) {
        if (e.f0 == -1 || e.f1 == -1) {
            // Only adjacent to one face
            return false;
        }
        auto v0 = Vector3{get_v0(shapes, e)};
        auto v1 = Vector3{get_v1(shapes, e)};
        auto ns_v0 = Vector3{get_non_shared_v0(shapes, e)};
        auto ns_v1 = Vector3{get_non_shared_v1(shapes, e)};
        auto n0 = normalize(cross(v0 - ns_v0, v1 - ns_v0));
        auto n1 = normalize(cross(v1 - ns_v1, v0 - ns_v1));
        return dot(n0, n1) >= (1 - 1e-6f);
    }

    const Shape *shapes;
};

struct primary_edge_weighter {
    DEVICE void operator()(int idx) {
        const auto &edge = edges[idx];
        auto &primary_edge_weight = primary_edge_weights[idx];
        auto v0 = get_v0(shapes, edge);
        auto v1 = get_v1(shapes, edge);
        auto v0p = Vector2{};
        auto v1p = Vector2{};
        primary_edge_weight = 0;
        // Project to screen space
        if (project(camera, Vector3(v0), Vector3(v1), v0p, v1p)) {
            auto v0c = v0p;
            auto v1c = v1p;
            // Clip against screen boundaries
            if (clip_line(v0p, v1p, v0c, v1c)) {
                // Reject non-silhouette edges
                auto org = xfm_point(camera.cam_to_world, Vector3{0, 0, 0});
                if (is_silhouette(shapes, org, edge)) {
                    primary_edge_weight = distance(v0c, v1c);
                }
            }
        }
    }

    Camera camera;
    const Shape *shapes;
    const Edge *edges;
    Real *primary_edge_weights;
};

struct secondary_edge_weighter {
    DEVICE void operator()(int idx) {
        const auto &edge = edges[idx];
        // We use the length * (pi - dihedral angle) to sample the edges
        // If the dihedral angle is large, it's less likely that the edge would be a silhouette
        auto &secondary_edge_weight = secondary_edge_weights[idx];
        auto exterior_dihedral = compute_exterior_dihedral_angle(shapes, edge);
        auto v0 = get_v0(shapes, edge);
        auto v1 = get_v1(shapes, edge);
        secondary_edge_weight = distance(v0, v1) * exterior_dihedral;
    }

    const Shape *shapes;
    const Edge *edges;
    Real *secondary_edge_weights;
};

EdgeSampler::EdgeSampler(const std::vector<const Shape*> &shapes,
                         const Scene &scene) {
    if (!scene.use_primary_edge_sampling && !scene.use_secondary_edge_sampling) {
        // No need to collect edges
        return;
    }
    auto shapes_buffer = scene.shapes.view(0, shapes.size());
    // Conservatively allocate a big buffer for all edges
    auto num_total_triangles = 0;
    for (int shape_id = 0; shape_id < (int)shapes.size(); shape_id++) {
        num_total_triangles += shapes[shape_id]->num_triangles;
    }
    // Collect the edges
    // TODO: this assumes each edge is only associated with two triangles
    //       which may be untrue for some pathological meshes.
    //       For edges associated to more than two triangles, 
    //       we should just ignore them
    edges = Buffer<Edge>(scene.use_gpu, 3 * num_total_triangles);
    auto edges_buffer = Buffer<Edge>(scene.use_gpu, 3 * num_total_triangles);
    auto current_num_edges = 0;
    for (int shape_id = 0; shape_id < (int)shapes.size(); shape_id++) {
        parallel_for(edge_collector{
            shape_id,
            shapes_buffer.begin() + shape_id,
            edges.data + current_num_edges
        }, 3 * shapes[shape_id]->num_triangles, scene.use_gpu);
        // Merge the edges
        auto edges_begin = edges.data + current_num_edges;
        DISPATCH(scene.use_gpu, thrust::sort,
                 edges_begin,
                 edges_begin + 3 * shapes[shape_id]->num_triangles,
                 edge_less_comparer{});
        auto edges_buffer_begin = edges_buffer.data;
        auto new_end = DISPATCH(scene.use_gpu, thrust::reduce_by_key,
            edges_begin, // input keys
            edges_begin + 3 * shapes[shape_id]->num_triangles,
            edges_begin, // input values
            edges_buffer_begin, // output keys
            edges_buffer_begin, // output values
            edge_equal_comparer{},
            edge_merger{}).first;
        auto num_edges = new_end - edges_buffer_begin;
        // Sometimes there are duplicated edges that don't get merged 
        // in the procedure above (e.g. UV seams), here we make sure these edges
        // are associated with two faces.
        // We do this by sorting the edges again based on vertex positions,
        // look at nearby edges and assign faces.
        DISPATCH(scene.use_gpu, thrust::sort,
                 edges_buffer_begin,
                 edges_buffer_begin + num_edges,
                 edge_vertex_comparer{shapes_buffer.begin() + shape_id});
        parallel_for(edge_face_assigner{
            shapes_buffer.begin() + shape_id,
            edges_buffer_begin,
            (int)num_edges
        }, num_edges, scene.use_gpu);

        DISPATCH(scene.use_gpu, thrust::copy, edges_buffer_begin, new_end, edges_begin);
        current_num_edges += num_edges;
    }
    // Remove edges with 180 degree dihedral angles
    auto edges_end = DISPATCH(scene.use_gpu, thrust::remove_if, edges.begin(),
        edges.begin() + current_num_edges, edge_remover{shapes_buffer.begin()});
    edges.count = edges_end - edges.begin();

    if (scene.use_primary_edge_sampling) {
        // Primary edge sampler:
        primary_edges_pmf = Buffer<Real>(scene.use_gpu, edges.count);
        primary_edges_cdf = Buffer<Real>(scene.use_gpu, edges.count);
        // For each edge, if it is a silhouette, we project them on screen
        // and compute the screen-space length. We store the length in
        // primary_edges_pmf
        {
            parallel_for(primary_edge_weighter{
                scene.camera,
                scene.shapes.data,
                edges.begin(),
                primary_edges_pmf.begin()
            }, edges.size(), scene.use_gpu);
            // Compute PMF & CDF
            // First normalize primary_edges_pmf.
            auto total_length = DISPATCH(scene.use_gpu, thrust::reduce,
                primary_edges_pmf.begin(),
                primary_edges_pmf.end(),
                Real(0),
                thrust::plus<Real>());
            DISPATCH(scene.use_gpu, thrust::transform,
                primary_edges_pmf.begin(),
                primary_edges_pmf.end(),
                thrust::make_constant_iterator(total_length),
                primary_edges_pmf.begin(),
                thrust::divides<Real>());
            // Next we compute a prefix sum
            DISPATCH(scene.use_gpu, thrust::transform_exclusive_scan,
                primary_edges_pmf.begin(),
                primary_edges_pmf.end(),
                primary_edges_cdf.begin(),
                thrust::identity<Real>(), Real(0), thrust::plus<Real>());
        }
    }

    if (scene.use_secondary_edge_sampling) {
        // Secondary edge sampler
        edge_tree = std::unique_ptr<EdgeTree>(
            new EdgeTree(scene.use_gpu,
                         scene.camera,
                         shapes_buffer,
                         edges.view(0, edges.size()),
                         scene.light_triangles.view(0, scene.light_triangles.size())));
    }
}

struct primary_edge_sampler {
    DEVICE void operator()(int idx) {
        // Initialize output
        edge_records[idx] = PrimaryEdgeRecord{};
        throughputs[2 * idx + 0] = Vector3{0, 0, 0};
        throughputs[2 * idx + 1] = Vector3{0, 0, 0};
        auto nd = channel_info.num_total_dimensions;
        for (int d = 0; d < nd; d++) {
            channel_multipliers[2 * nd * idx + d] = 0;
            channel_multipliers[2 * nd * idx + d + nd] = 0;
        }
        rays[2 * idx + 0] = Ray(Vector3{0, 0, 0}, Vector3{0, 0, 0});
        rays[2 * idx + 1] = Ray(Vector3{0, 0, 0}, Vector3{0, 0, 0});

        // Sample an edge by binary search on cdf
        auto sample = samples[idx];
        const Real *edge_ptr = thrust::upper_bound(thrust::seq,
                edges_cdf, edges_cdf + num_edges,
                sample.edge_sel);
        auto edge_id = clamp((int)(edge_ptr - edges_cdf - 1),
                                   0, num_edges - 1);
        const auto &edge = edges[edge_id];
        // Sample a point on the edge
        auto v0 = Vector3{get_v0(shapes, edge)};
        auto v1 = Vector3{get_v1(shapes, edge)};
        // Project the edge onto screen space
        auto v0_ss = Vector2{0, 0};
        auto v1_ss = Vector2{0, 0};
        if (!project(camera, v0, v1, v0_ss, v1_ss)) {
            return;
        }
        if (edges_pmf[edge_id] <= 0.f) {
            return;
        }

        if (camera.camera_type != CameraType::Fisheye) {
            // Perspective or Orthographic cameras

            // Uniform sample on the edge
            auto edge_pt = v0_ss + sample.t * (v1_ss - v0_ss);
            // Reject samples outside of image plane
            if (!in_screen(camera, edge_pt)) {
                return;
            }

            edge_records[idx].edge = edge;
            edge_records[idx].edge_pt = edge_pt;

            // Generate two rays at the two sides of the edge
            auto half_space_normal = get_normal(normalize(v0_ss - v1_ss));
            // The half space normal always points to the upper half-space.
            auto offset = 1e-6f;
            auto upper_pt = edge_pt + half_space_normal * offset;
            auto upper_ray = sample_primary(camera, upper_pt);
            auto lower_pt = edge_pt - half_space_normal * offset;
            auto lower_ray = sample_primary(camera, lower_pt);
            rays[2 * idx + 0] = upper_ray;
            rays[2 * idx + 1] = lower_ray;

            // Compute the corresponding backprop derivatives
            auto xi = clamp(int(edge_pt[0] * camera.width), 0, camera.width - 1);
            auto yi = clamp(int(edge_pt[1] * camera.height), 0, camera.height - 1);
            auto rd = channel_info.radiance_dimension;
            auto d_color = Vector3{
                d_rendered_image[nd * (yi * camera.width + xi) + rd + 0],
                d_rendered_image[nd * (yi * camera.width + xi) + rd + 1],
                d_rendered_image[nd * (yi * camera.width + xi) + rd + 2]
            };
            // The weight is the length of edge divided by the probability
            // of selecting this edge, divided by the length of gradients
            // of the edge equation w.r.t. screen coordinate.
            // For perspective projection the length of edge and gradients
            // cancel each other out.
            // For fisheye we need to compute the Jacobians
            auto upper_weight = d_color / edges_pmf[edge_id];
            auto lower_weight = -d_color / edges_pmf[edge_id];

            assert(isfinite(d_color));
            assert(isfinite(upper_weight));

            throughputs[2 * idx + 0] = upper_weight;
            throughputs[2 * idx + 1] = lower_weight;

            for (int d = 0; d < nd; d++) {
                auto d_channel = d_rendered_image[nd * (yi * camera.width + xi) + d];
                channel_multipliers[2 * nd * idx + d] = d_channel / edges_pmf[edge_id];
                channel_multipliers[2 * nd * idx + d + nd] = -d_channel / edges_pmf[edge_id];
            }
        } else {
            // In paper we focused on linear projection model.
            // However we also support non-linear models such as fisheye
            // projection.
            // To sample a point on the edge for non-linear models,
            // we need to sample in camera space instead of screen space,
            // since the edge is no longer a line segment in screen space.
            // Therefore we perform an "unprojection" to project the edge
            // from screen space to the film in camera space.
            // For perspective camera this is equivalent to sample in screen space:
            // we unproject (x, y) to (x', y', 1) where x', y' are just individual
            // affine transforms of x, y.
            // For fisheye camera we unproject from screen-space to the unit
            // sphere.
            // Therefore the following code also works for perspective camera,
            // but to make things more consistent to the paper we provide
            // two versions of code.
            auto v0_dir = screen_to_camera(camera, v0_ss);
            auto v1_dir = screen_to_camera(camera, v1_ss);
            // Uniform sample in camera space
            auto v_dir3 = v1_dir - v0_dir;
            auto edge_pt3 = v0_dir + sample.t * v_dir3;
            // Project back to screen space
            auto edge_pt = camera_to_screen(camera, edge_pt3);
            // Reject samples outside of image plane
            if (!in_screen(camera, edge_pt)) {
                // In theory this shouldn't happen since we clamp the edges
                return;
            }

            edge_records[idx].edge = edge;
            edge_records[idx].edge_pt = edge_pt;

            // The edge equation for the fisheye camera is:
            // alpha(p) = dot(p, cross(v0_dir, v1_dir))
            // Thus the half-space normal is cross(v0_dir, v1_dir)
            // Generate two rays at the two sides of the edge
            // We choose the ray offset such that the longer the edge is from
            // the camera, the smaller the offset is.
            auto half_space_normal = normalize(cross(v0_dir, v1_dir));
            auto v0_local = xfm_point(camera.world_to_cam, v0);
            auto v1_local = xfm_point(camera.world_to_cam, v1);
            auto edge_local = v0_local + sample.t * v1_local;
            auto offset = 1e-5f / length(edge_local);
            auto upper_dir = normalize(edge_pt3 + offset * half_space_normal);
            auto upper_pt = camera_to_screen(camera, upper_dir);
            auto upper_ray = sample_primary(camera, upper_pt);
            auto lower_dir = normalize(edge_pt3 - offset * half_space_normal);
            auto lower_pt = camera_to_screen(camera, lower_dir);
            auto lower_ray = sample_primary(camera, lower_pt);
            rays[2 * idx + 0] = upper_ray;
            rays[2 * idx + 1] = lower_ray;

            // Compute the corresponding backprop derivatives
            auto xi = int(edge_pt[0] * camera.width);
            auto yi = int(edge_pt[1] * camera.height);
            auto rd = channel_info.radiance_dimension;
            auto d_color = Vector3{
                d_rendered_image[nd * (yi * camera.width + xi) + rd + 0],
                d_rendered_image[nd * (yi * camera.width + xi) + rd + 1],
                d_rendered_image[nd * (yi * camera.width + xi) + rd + 2]
            };
            // The weight is the length of edge divided by the probability
            // of selecting this edge, divided by the length of gradients
            // of the edge equation w.r.t. screen coordinate.
            // For perspective projection the length of edge and gradients
            // cancel each other out.
            // For fisheye we need to compute the Jacobians
            auto upper_weight = d_color / edges_pmf[edge_id];
            auto lower_weight = -d_color / edges_pmf[edge_id];

            // alpha(p(x, y)) = dot(p(x, y), cross(v0_dir, v1_dir))
            // p = screen_to_camera(x, y)
            // dp/dx & dp/dy
            auto d_edge_dir_x = Vector3{0, 0, 0};
            auto d_edge_dir_y = Vector3{0, 0, 0};
            d_screen_to_camera(camera, edge_pt, d_edge_dir_x, d_edge_dir_y);
            // d alpha / d p = cross(v0_dir, v1_dir)
            auto d_alpha_dx = dot(d_edge_dir_x, cross(v0_dir, v1_dir));
            auto d_alpha_dy = dot(d_edge_dir_y, cross(v0_dir, v1_dir));
            auto dirac_jacobian = 1.f / sqrt(square(d_alpha_dx) + square(d_alpha_dy));
            // We use finite difference to compute the Jacobian
            // for sampling on the line
            auto jac_offset = Real(1e-6);
            auto edge_pt3_delta = v0_dir + (sample.t + jac_offset) * v_dir3;
            auto edge_pt_delta = camera_to_screen(camera, edge_pt3_delta);
            auto line_jacobian = length((edge_pt_delta - edge_pt) / offset);
            auto jacobian = line_jacobian * dirac_jacobian;
            upper_weight *= jacobian;
            lower_weight *= jacobian;

            assert(isfinite(upper_weight));

            throughputs[2 * idx + 0] = upper_weight;
            throughputs[2 * idx + 1] = lower_weight;
            for (int d = 0; d < nd; d++) {
                auto d_channel = d_rendered_image[nd * (yi * camera.width + xi) + d];
                channel_multipliers[2 * nd * idx + d] =
                    d_channel * jacobian / edges_pmf[edge_id];
                channel_multipliers[2 * nd * idx + d + nd] =
                    -d_channel * jacobian / edges_pmf[edge_id];
            }
        }

        // Ray differential computation
        auto screen_pos = edge_records[idx].edge_pt;
        auto ray = sample_primary(camera, screen_pos);
        auto delta = Real(1e-3);
        auto screen_pos_dx = screen_pos + Vector2{delta, Real(0)};
        auto ray_dx = sample_primary(camera, screen_pos_dx);
        auto screen_pos_dy = screen_pos + Vector2{Real(0), delta};
        auto ray_dy = sample_primary(camera, screen_pos_dy);
        auto pixel_size_x = Real(0.5) / camera.width;
        auto pixel_size_y = Real(0.5) / camera.height;
        auto org_dx = pixel_size_x * (ray_dx.org - ray.org) / delta;
        auto org_dy = pixel_size_y * (ray_dy.org - ray.org) / delta;
        auto dir_dx = pixel_size_x * (ray_dx.dir - ray.dir) / delta;
        auto dir_dy = pixel_size_y * (ray_dy.dir - ray.dir) / delta;
        primary_ray_differentials[idx] = RayDifferential{org_dx, org_dy, dir_dx, dir_dy};
    }

    const Camera camera;
    const Shape *shapes;
    const Edge *edges;
    int num_edges;
    const Real *edges_pmf;
    const Real *edges_cdf;
    const PrimaryEdgeSample *samples;
    const float *d_rendered_image;
    const ChannelInfo channel_info;
    PrimaryEdgeRecord *edge_records;
    Ray *rays;
    RayDifferential *primary_ray_differentials;
    Vector3 *throughputs;
    Real *channel_multipliers;
};

void sample_primary_edges(const Scene &scene,
                          const BufferView<PrimaryEdgeSample> &samples,
                          const float *d_rendered_image,
                          const ChannelInfo &channel_info,
                          BufferView<PrimaryEdgeRecord> edge_records,
                          BufferView<Ray> rays,
                          BufferView<RayDifferential> primary_ray_differentials,
                          BufferView<Vector3> throughputs,
                          BufferView<Real> channel_multipliers) {
    parallel_for(primary_edge_sampler{
        scene.camera,
        scene.shapes.data,
        scene.edge_sampler.edges.begin(),
        (int)scene.edge_sampler.edges.size(),
        scene.edge_sampler.primary_edges_pmf.begin(),
        scene.edge_sampler.primary_edges_cdf.begin(),
        samples.begin(),
        d_rendered_image,
        channel_info,
        edge_records.begin(),
        rays.begin(),
        primary_ray_differentials.begin(),
        throughputs.begin(),
        channel_multipliers.begin()
    }, samples.size(), scene.use_gpu);
}

struct primary_edge_weights_updater {
    DEVICE void operator()(int idx) {
        const auto &edge_record = edge_records[idx];
        auto isect_upper = shading_isects[2 * idx + 0];
        auto isect_lower = shading_isects[2 * idx + 1];
        auto &throughputs_upper = throughputs[2 * idx + 0];
        auto &throughputs_lower = throughputs[2 * idx + 1];
        // At least one of the intersections should be connected to the edge
        bool upper_connected = isect_upper.shape_id == edge_record.edge.shape_id &&
            (isect_upper.tri_id == edge_record.edge.f0 || isect_upper.tri_id == edge_record.edge.f1);
        bool lower_connected = isect_lower.shape_id == edge_record.edge.shape_id &&
            (isect_lower.tri_id == edge_record.edge.f0 || isect_lower.tri_id == edge_record.edge.f1);
        if (!upper_connected && !lower_connected) {
            throughputs_upper = Vector3{0, 0, 0};
            throughputs_lower = Vector3{0, 0, 0};
            auto nd = channel_info.num_total_dimensions;
            for (int d = 0; d < nd; d++) {
                channel_multipliers[2 * nd * idx + d] = 0;
                channel_multipliers[2 * nd * idx + d + nd] = 0;
            }
        }
    }

    const PrimaryEdgeRecord *edge_records;
    const Intersection *shading_isects;
    const ChannelInfo channel_info;
    Vector3 *throughputs;
    Real *channel_multipliers;
};

void update_primary_edge_weights(const Scene &scene,
                                 const BufferView<PrimaryEdgeRecord> &edge_records,
                                 const BufferView<Intersection> &edge_isects,
                                 const ChannelInfo &channel_info,
                                 BufferView<Vector3> throughputs,
                                 BufferView<Real> channel_multipliers) {
    // XXX: Disable this at the moment. Not sure if this is more robust or not.
    // parallel_for(primary_edge_weights_updater{
    //     edge_records.begin(),
    //     edge_isects.begin(),
    //     channel_info,
    //     throughputs.begin(),
    //     channel_multipliers.begin()
    // }, edge_records.size(), scene.use_gpu);
}

struct primary_edge_derivatives_computer {
    DEVICE void operator()(int idx) {
        const auto &edge_record = edge_records[idx];
        auto edge_contrib_upper = edge_contribs[2 * idx + 0];
        auto edge_contrib_lower = edge_contribs[2 * idx + 1];
        auto edge_contrib = edge_contrib_upper + edge_contrib_lower;

        // Initialize derivatives
        if (edge_record.edge.shape_id < 0) {
            return;
        }

        auto v0 = Vector3{get_v0(shapes, edge_record.edge)};
        auto v1 = Vector3{get_v1(shapes, edge_record.edge)};
        auto v0_ss = Vector2{0, 0};
        auto v1_ss = Vector2{0, 0};
        if (!project(camera, v0, v1, v0_ss, v1_ss)) {
            return;
        }
        auto d_v0_ss = Vector2{0, 0};
        auto d_v1_ss = Vector2{0, 0};
        auto edge_pt = edge_record.edge_pt;
        if (camera.camera_type != CameraType::Fisheye) {
            // Equation 8 in the paper
            d_v0_ss.x = v1_ss.y - edge_pt.y;
            d_v0_ss.y = edge_pt.x - v1_ss.x;
            d_v1_ss.x = edge_pt.y - v0_ss.y;
            d_v1_ss.y = v0_ss.x - edge_pt.x;
        } else {
            // This also works for perspective camera,
            // but for consistency we provide two versions.
            // alpha(p) = dot(p, cross(v0_dir, v1_dir))
            // v0_dir = screen_to_camera(v0_ss)
            // v1_dir = screen_to_camera(v1_ss)
            // d alpha / d v0_ss_x = dot(cross(v1_dir, p),
            //     d_screen_to_camera(v0_ss).x)
            auto v0_dir = screen_to_camera(camera, v0_ss);
            auto v1_dir = screen_to_camera(camera, v1_ss);
            auto edge_dir = screen_to_camera(camera, edge_pt);
            auto d_v0_dir_x = Vector3{0, 0, 0};
            auto d_v0_dir_y = Vector3{0, 0, 0};
            d_screen_to_camera(camera, v0_ss, d_v0_dir_x, d_v0_dir_y);
            auto d_v1_dir_x = Vector3{0, 0, 0};
            auto d_v1_dir_y = Vector3{0, 0, 0};
            d_screen_to_camera(camera, v1_ss, d_v1_dir_x, d_v1_dir_y);
            d_v0_ss.x = dot(cross(v1_dir, edge_dir), d_v0_dir_x);
            d_v0_ss.y = dot(cross(v1_dir, edge_dir), d_v0_dir_y);
            d_v1_ss.x = dot(cross(edge_dir, v0_dir), d_v1_dir_x);
            d_v1_ss.y = dot(cross(edge_dir, v0_dir), d_v1_dir_y);
        }
        d_v0_ss *= edge_contrib;
        d_v1_ss *= edge_contrib;

        // v0_ss, v1_ss = project(camera, v0, v1)
        auto d_v0 = Vector3{0, 0, 0};
        auto d_v1 = Vector3{0, 0, 0};
        d_project(camera, v0, v1,
            d_v0_ss.x, d_v0_ss.y,
            d_v1_ss.x, d_v1_ss.y,
            d_camera, d_v0, d_v1);
        atomic_add(&d_shapes[edge_record.edge.shape_id].vertices[3 * edge_record.edge.v0], d_v0);
        atomic_add(&d_shapes[edge_record.edge.shape_id].vertices[3 * edge_record.edge.v1], d_v1);

    }

    const Camera camera;
    const Shape *shapes;
    const PrimaryEdgeRecord *edge_records;
    const Real *edge_contribs;
    DShape *d_shapes;
    DCamera d_camera;
};

void compute_primary_edge_derivatives(const Scene &scene,
                                      const BufferView<PrimaryEdgeRecord> &edge_records,
                                      const BufferView<Real> &edge_contribs,
                                      BufferView<DShape> d_shapes,
                                      DCamera d_camera) {
    parallel_for(primary_edge_derivatives_computer{
        scene.camera,
        scene.shapes.data,
        edge_records.begin(),
        edge_contribs.begin(),
        d_shapes.begin(), d_camera
    }, edge_records.size(), scene.use_gpu);
}

struct BVHStackItem {
    BVHNodePtr node_ptr;
};

struct secondary_edge_sampler {
    DEVICE bool contains_silhouette(const BVHNodePtr &node_ptr,
                                    const Vector3 &p) {
        if (node_ptr.is_bvh_node3) {
            return true;
        } else {
            auto bounds = node_ptr.ptr6->bounds;
            auto d_bounds = AABB3{bounds.d_min, bounds.d_max};
            return intersect(Sphere{0.5f * (p - cam_org),
                0.5f * distance(p, cam_org)}, d_bounds);
        }
    }

    template <typename BVHNodeType>
    DEVICE Real leaf_importance(const BVHNodeType &node,
                                const SurfacePoint &p,
                                const SurfacePoint &occluder_point) {
        const auto &edge = edges[node.edge_id];
        if (!is_silhouette(scene.shapes, p.position, edge)) {
            return 0;
        }
        // Project occluder_point onto the edge. If the distance
        // is larger than edge_cylinder_radius than return 0.
        auto v0 = Vector3{get_v0(scene.shapes, edge)};
        auto v1 = Vector3{get_v1(scene.shapes, edge)};
        auto e01 = v1 - v0;
        auto t = (dot(occluder_point.position - v0, e01) / dot(e01, e01));
        if (t < 0 || t > 1) {
            return 0;
        }
        auto edge_pt = v0 + t * e01;
        auto dist = distance(edge_pt, occluder_point.position);
        if (dist > edge_cylinder_radius) {
            return 0;
        }
        return 1;
    }

    DEVICE Real leaf_importance(const BVHNodePtr &node_ptr,
                                const SurfacePoint &p,
                                const SurfacePoint &occluder_point) {
        if (node_ptr.is_bvh_node3) {
            return leaf_importance(*node_ptr.ptr3, p, occluder_point);
        } else {
            return leaf_importance(*node_ptr.ptr6, p, occluder_point);
        }
    }

    DEVICE bool inside(const BVHNodePtr &node_ptr, const SurfacePoint &p) {
        if (node_ptr.is_bvh_node3) {
            return ::inside(node_ptr.ptr3->bounds, p.position, edge_cylinder_radius);
        } else {
            return ::inside(node_ptr.ptr6->bounds, p.position, edge_cylinder_radius);
        }
    }

    DEVICE int sample_edge_cylinder(const EdgeTreeRoots &edge_tree_roots,
                                    const SurfacePoint &p,
                                    const SurfacePoint &occluder_point,
                                    Real &resample_sample,
                                    Real &sample_weight,
                                    Vector3 &edge_pt) {
        constexpr auto buffer_size = 128;
        BVHStackItem buffer[buffer_size];
        auto selected_edge = -1;
        auto edge_weight = Real(0);
        auto wsum = Real(0);

        auto stack_ptr = &buffer[0];
        // randomly sample a point on an edge by collecting 
        // all edge cylinders containing occluder_point
        // push both nodes into stack
        if (edge_tree_roots.cs_bvh_root != nullptr) {
            *stack_ptr++ = BVHStackItem{
                BVHNodePtr{edge_tree_roots.cs_bvh_root}};
        }
        if (edge_tree_roots.ncs_bvh_root != nullptr) {
            *stack_ptr++ = BVHStackItem{
                BVHNodePtr{edge_tree_roots.ncs_bvh_root}};
        }
        while (stack_ptr != &buffer[0]) {
            assert(stack_ptr > &buffer[0] && stack_ptr < &buffer[buffer_size]);
            // pop from stack
            const auto &stack_item = *--stack_ptr;
            if (is_leaf(stack_item.node_ptr)) {
                auto w = leaf_importance(stack_item.node_ptr, p, occluder_point);
                if (w > 0) {
                    auto prev_wsum = wsum;
                    wsum += w;
                    auto normalized_w = w / wsum;
                    if (resample_sample <= normalized_w || prev_wsum == 0) {
                        selected_edge = get_edge_id(stack_item.node_ptr);
                        edge_weight = w;
                        // rescale sample to [0, 1]
                        resample_sample /= normalized_w;
                    } else {
                        // rescale sample to [0, 1]
                        resample_sample = (resample_sample - normalized_w) / (1 - normalized_w);
                    }
                }
            } else {
                BVHNodePtr children[2];
                get_children(stack_item.node_ptr, children);
                if (contains_silhouette(children[0], p.position) &&
                        inside(children[0], occluder_point)) {
                    *stack_ptr++ = BVHStackItem{BVHNodePtr(children[0])};
                }
                if (contains_silhouette(children[1], p.position) &&
                        inside(children[1], occluder_point)) {
                    *stack_ptr++ = BVHStackItem{BVHNodePtr(children[1])};
                }
            }
        }
        if (selected_edge == -1) {
            return -1;
        }

        auto pmf = edge_weight / wsum;
        if (pmf <= 0) {
            return -1;
        }
        // Project isect_point to edge, compute distance
        const auto &edge = edges[selected_edge];
        auto v0 = Vector3{get_v0(scene.shapes, edge)};
        auto v1 = Vector3{get_v1(scene.shapes, edge)};
        auto e01 = v1 - v0;
        auto t = (dot(occluder_point.position - v0, e01) / dot(e01, e01));
        if (t < 0 || t > 1) {
            return -1;
        }
        edge_pt = v0 + t * e01;
        auto dist = distance(edge_pt, occluder_point.position);
        if (dist > edge_cylinder_radius) {
            return -1;
        }
        // HACK: approximte the kernel normalization factor by line length R
        // This makes the estimation biased. The proper way to do this is to estimate
        // the kernel length by tracing rays to take geometry into consideration.
        // See Qin et al. 2015.
        sample_weight = 1 / (edge_cylinder_radius * pmf);
        return selected_edge;
    }

    DEVICE Real collect_pdf_nee(const Ray &ray,
                                const SurfacePoint &occluder_point,
                                const Vector3 &edge_v0,
                                const Vector3 &edge_v1) {
        constexpr auto buffer_size = 128;
        const LightBVHNode* buffer[buffer_size];
        auto stack_ptr = &buffer[0];
        // push root into stack
        if (edge_tree_roots.light_bvh_root != nullptr) {
            *stack_ptr++ = edge_tree_roots.light_bvh_root;
        }
        auto pdf_sum = Real(0);
        while (stack_ptr != &buffer[0]) {
            assert(stack_ptr > &buffer[0] && stack_ptr < &buffer[buffer_size]);
            // pop from stack
            const auto &node = *--stack_ptr;
            if (node->shape_id != -1) {
                // leaf node. intersect with the triangle.
                const auto &shape = scene.shapes[node->shape_id];
                auto index = get_indices(shape, node->tri_id);
                auto v0 = Vector3{get_vertex(shape, index[0])};
                auto v1 = Vector3{get_vertex(shape, index[1])};
                auto v2 = Vector3{get_vertex(shape, index[2])};
                auto hit_p = Vector3{0, 0, 0};
                auto hit_n = Vector3{0, 0, 0};
                if (intersect(v0, v1, v2, ray, &hit_p, &hit_n)) {
                    auto light_pmf = scene.light_pmf[shape.light_id];
                    auto light_area = scene.light_areas[shape.light_id];
                    // Intersection Jacobian
                    // Intersecting the plane (occluder_point, occluder_normal)
                    // with ray (shading_point, nee_point - shading_point)
                    // let w = nee_point - shading_point
                    // po = nee_occluder_point
                    // no = nee_occluder_normal
                    // t = dot(np, no) / dot(w, no)
                    // m = shading_point + w * t
                    // mp = edge_v0 + dot(m - edge_v0, edge_v1 - edge_v0) / 
                    //                dot(edge_v1 - edge_v0, edge_v1 - edge_v0) * (edge_v1 - edge_v0)
                    // tp = dot(m - edge_v0, edge_v1 - edge_v0) / 
                    //      dot(edge_v1 - edge_v0, edge_v1 - edge_v0)
                    // sp = length(mp - m)
                    auto hit_geom_frame = Frame(hit_n);
                    auto w = hit_p - ray.org;
                    auto w_du = hit_geom_frame.x;
                    auto w_dv = hit_geom_frame.y;
                    auto dot_w_no = dot(w, occluder_point.geom_normal);
                    auto dot_w_no_du = dot(w_du, occluder_point.geom_normal);
                    auto dot_w_no_dv = dot(w_dv, occluder_point.geom_normal);
                    auto inv_dot_w_no = 1 / dot_w_no;
                    auto inv_dot_w_no_du = -dot_w_no_du * inv_dot_w_no / dot_w_no;
                    auto inv_dot_w_no_dv = -dot_w_no_dv * inv_dot_w_no / dot_w_no;
                    auto t = dot(occluder_point.position - ray.org, occluder_point.geom_normal) *
                             inv_dot_w_no;
                    auto t_du = dot(occluder_point.position - ray.org, occluder_point.geom_normal) *
                                inv_dot_w_no_du;
                    auto t_dv = dot(occluder_point.position - ray.org, occluder_point.geom_normal) *
                                inv_dot_w_no_dv;
                    auto m = ray.org + w * t;
                    auto m_du = w_du * t + w * t_du;
                    auto m_dv = w_dv * t + w * t_dv;
                    auto tp = dot(m - edge_v0, edge_v1 - edge_v0) /
                              dot(edge_v1 - edge_v0, edge_v1 - edge_v0);
                    auto tp_du = dot(m_du, edge_v1 - edge_v0) /
                                 dot(edge_v1 - edge_v0, edge_v1 - edge_v0);
                    auto tp_dv = dot(m_dv, edge_v1 - edge_v0) /
                                 dot(edge_v1 - edge_v0, edge_v1 - edge_v0);
                    auto mp = edge_v0 + tp * (edge_v1 - edge_v0);
                    auto mp_du = tp_du * (edge_v1 - edge_v0);
                    auto mp_dv = tp_dv * (edge_v1 - edge_v0);
                    auto sp_sq = dot(mp - m, mp - m);
                    auto sp_sq_du = 2.f * sum((mp_du - m_du) * (mp - m));
                    auto sp_sq_dv = 2.f * sum((mp_dv - m_dv) * (mp - m));
                    auto sp = sqrt(sp_sq);
                    auto sp_du = Real(0);
                    auto sp_dv = Real(0);
                    if (sp > 0) {
                        sp_du = 0.5f * sp_sq_du / sp;
                        sp_dv = 0.5f * sp_sq_dv / sp;
                    } else {
                        // L'Hopital's rule
                        sp_du = sum(mp_du - m_du);
                        sp_dv = sum(mp_dv - m_dv);
                    }
                    auto jacobian = sqrt(tp_du*tp_du + tp_dv*tp_dv) *
                                    sqrt(sp_du*sp_du + sp_dv*sp_dv);
                    assert(isfinite(jacobian));
                    pdf_sum += light_pmf / (jacobian * light_area);
                }
            } else {
                const LightBVHNode* children[2] = {
                    node->children[0], node->children[1]};
                if (intersect(children[0]->bounds, ray)) {
                    *stack_ptr++ = children[0];
                }
                if (intersect(children[1]->bounds, ray)) {
                    *stack_ptr++ = children[1];
                }
            }
        }
        // Environment map
        if (scene.envmap != nullptr) {
            // Intersecting occluder: convert to area measure, then project to t/s space
            auto diff = occluder_point.position - ray.org;
            auto dist_sq = length_squared(diff);
            auto dist = sqrt(dist_sq);
            auto wo = diff / dist;
            auto geometry_term = dot(occluder_point.geom_normal, wo) / dist_sq;
            auto geom_frame = Frame(occluder_point.geom_normal);
            auto m = occluder_point.position;
            auto m_du = geom_frame.x;
            auto m_dv = geom_frame.y;
            auto tp = dot(m - edge_v0, edge_v1 - edge_v0) /
                      dot(edge_v1 - edge_v0, edge_v1 - edge_v0);
            auto tp_du = dot(m_du, edge_v1 - edge_v0) /
                         dot(edge_v1 - edge_v0, edge_v1 - edge_v0);
            auto tp_dv = dot(m_dv, edge_v1 - edge_v0) /
                         dot(edge_v1 - edge_v0, edge_v1 - edge_v0);
            auto mp = edge_v0 + tp * (edge_v1 - edge_v0);
            auto mp_du = tp_du * (edge_v1 - edge_v0);
            auto mp_dv = tp_dv * (edge_v1 - edge_v0);
            auto sp_sq = dot(mp - m, mp - m);
            auto sp_sq_du = 2.f * sum((mp_du - m_du) * (mp - m));
            auto sp_sq_dv = 2.f * sum((mp_dv - m_dv) * (mp - m));
            auto sp = sqrt(sp_sq);
            auto sp_du = 0.5f * sp_sq_du / sp;
            auto sp_dv = 0.5f * sp_sq_dv / sp;
            auto jacobian = sqrt(tp_du*tp_du + tp_dv*tp_dv) *
                            sqrt(sp_du*sp_du + sp_dv*sp_dv);
            pdf_sum += envmap_pdf(*scene.envmap, ray.dir) * geometry_term / jacobian;
        }
        return pdf_sum;
    }

    DEVICE Real compute_bsdf_pdf(const Vector3 &wi,
                                 const Material &material,
                                 const Real min_rough,
                                 const SurfacePoint &bsdf_point,    
                                 const SurfacePoint &shading_point,
                                 const Vector3 &edge_v0,
                                 const Vector3 &edge_v1) {
        auto diff = bsdf_point.position - shading_point.position;
        auto dist_sq = length_squared(diff);
        auto dist = sqrt(dist_sq);
        auto wo = diff / dist;
        auto geometry_term = dot(bsdf_point.geom_normal, wo) / dist_sq;
        auto geom_frame = Frame(bsdf_point.geom_normal);
        auto m = bsdf_point.position;
        auto m_du = geom_frame.x;
        auto m_dv = geom_frame.y;
        auto tp = dot(m - edge_v0, edge_v1 - edge_v0) /
                  dot(edge_v1 - edge_v0, edge_v1 - edge_v0);
        auto tp_du = dot(m_du, edge_v1 - edge_v0) /
                     dot(edge_v1 - edge_v0, edge_v1 - edge_v0);
        auto tp_dv = dot(m_dv, edge_v1 - edge_v0) /
                     dot(edge_v1 - edge_v0, edge_v1 - edge_v0);
        auto mp = edge_v0 + tp * (edge_v1 - edge_v0);
        auto mp_du = tp_du * (edge_v1 - edge_v0);
        auto mp_dv = tp_dv * (edge_v1 - edge_v0);
        auto sp_sq = dot(mp - m, mp - m);
        auto sp_sq_du = 2.f * sum((mp_du - m_du) * (mp - m));
        auto sp_sq_dv = 2.f * sum((mp_dv - m_dv) * (mp - m));
        auto sp = sqrt(sp_sq);
        auto sp_du = 0.5f * sp_sq_du / sp;
        auto sp_dv = 0.5f * sp_sq_dv / sp;
        auto jacobian = sqrt(tp_du*tp_du + tp_dv*tp_dv) *
                        sqrt(sp_du*sp_du + sp_dv*sp_dv);
        return bsdf_pdf(material, shading_point, wi, wo, min_rough) *
               geometry_term / jacobian;
    }
    
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &edge_sample = edge_samples[idx];
        const auto &wi = -incoming_rays[pixel_id].dir;
        const auto &shading_isect = shading_isects[pixel_id];
        const auto &shading_point = shading_points[pixel_id];
        const auto &throughput = throughputs[pixel_id];
        const auto &min_rough = min_roughness[pixel_id];
        const auto &nee_occluder_isect = nee_occluder_isects[pixel_id];
        const auto &nee_occluder_point = nee_occluder_points[pixel_id];
        const auto &bsdf_isect = bsdf_isects[pixel_id];
        const auto &bsdf_point = bsdf_points[pixel_id];

        // Initialize output
        edge_records[idx] = SecondaryEdgeRecord{};
        new_throughputs[2 * idx + 0] = Vector3{0, 0, 0};
        new_throughputs[2 * idx + 1] = Vector3{0, 0, 0};
        rays[2 * idx + 0] = Ray(Vector3{0, 0, 0}, Vector3{0, 0, 0});
        rays[2 * idx + 1] = Ray(Vector3{0, 0, 0}, Vector3{0, 0, 0});
        edge_min_roughness[2 * idx + 0] = min_rough;
        edge_min_roughness[2 * idx + 1] = min_rough;

        // XXX Hack: don't compute secondary edge derivatives if we already hit a diffuse vertex
        // before shading_point.
        // Such paths are extremely noisy and have very small contribution to the actual derivatives.
        if (min_rough > 1e-2f) {
            return;
        }

        const Shape &shape = scene.shapes[shading_isect.shape_id];
        const Material &material = scene.materials[shape.material_id];
        auto diffuse_reflectance = get_diffuse_reflectance(material, shading_point);
        auto specular_reflectance = get_specular_reflectance(material, shading_point);
        auto diffuse_weight = luminance(diffuse_reflectance);
        auto specular_weight = luminance(specular_reflectance);
        auto weight_sum = diffuse_weight + specular_weight;
        if (weight_sum <= 0.f) {
            // black material
            return;
        }

        // auto x = pixel_id % 256;
        // auto y = pixel_id / 256;
        // auto debug = x == 213 && y == 180;

        // We sample the secondary edges by reusing NEE & BSDF importance sampling,
        // since these rays usually represent where the importance contributions are.
        // However, edges are 1D line segments in 3D space, the probability of 
        // regular importance sampling hitting the line segment is exactly 0.
        // The way we deal with this is to convolve each point on the edge with
        // a disc, and splat the contribution of the edges onto the surfaces.
        // More precisely, we transform the edge integral
        // \int f(t) dt
        // to an integral on the scene surface manifold M
        // \int\int f(t) k(s) dt ds / \int k(s) ds
        // where s is a line from point t to the geometry.
        // As long as we ensure \int k(s) ds = 1, this doesn't introduce any bias.
        // We use a constant kernel k here, by splatting the edge contribution to
        // the intersection between a cylinder surrounding the edge and the scene manifold.
        // The reciprocal of \int k(s) ds can be estimated by Monte Carlo, but
        // for now we use an approximation by assuming it's always the radius of the cylinder.
        auto edge_sel = edge_sample.edge_sel;
        auto nee_edge_id = -1;
        auto nee_edge_weight = Real(0);
        auto nee_edge_pt = Vector3{0, 0, 0};
        auto nee_mis_weight = Real(0);
        if (nee_occluder_isect.valid()) {
            nee_edge_id = sample_edge_cylinder(
                edge_tree_roots, shading_point, nee_occluder_point,
                edge_sel, nee_edge_weight, nee_edge_pt);
            if (nee_edge_id != -1) {
                // Take nee pdf into consideration
                // Importantly, we don't just take the PDF of the light source we sampled from.
                // We take the PDFs from *all* light sources that is hit by the nee ray,
                // since they all would hit this occluder.
                auto v0 = Vector3{get_v0(scene.shapes, edges[nee_edge_id])};
                auto v1 = Vector3{get_v1(scene.shapes, edges[nee_edge_id])};
                auto wo = normalize(nee_occluder_point.position - shading_point.position);
                auto pdf_nee = collect_pdf_nee(
                    Ray{shading_point.position, wo, Real(1e-3)},
                    nee_occluder_point,
                    v0, v1);
                assert(isfinite(pdf_nee));
                if (pdf_nee > 0) {
                    nee_edge_weight /= pdf_nee;
                    auto pdf_bsdf = compute_bsdf_pdf(
                        wi, material, min_rough, nee_occluder_point, shading_point,
                        v0, v1);
                    nee_mis_weight = pdf_nee / (pdf_nee + pdf_bsdf);
                } else {
                    nee_edge_id = -1;
                    nee_edge_weight = 0;
                }
            }
        }
        auto bsdf_edge_id = -1;
        auto bsdf_edge_weight = Real(0);
        auto bsdf_edge_pt = Vector3{0, 0, 0};
        auto bsdf_mis_weight = Real(0);
        if (bsdf_isect.valid()) {
            bsdf_edge_id = sample_edge_cylinder(
                edge_tree_roots, shading_point, bsdf_point,
                edge_sel, bsdf_edge_weight, bsdf_edge_pt);
            if (bsdf_edge_id != -1) {
                auto v0 = Vector3{get_v0(scene.shapes, edges[bsdf_edge_id])};
                auto v1 = Vector3{get_v1(scene.shapes, edges[bsdf_edge_id])};
                auto pdf_bsdf = compute_bsdf_pdf(
                    wi, material, min_rough, bsdf_point, shading_point,
                    v0, v1);
                if (pdf_bsdf > 0) {
                    bsdf_edge_weight /= pdf_bsdf;
                    // auto wo = normalize(bsdf_point.position - shading_point.position);
                    // Hack: use bsdf_edge_pt for computing PDFs.
                    // This should gives lower variance without introducing bias.
                    auto wo = normalize(bsdf_edge_pt - shading_point.position);
                    auto new_bsdf_point = bsdf_point;
                    new_bsdf_point.position = bsdf_edge_pt;
                    auto pdf_nee = collect_pdf_nee(
                        Ray{shading_point.position, wo, Real(1e-3)},
                        new_bsdf_point,
                        v0, v1);
                    bsdf_mis_weight = pdf_bsdf / (pdf_nee + pdf_bsdf);
                }
            }
        }
        if (nee_mis_weight + bsdf_mis_weight <= 0) {
            return;
        }
        // Choose one between nee & bsdf sample.
        // We use MIS weight to decide which one to sample
        auto nee_prob = nee_mis_weight / (nee_mis_weight + bsdf_mis_weight);
        auto bsdf_prob = bsdf_mis_weight / (nee_mis_weight + bsdf_mis_weight);
        auto edge_id = -1;
        auto edge_weight = 0.f;
        auto edge_pt = Vector3{0, 0, 0};
        if (edge_sel <= nee_prob) {
            edge_id = nee_edge_id;
            edge_weight = nee_edge_weight * nee_mis_weight / nee_prob;
            assert(isfinite(edge_weight));
            edge_pt = nee_edge_pt;
        } else {
            edge_id = bsdf_edge_id;
            edge_weight = bsdf_edge_weight * bsdf_mis_weight / bsdf_prob;
            edge_pt = bsdf_edge_pt;
        }
        if (edge_id == -1) {
            return;
        }
        auto sample_p = edge_pt - shading_point.position;

        const auto &edge = edges[edge_id];
        auto v0 = Vector3{get_v0(scene.shapes, edge)};
        auto v1 = Vector3{get_v1(scene.shapes, edge)};
        // shading_point.position, v0 and v1 forms a half-plane
        // that splits the spaces into upper half-space and lower half-space
        auto half_plane_normal =
            normalize(cross(v0 - shading_point.position,
                            v1 - shading_point.position));
        // Generate sample directions
        auto offset = 1e-5f / length(sample_p);
        auto sample_dir = normalize(sample_p);
        // Sample two rays on the two sides of the edge
        auto v_upper_dir = normalize(sample_dir + offset * half_plane_normal);
        auto v_lower_dir = normalize(sample_dir - offset * half_plane_normal);

        auto eval_bsdf = bsdf(material, shading_point, wi, sample_dir, min_rough);
        if (sum(eval_bsdf) < 1e-6f) {
            return;
        }

        // Setup output
        auto nd = channel_info.num_total_dimensions;
        auto rd = channel_info.radiance_dimension;
        auto d_color = Vector3{
            d_rendered_image[nd * pixel_id + rd + 0],
            d_rendered_image[nd * pixel_id + rd + 1],
            d_rendered_image[nd * pixel_id + rd + 2]
        };
        edge_records[idx].edge = edge;
        edge_records[idx].edge_pt = sample_p; // for Jacobian computation 
        rays[2 * idx + 0] = Ray(shading_point.position, v_upper_dir, 1e-3f * length(sample_p));
        rays[2 * idx + 1] = Ray(shading_point.position, v_lower_dir, 1e-3f * length(sample_p));
        const auto &incoming_ray_differential = incoming_ray_differentials[pixel_id];
        // Propagate ray differentials
        auto bsdf_ray_differential = RayDifferential{};
        bsdf_ray_differential.org_dx = incoming_ray_differential.org_dx;
        bsdf_ray_differential.org_dy = incoming_ray_differential.org_dy;
        // Decide which component of BRDF to sample
        auto diffuse_pmf = diffuse_weight / weight_sum;
        if (edge_sample.bsdf_component <= diffuse_pmf) {
            // HACK: Output direction has no dependencies w.r.t. input
            // However, since the diffuse BRDF serves as a low pass filter,
            // we want to assign a larger prefilter.
            bsdf_ray_differential.dir_dx = Vector3{0.03f, 0.03f, 0.03f};
            bsdf_ray_differential.dir_dy = Vector3{0.03f, 0.03f, 0.03f};
        } else {
            // HACK: we compute the half vector as the micronormal,
            // and use dndx/dndy to approximate the micronormal screen derivatives
            auto m = normalize(wi + sample_dir);
            auto m_local2 = dot(m, shading_point.shading_frame.n);
            auto dmdx = shading_point.dn_dx * m_local2;
            auto dmdy = shading_point.dn_dy * m_local2;
            auto dir_dx = incoming_ray_differential.dir_dx;
            auto dir_dy = incoming_ray_differential.dir_dy;
            // Igehy 1999, Equation 15
            auto ddotn_dx = dir_dx * m - wi * dmdx;
            auto ddotn_dy = dir_dy * m - wi * dmdy;
            // Igehy 1999, Equation 14
            bsdf_ray_differential.dir_dx =
                dir_dx - 2 * (-dot(wi, m) * shading_point.dn_dx + ddotn_dx * m);
            bsdf_ray_differential.dir_dy =
                dir_dy - 2 * (-dot(wi, m) * shading_point.dn_dy + ddotn_dy * m);
        }
        bsdf_differentials[2 * idx + 0] = bsdf_ray_differential;
        bsdf_differentials[2 * idx + 1] = bsdf_ray_differential;
        // edge_weight doesn't take the Jacobian between the shading point
        // and the ray intersection into account. We'll compute this later
        auto nt = throughput * eval_bsdf * d_color * edge_weight;
        // assert(isfinite(throughput));
        // assert(isfinite(eval_bsdf));
        // assert(isfinite(d_color));
        assert(isfinite(edge_weight));
        new_throughputs[2 * idx + 0] = nt;
        new_throughputs[2 * idx + 1] = -nt;
    }

    const FlattenScene scene;
    const Edge *edges;
    int num_edges;
    const Vector3 cam_org;
    const EdgeTreeRoots edge_tree_roots;
    const Real edge_cylinder_radius;
    const int *active_pixels;
    const SecondaryEdgeSample *edge_samples;
    const Ray *incoming_rays;
    const RayDifferential *incoming_ray_differentials;
    const Intersection *shading_isects;
    const SurfacePoint *shading_points;
    const Intersection *nee_occluder_isects;
    const SurfacePoint *nee_occluder_points;
    const Intersection *bsdf_isects;
    const SurfacePoint *bsdf_points;
    const Vector3 *throughputs;
    const Real *min_roughness;
    const float *d_rendered_image;
    const ChannelInfo channel_info;
    SecondaryEdgeRecord *edge_records;
    Ray *rays;
    RayDifferential *bsdf_differentials;
    Vector3 *new_throughputs;
    Real *edge_min_roughness;
};

void sample_secondary_edges(const Scene &scene,
                            const BufferView<int> &active_pixels,
                            const BufferView<SecondaryEdgeSample> &samples,
                            const BufferView<Ray> &incoming_rays,
                            const BufferView<RayDifferential> &incoming_ray_differentials,
                            const BufferView<Intersection> &shading_isects,
                            const BufferView<SurfacePoint> &shading_points,
                            const BufferView<Intersection> &nee_occluder_isects,
                            const BufferView<SurfacePoint> &nee_occluder_points,
                            const BufferView<Intersection> &bsdf_isects,
                            const BufferView<SurfacePoint> &bsdf_points,
                            const BufferView<Vector3> &throughputs,
                            const BufferView<Real> &min_roughness,
                            const float *d_rendered_image,
                            const ChannelInfo &channel_info,
                            BufferView<SecondaryEdgeRecord> edge_records,
                            BufferView<Ray> rays,
                            BufferView<RayDifferential> &bsdf_differentials,
                            BufferView<Vector3> new_throughputs,
                            BufferView<Real> edge_min_roughness) {
    auto cam_org = xfm_point(scene.camera.cam_to_world, Vector3{0, 0, 0});
    auto edge_tree = scene.edge_sampler.edge_tree.get();
    parallel_for(secondary_edge_sampler{
        get_flatten_scene(scene),
        scene.edge_sampler.edges.begin(),
        (int)scene.edge_sampler.edges.size(),
        cam_org,
        get_edge_tree_roots(edge_tree),
        edge_tree != nullptr ? edge_tree->edge_cylinder_radius : Real(0),
        active_pixels.begin(),
        samples.begin(),
        incoming_rays.begin(),
        incoming_ray_differentials.begin(),
        shading_isects.begin(),
        shading_points.begin(),
        nee_occluder_isects.begin(),
        nee_occluder_points.begin(),
        bsdf_isects.begin(),
        bsdf_points.begin(),        
        throughputs.begin(),
        min_roughness.begin(),
        d_rendered_image,
        channel_info,
        edge_records.begin(),
        rays.begin(),
        bsdf_differentials.begin(),
        new_throughputs.begin(),
        edge_min_roughness.begin()},
        active_pixels.size(), scene.use_gpu);
}

// The derivative of the intersection point w.r.t. a line parameter t
DEVICE
inline Vector3 intersect_jacobian(const Vector3 &org,
                                  const Vector3 &dir,
                                  const Vector3 &p,
                                  const Vector3 &n,
                                  const Vector3 &l) {
    // Jacobian of ray-plane intersection:
    // https://www.cs.princeton.edu/courses/archive/fall00/cs426/lectures/raycast/sld017.htm
    // d = -(p dot n)
    // t = -(org dot n + d) / (dir dot n)
    // p = org + t * dir
    // d p[i] / d dir[i] = t
    // d p[i] / d t = dir[i]
    // d t / d dir_dot_n = (org dot n - p dot n) / dir_dot_n^2
    // d dir_dot_n / d dir[j] = n[j]
    auto dir_dot_n = dot(dir, n);
    if (fabs(dir_dot_n) < 1e-10f) {
        return Vector3{0.f, 0.f, 0.f};
    }
    auto d = -dot(p, n);
    auto t = -(dot(org, n) + d) / dir_dot_n;
    if (t <= 0) {
        return Vector3{0.f, 0.f, 0.f};
    }
    return t * (l - dir * (dot(l, n) / dot(dir, n)));
}

struct secondary_edge_weights_updater {
    DEVICE void update_throughput(const Intersection &edge_isect,
                                  const SurfacePoint &edge_surface_point,
                                  const SurfacePoint &shading_point,
                                  const SecondaryEdgeRecord &edge_record,
                                  Vector3 &edge_throughput) {
        if (edge_isect.valid()) {
            // Hit a surface
            // Geometry term
            auto dir = edge_surface_point.position - shading_point.position;
            auto dist_sq = length_squared(dir);
            if (dist_sq < 1e-8f) {
                // Likely a self-intersection
                edge_throughput = Vector3{0, 0, 0};
                return;
            }

            auto n_dir = dir / sqrt(dist_sq);
            auto geometry_term = fabs(dot(edge_surface_point.geom_normal, n_dir)) / dist_sq;
            auto v0 = Vector3{get_v0(scene.shapes, edge_record.edge)};
            auto v1 = Vector3{get_v1(scene.shapes, edge_record.edge)};
            // area of projection
            auto half_plane_normal = normalize(cross(v0 - shading_point.position,
                                                     v1 - shading_point.position));
            // Intersection Jacobian Jm(t) (Eq. 18 in the paper)
            auto isect_jacobian = intersect_jacobian(shading_point.position,
                                                     edge_record.edge_pt,
                                                     edge_surface_point.position,
                                                     edge_surface_point.geom_normal,
                                                     v1 - v0);
            // ||J_m|| / ||n_m x n_h|| in Eq. 15 in the paper
            auto line_jacobian = length(isect_jacobian) /
                length(cross(edge_surface_point.geom_normal, half_plane_normal));
            auto p = shading_point.position;
            auto d0 = v0 - p;
            auto d1 = v1 - p;
            auto dirac_jacobian = length(cross(d0, d1)); // Eq. 16 in the paper
            auto w = line_jacobian / dirac_jacobian;

            edge_throughput *= geometry_term * w;
            assert(isfinite(geometry_term));
            assert(isfinite(w));
        } else if (scene.envmap != nullptr) {
            // Hit an environment light
            auto p = shading_point.position;
            auto v0 = Vector3{get_v0(scene.shapes, edge_record.edge)};
            auto v1 = Vector3{get_v1(scene.shapes, edge_record.edge)};
            auto d0 = v0 - p;
            auto d1 = v1 - p;
            auto dirac_jacobian = length(cross(d0, d1)); // Eq. 16 in the paper
            // TODO: check the correctness of this
            auto line_jacobian = 1 / length_squared(edge_record.edge_pt - p);
            auto w = line_jacobian / dirac_jacobian;

            edge_throughput *= w;
        }
    }

    DEVICE void operator()(int idx) {
        const auto &edge_record = edge_records[idx];
        if (edge_record.edge.shape_id < 0) {
            return;
        }

        auto pixel_id = active_pixels[idx];
        const auto &shading_point = shading_points[pixel_id];
        const auto &edge_isect0 = edge_isects[2 * idx + 0];
        const auto &edge_surface_point0 = edge_surface_points[2 * idx + 0];
        const auto &edge_isect1 = edge_isects[2 * idx + 1];
        const auto &edge_surface_point1 = edge_surface_points[2 * idx + 1];
        update_throughput(edge_isect0,
                          edge_surface_point0,
                          shading_point,
                          edge_record,
                          edge_throughputs[2 * idx + 0]);
        update_throughput(edge_isect1,
                          edge_surface_point1,
                          shading_point,
                          edge_record,
                          edge_throughputs[2 * idx + 1]);
    }

    const FlattenScene scene;
    const int *active_pixels;
    const SurfacePoint *shading_points;
    const Intersection *edge_isects;
    const SurfacePoint *edge_surface_points;
    const SecondaryEdgeRecord *edge_records;
    Vector3 *edge_throughputs;
};

void update_secondary_edge_weights(const Scene &scene,
                                   const BufferView<int> &active_pixels,
                                   const BufferView<SurfacePoint> &shading_points,
                                   const BufferView<Intersection> &edge_isects,
                                   const BufferView<SurfacePoint> &edge_surface_points,
                                   const BufferView<SecondaryEdgeRecord> &edge_records,
                                   BufferView<Vector3> edge_throughputs) {
    parallel_for(secondary_edge_weights_updater{
        get_flatten_scene(scene),
        active_pixels.begin(),
        shading_points.begin(),
        edge_isects.begin(),
        edge_surface_points.begin(),
        edge_records.begin(),
        edge_throughputs.begin()},
        active_pixels.size(), scene.use_gpu);
}

struct secondary_edge_derivatives_accumulator {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &shading_point = shading_points[pixel_id];
        const auto &edge_record = edge_records[idx];
        if (edge_record.edge.shape_id < 0) {
            return;
        }

        auto edge_contrib0 = edge_contribs[2 * idx + 0];
        auto edge_contrib1 = edge_contribs[2 * idx + 1];
        const auto &edge_surface_point0 = edge_surface_points[2 * idx + 0];
        const auto &edge_surface_point1 = edge_surface_points[2 * idx + 1];

        auto dcolor_dp = Vector3{0, 0, 0};
        auto dcolor_dv0 = Vector3{0, 0, 0};
        auto dcolor_dv1 = Vector3{0, 0, 0};
        auto v0 = Vector3{get_v0(shapes, edge_record.edge)};
        auto v1 = Vector3{get_v1(shapes, edge_record.edge)};
        auto grad = [&](const Vector3 &p, const Vector3 &x, Real edge_contrib) {
            if (edge_contrib == 0) {
                return;
            }
            auto d0 = v0 - p;
            auto d1 = v1 - p;
            // Eq. 16 in the paper (see the errata)
            auto dp = cross(d1, d0) + cross(x - p, d1) + cross(d0, x - p);
            auto dv0 = cross(d1, x - p);
            auto dv1 = cross(x - p, d0);
            dcolor_dp += dp * edge_contrib;
            dcolor_dv0 += dv0 * edge_contrib;
            dcolor_dv1 += dv1 * edge_contrib;
        };
        grad(shading_point.position, edge_surface_point0, edge_contrib0);
        grad(shading_point.position, edge_surface_point1, edge_contrib1);
        assert(isfinite(edge_contrib0));
        assert(isfinite(edge_contrib1));
        assert(isfinite(dcolor_dp));

        d_points[pixel_id].position += dcolor_dp;
        atomic_add(&(d_shapes[edge_record.edge.shape_id].vertices[3 * edge_record.edge.v0]), dcolor_dv0);
        atomic_add(&(d_shapes[edge_record.edge.shape_id].vertices[3 * edge_record.edge.v1]), dcolor_dv1);
        if (debug_image != nullptr) {
            if (edge_record.edge.shape_id == 1) {
                auto x = pixel_id % 256;
                auto y = pixel_id / 256;
                debug_image[3 * (y * 256 + x) + 0] += dcolor_dv0.x + dcolor_dv1.x;
                debug_image[3 * (y * 256 + x) + 1] += dcolor_dv0.y + dcolor_dv1.y;
                debug_image[3 * (y * 256 + x) + 2] += dcolor_dv0.z + dcolor_dv1.z;
            }
        }
    }

    const Shape *shapes;
    const int *active_pixels;
    const SurfacePoint *shading_points;
    const SecondaryEdgeRecord *edge_records;
    const Vector3 *edge_surface_points;
    const Real *edge_contribs;
    SurfacePoint *d_points;
    DShape *d_shapes;
    float *debug_image;
};

void accumulate_secondary_edge_derivatives(const Scene &scene,
                                           const BufferView<int> &active_pixels,
                                           const BufferView<SurfacePoint> &shading_points,
                                           const BufferView<SecondaryEdgeRecord> &edge_records,
                                           const BufferView<Vector3> &edge_surface_points,
                                           const BufferView<Real> &edge_contribs,
                                           BufferView<SurfacePoint> d_points,
                                           BufferView<DShape> d_shapes,
                                           float *debug_image) {
    parallel_for(secondary_edge_derivatives_accumulator{
        scene.shapes.data,
        active_pixels.begin(),
        shading_points.begin(),
        edge_records.begin(),
        edge_surface_points.begin(),
        edge_contribs.begin(),
        d_points.begin(),
        d_shapes.begin(),
        debug_image
    }, active_pixels.size(), scene.use_gpu);
}
