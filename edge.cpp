#include "edge.h"
#include "line_clip.h"
#include "scene.h"
#include "parallel.h"
#include "thrust_utils.h"
#include "ltc.inc"

#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>

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
        // We use the length * cos(dihedral angle) to sample the edges
        // If the dihedral angle is large, it's less likely that the edge would be an silhouette        
        auto &secondary_edge_weight = secondary_edge_weights[idx];
        auto cos_dihedral = Real(1);
        if (edge.f1 != -1) {
            auto n0 = get_n0(shapes, edge);
            auto n1 = get_n1(shapes, edge);
            cos_dihedral = fabs(dot(n0, n1));
        }
        auto v0 = get_v0(shapes, edge);
        auto v1 = get_v1(shapes, edge);
        secondary_edge_weight = distance(v0, v1) * cos_dihedral;
    }

    const Shape *shapes;
    const Edge *edges;
    Real *secondary_edge_weights;
};

EdgeSampler::EdgeSampler(const std::vector<const Shape*> &shapes,
                         const Scene &scene) {
    auto shapes_buffer = scene.shapes.view(0, shapes.size());
    // Conservatively allocate a big buffer for all edges
    auto num_total_triangles = 0;
    for (int shape_id = 0; shape_id < (int)shapes.size(); shape_id++) {
        num_total_triangles += shapes[shape_id]->num_triangles;
    }
    // Collect the edges
    // TODO: this assumes each edge is only associated with two triangles
    //       which may be untrue for some pathological meshes
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
        DISPATCH(scene.use_gpu, thrust::copy, edges_buffer_begin, new_end, edges_begin);
        current_num_edges += num_edges;
    }
    edges.count = current_num_edges;
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

    // Secondary edge sampler:
    secondary_edges_pmf = Buffer<Real>(scene.use_gpu, edges.count);
    secondary_edges_cdf = Buffer<Real>(scene.use_gpu, edges.count);
    // For each edge we compute the length and store the length in 
    // secondary_edges_pmf
    parallel_for(secondary_edge_weighter{
        scene.shapes.data,
        edges.begin(),
        secondary_edges_pmf.begin()
    }, edges.size(), scene.use_gpu);
    {
        // Compute PMF & CDF
        // First normalize secondary_edges_pmf.
        auto total_length = DISPATCH(scene.use_gpu, thrust::reduce,
            secondary_edges_pmf.begin(),
            secondary_edges_pmf.end(),
            Real(0),
            thrust::plus<Real>());
        DISPATCH(scene.use_gpu, thrust::transform,
            secondary_edges_pmf.begin(),
            secondary_edges_pmf.end(),
            thrust::make_constant_iterator(total_length),
            secondary_edges_pmf.begin(),
            thrust::divides<Real>());
        // Next we compute a prefix sum
        DISPATCH(scene.use_gpu, thrust::transform_exclusive_scan,
            secondary_edges_pmf.begin(),
            secondary_edges_pmf.end(),
            secondary_edges_cdf.begin(),
            thrust::identity<Real>(), Real(0), thrust::plus<Real>());
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

        if (!camera.fisheye) {
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

        auto &d_v0 = d_vertices[2 * idx + 0];
        auto &d_v1 = d_vertices[2 * idx + 1];
        auto &d_camera = d_cameras[idx];
        // Initialize derivatives
        d_v0 = DVertex{};
        d_v1 = DVertex{};
        d_camera = DCameraInst{};
        if (edge_record.edge.shape_id < 0) {
            return;
        }
        d_v0.shape_id = edge_record.edge.shape_id;
        d_v1.shape_id = edge_record.edge.shape_id;
        d_v0.vertex_id = edge_record.edge.v0;
        d_v1.vertex_id = edge_record.edge.v1;

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
        if (!camera.fisheye) {
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
        d_project(camera, v0, v1,
            d_v0_ss.x, d_v0_ss.y,
            d_v1_ss.x, d_v1_ss.y,
            d_camera,
            d_v0.d_v, d_v1.d_v);
    }

    const Camera camera;
    const Shape *shapes;
    const PrimaryEdgeRecord *edge_records;
    const Real *edge_contribs;
    DVertex *d_vertices;
    DCameraInst *d_cameras;
};

void compute_primary_edge_derivatives(const Scene &scene,
                                      const BufferView<PrimaryEdgeRecord> &edge_records,
                                      const BufferView<Real> &edge_contribs,
                                      BufferView<DVertex> d_vertices,
                                      BufferView<DCameraInst> d_cameras) {
    parallel_for(primary_edge_derivatives_computer{
        scene.camera,
        scene.shapes.data,
        edge_records.begin(),
        edge_contribs.begin(),
        d_vertices.begin(), d_cameras.begin()
    }, edge_records.size(), scene.use_gpu);
}

DEVICE
inline Matrix3x3 get_ltc_matrix(const Material &material,
                                const SurfacePoint &surface_point,
                                const Vector3 &wi,
                                Real min_rough) {
    auto roughness = max(get_roughness(material, surface_point), min_rough);
    auto cos_theta = dot(wi, surface_point.shading_frame.n);
    auto theta = acos(cos_theta);
    // search lookup table
    auto rid = clamp(int(roughness * (ltc::size - 1)), 0, ltc::size - 1);
    auto tid = clamp(int((theta / (M_PI / 2.f)) * (ltc::size - 1)), 0, ltc::size - 1);
    // TODO: linear interpolation?
    return Matrix3x3(ltc::tabM[rid+tid*ltc::size]);
}

struct secondary_edge_sampler {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &edge_sample = edge_samples[idx];
        const auto &wi = -incoming_rays[pixel_id].dir;
        const auto &shading_isect = shading_isects[pixel_id];
        const auto &shading_point = shading_points[pixel_id];
        const auto &throughput = throughputs[pixel_id];
        const auto &min_rough = min_roughness[pixel_id];

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

        // Setup the Linearly Transformed Cosine Distribution
        const Shape &shape = shapes[shading_isect.shape_id];
        const Material &material = materials[shape.material_id];
        // First decide which component of BRDF to sample
        auto diffuse_reflectance = get_diffuse_reflectance(material, shading_point);
        auto specular_reflectance = get_specular_reflectance(material, shading_point);
        auto diffuse_weight = luminance(diffuse_reflectance);
        auto specular_weight = luminance(specular_reflectance);
        auto weight_sum = diffuse_weight + specular_weight;
        if (weight_sum <= 0.f) {
            // black material
            return;
        }
        auto diffuse_pmf = diffuse_weight / weight_sum;
        auto specular_pmf = specular_weight / weight_sum;
        auto m_pmf = 0.f;
        auto n = shading_point.shading_frame.n;
        auto frame_x = normalize(wi - n * dot(wi, n));
        auto frame_y = cross(n, frame_x);
        auto isotropic_frame = Frame{frame_x, frame_y, n};
        auto m = Matrix3x3{};
        auto m_inv = Matrix3x3{};
        if (edge_sample.bsdf_component <= diffuse_pmf) {
            // M is shading frame * identity
            m_inv = Matrix3x3(isotropic_frame);
            m = inverse(m_inv);
            m_pmf = diffuse_pmf;
        } else {
            m_inv = inverse(get_ltc_matrix(material, shading_point, wi, min_rough)) *
                    Matrix3x3(isotropic_frame);
            m = inverse(m_inv);
            m_pmf = specular_pmf;
        }

        // Sample an edge by importance resampling:
        // We randomly sample M edges, estimate contribution based on LTC, 
        // then sample based on the estimated contribution.
        // TODO: a properer strategy is to traverse a tree to fill up the M slots
        constexpr int M = 64;
        int edge_ids[M];
        Real edge_weights[M];
        Real resample_cdf[M];
        for (int sample_id = 0; sample_id < M; sample_id++) {
            // Sample an edge by binary search on cdf
            // We use some form of stratification over the M samples here: 
            // the random number we use is mod(edge_sample.edge_sel + i / M, 1)
            // It enables us to choose M edges with a single random number
            const Real *edge_ptr = thrust::upper_bound(thrust::seq,
                edges_cdf, edges_cdf + num_edges,
                modulo(edge_sample.edge_sel + Real(sample_id) / M, Real(1)));
            auto edge_id = clamp((int)(edge_ptr - edges_cdf - 1), 0, num_edges - 1);
            edge_ids[sample_id] = edge_id;
            edge_weights[sample_id] = 0;
            const auto &edge = edges[edge_id];
            // If the edge lies on the same triangle of shading isects, the weight is 0
            // If not a silhouette edge, the weight is 0
            bool same_tri = edge.shape_id == shading_isect.shape_id &&
                (edge.v0 == shading_isect.tri_id || edge.v1 == shading_isect.tri_id);
            if (edges_pmf[edge_id] > 0 &&
                    is_silhouette(shapes, shading_point.position, edge) &&
                    !same_tri) {
                auto v0 = Vector3{get_v0(shapes, edge)};
                auto v1 = Vector3{get_v1(shapes, edge)};
                // If degenerate, the weight is 0
                if (length_squared(v1 - v0) > 1e-10f) {
                    // Transform the vertices to local coordinates
                    auto v0o = m_inv * (v0 - shading_point.position);
                    auto v1o = m_inv * (v1 - shading_point.position);
                    // If below surface, the weight is 0
                    if (v0o[2] > 0.f || v1o[2] > 0.f) {
                        // Clip to the surface tangent plane
                        if (v0o[2] < 0.f) {
                            v0o = (v0o*v1o[2] - v1o*v0o[2]) / (v1o[2] - v0o[2]);
                        }
                        if (v1o[2] < 0.f) {
                            v1o = (v0o*v1o[2] - v1o*v0o[2]) / (v1o[2] - v0o[2]);
                        }
                        // Integrate over the edge using LTC
                        auto vodir = v1o - v0o;
                        auto wt = normalize(vodir);
                        auto l0 = dot(v0o, wt);
                        auto l1 = dot(v1o, wt);
                        auto vo = v0o - l0 * wt;
                        auto d = length(vo);
                        auto I = [&](Real l) {
                            return (l/(d*(d*d+l*l))+atan(l/d)/(d*d))*vo[2] +
                                   (l*l/(d*(d*d+l*l)))*wt[2];
                        };
                        auto Il0 = I(l0);
                        auto Il1 = I(l1);
                        edge_weights[sample_id] = max((Il1 - Il0) / edges_pmf[edge_id], Real(0));
                    }
                }
            }
            if (sample_id == 0) {
                resample_cdf[sample_id] = edge_weights[sample_id];
            } else { // sample_id > 0
                resample_cdf[sample_id] = resample_cdf[sample_id - 1] + edge_weights[sample_id];
            }
        }
        if (resample_cdf[M - 1] <= 0) {
            return;
        }
        // Use resample_cdf to pick one edge
        auto resample_u = edge_sample.resample_sel * resample_cdf[M - 1];
        auto resample_id = -1;
        for (int sample_id = 0; sample_id < M; sample_id++) {
            if (resample_u <= resample_cdf[sample_id]) {
                resample_id = sample_id;
                break;
            }
        }
        if (edge_weights[resample_id] <= 0 || resample_id == -1) {
            // Just in case if there's some numerical error
            return;
        }
        auto resample_weight = (resample_cdf[M - 1] / M) /
            (edge_weights[resample_id] * edges_pmf[edge_ids[resample_id]]);
        const auto &edge = edges[edge_ids[resample_id]];

        auto v0 = Vector3{get_v0(shapes, edge)};
        auto v1 = Vector3{get_v1(shapes, edge)};

        // Transform the vertices to local coordinates
        auto v0o = m_inv * (v0 - shading_point.position);
        auto v1o = m_inv * (v1 - shading_point.position);
        // Clip to the surface tangent plane
        if (v0o[2] < 0.f) {
            v0o = (v0o*v1o[2] - v1o*v0o[2]) / (v1o[2] - v0o[2]);
        }
        if (v1o[2] < 0.f) {
            v1o = (v0o*v1o[2] - v1o*v0o[2]) / (v1o[2] - v0o[2]);
        }
        auto vodir = v1o - v0o;
        auto wt = normalize(vodir);
        auto l0 = dot(v0o, wt);
        auto l1 = dot(v1o, wt);
        auto vo = v0o - l0 * wt;
        auto d = length(vo);
        auto I = [&](Real l) {
            return (l/(d*(d*d+l*l))+atan(l/d)/(d*d))*vo[2] +
                   (l*l/(d*(d*d+l*l)))*wt[2];
        };
        auto Il0 = I(l0);
        auto Il1 = I(l1);
        auto normalization = Il1 - Il0;
        auto line_cdf = [&](Real l) {
            return (I(l)-Il0)/normalization;
        };
        auto line_pdf = [&](Real l) {
            auto dist_sq=d*d+l*l;
            return 2.f*d*(vo+l*wt)[2]/(normalization*dist_sq*dist_sq);
        };
        // Hybrid bisection & Newton iteration
        // Here we are trying to find a point l s.t. line_cdf(l) = edge_sample.t
        auto lb = l0;
        auto ub = l1;
        if (lb > ub) {
            swap(lb, ub);
        }
        auto l = 0.5f * (lb + ub);
        for (int it = 0; it < 20; it++) {
            if (!(l >= lb && l <= ub)) {
                l = 0.5f * (lb + ub);
            }
            auto value = line_cdf(l) - edge_sample.t;
            if (fabs(value) < 1e-5f || it == 19) {
                break;
            }
            // The derivative may not be entirely accurate,
            // but the bisection is going to handle this
            if (value > 0.f) {
                ub = l;
            } else {
                lb = l;
            }
            auto derivative = line_pdf(l);
            l -= value / derivative;
        }
        if (line_pdf(l) <= 0.f) {
            // Numerical issue
            return;
        }
        // Convert from l to position
        auto sample_p = m * (vo + l * wt);

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
        edge_records[idx].mwt = m * wt; // for Jacobian computation
        rays[2 * idx + 0] = Ray(shading_point.position, v_upper_dir, 1e-3f * length(sample_p));
        rays[2 * idx + 1] = Ray(shading_point.position, v_lower_dir, 1e-3f * length(sample_p));
        const auto &incoming_ray_differential = incoming_ray_differentials[pixel_id];
        // Propagate ray differentials
        auto bsdf_ray_differential = RayDifferential{};
        bsdf_ray_differential.org_dx = incoming_ray_differential.org_dx;
        bsdf_ray_differential.org_dy = incoming_ray_differential.org_dy;
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
        auto edge_weight = resample_weight / (m_pmf * line_pdf(l));
        auto nt = throughput * eval_bsdf * d_color * edge_weight;
        // assert(isfinite(throughput));
        // assert(isfinite(eval_bsdf));
        // assert(isfinite(d_color));
        assert(isfinite(edge_weight));
        new_throughputs[2 * idx + 0] = nt;
        new_throughputs[2 * idx + 1] = -nt;
    }

    const Shape *shapes;
    const Material *materials;
    const Edge *edges;
    int num_edges;
    const Real *edges_pmf;
    const Real *edges_cdf;
    const int *active_pixels;
    const SecondaryEdgeSample *edge_samples;
    const Ray *incoming_rays;
    const RayDifferential *incoming_ray_differentials;
    const Intersection *shading_isects;
    const SurfacePoint *shading_points;
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
                            const BufferView<Vector3> &throughputs,
                            const BufferView<Real> &min_roughness,
                            const float *d_rendered_image,
                            const ChannelInfo &channel_info,
                            BufferView<SecondaryEdgeRecord> edge_records,
                            BufferView<Ray> rays,
                            BufferView<RayDifferential> &bsdf_differentials,
                            BufferView<Vector3> new_throughputs,
                            BufferView<Real> edge_min_roughness) {
    parallel_for(secondary_edge_sampler{
        scene.shapes.data,
        scene.materials.data,
        scene.edge_sampler.edges.begin(),
        (int)scene.edge_sampler.edges.size(),
        scene.edge_sampler.secondary_edges_pmf.begin(),
        scene.edge_sampler.secondary_edges_cdf.begin(),
        active_pixels.begin(),
        samples.begin(),
        incoming_rays.begin(),
        incoming_ray_differentials.begin(),
        shading_isects.begin(),
        shading_points.begin(),
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


            // Intersection Jacobian Jm(t) (Eq. 18 in the paper)
            auto isect_jacobian = intersect_jacobian(shading_point.position,
                                                     edge_record.edge_pt,
                                                     edge_surface_point.position,
                                                     edge_surface_point.geom_normal,
                                                     edge_record.mwt);
            // area of projection
            auto v0 = Vector3{get_v0(scene.shapes, edge_record.edge)};
            auto v1 = Vector3{get_v1(scene.shapes, edge_record.edge)};
            auto half_plane_normal = normalize(cross(v0 - shading_point.position,
                                                     v1 - shading_point.position));
            // ||Jm(t)|| / ||n_m x n_h|| in Eq. 15 in the paper
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
        auto pixel_id = active_pixels[idx];
        const auto &shading_point = shading_points[pixel_id];
        const auto &edge_isect0 = edge_isects[2 * idx + 0];
        const auto &edge_surface_point0 = edge_surface_points[2 * idx + 0];
        const auto &edge_isect1 = edge_isects[2 * idx + 1];
        const auto &edge_surface_point1 = edge_surface_points[2 * idx + 1];
        const auto &edge_record = edge_records[idx];
        if (edge_record.edge.shape_id < 0) {
            return;
        }

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
        d_vertices[2 * idx + 0] = DVertex{};
        d_vertices[2 * idx + 1] = DVertex{};
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
        d_vertices[2 * idx + 0].shape_id = edge_record.edge.shape_id;
        d_vertices[2 * idx + 0].vertex_id = edge_record.edge.v0;
        d_vertices[2 * idx + 0].d_v = dcolor_dv0;
        d_vertices[2 * idx + 1].shape_id = edge_record.edge.shape_id;
        d_vertices[2 * idx + 1].vertex_id = edge_record.edge.v1;
        d_vertices[2 * idx + 1].d_v = dcolor_dv1;
    }

    const Shape *shapes;
    const int *active_pixels;
    const SurfacePoint *shading_points;
    const SecondaryEdgeRecord *edge_records;
    const Vector3 *edge_surface_points;
    const Real *edge_contribs;
    SurfacePoint *d_points;
    DVertex *d_vertices;
};

void accumulate_secondary_edge_derivatives(const Scene &scene,
                                           const BufferView<int> &active_pixels,
                                           const BufferView<SurfacePoint> &shading_points,
                                           const BufferView<SecondaryEdgeRecord> &edge_records,
                                           const BufferView<Vector3> &edge_surface_points,
                                           const BufferView<Real> &edge_contribs,
                                           BufferView<SurfacePoint> d_points,
                                           BufferView<DVertex> d_vertices) {
    parallel_for(secondary_edge_derivatives_accumulator{
        scene.shapes.data,
        active_pixels.begin(),
        shading_points.begin(),
        edge_records.begin(),
        edge_surface_points.begin(),
        edge_contribs.begin(),
        d_points.begin(),
        d_vertices.begin()
    }, active_pixels.size(), scene.use_gpu);
}
