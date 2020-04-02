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

// Set this to false to fallback to importance resampling if edge tree doesn't work
constexpr bool c_use_edge_tree = true;
constexpr bool c_uniform_sampling = false;
constexpr bool c_use_nee_ray = true;

namespace ltc {

const float *tabMcpu = &tabM_[0];
const float *tabMgpu = nullptr;
const float *tabM = nullptr;

}

void initialize_ltc_table(bool use_gpu) {
    ltc::tabM = use_gpu ? ltc::tabMgpu : ltc::tabMcpu;
    if (use_gpu && ltc::tabM == nullptr) {
#ifdef __CUDACC__
        checkCuda(cudaMallocManaged(&ltc::tabMgpu, sizeof(ltc::tabM_)));
        checkCuda(cudaMemcpy((void*)ltc::tabMgpu,
                             (void*)ltc::tabM_, sizeof(ltc::tabM_), cudaMemcpyHostToDevice));
        ltc::tabM = ltc::tabMgpu;
#else
        assert(false);
#endif 
    }
}

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
    auto shapes_buffer = scene.shapes.view(0, (int)shapes.size());
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
        current_num_edges += (int)num_edges;
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
        if (!c_use_edge_tree) {
            // Build a global distribution if we are not using edge tree
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
            // Build a hierarchical data structure for edge sampling
            edge_tree = std::unique_ptr<EdgeTree>(
                new EdgeTree(scene.use_gpu,
                             scene.camera,
                             shapes_buffer,
                             edges.view(0, edges.size())));
        } else {
            // Build a hierarchical data structure for edge sampling
            edge_tree = std::unique_ptr<EdgeTree>(
                new EdgeTree(scene.use_gpu,
                             scene.camera,
                             shapes_buffer,
                             edges.view(0, edges.size())));
        }
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

        if ((camera.camera_type == CameraType::Perspective ||
                camera.camera_type == CameraType::Orthographic) &&
                !camera.distortion_params.defined) {
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
            auto xi = clamp(int(edge_pt[0] * camera.width - camera.viewport_beg.x),
                            0, camera.viewport_end.x - camera.viewport_beg.x);
            auto yi = clamp(int(edge_pt[1] * camera.height - camera.viewport_beg.y),
                            0, camera.viewport_end.y - camera.viewport_beg.y);
            auto rd = channel_info.radiance_dimension;
            auto d_color = Vector3{0, 0, 0};
            if (rd != -1) {
                auto viewport_width = camera.viewport_end.x - camera.viewport_beg.x;
                d_color = Vector3{
                    d_rendered_image[nd * (yi * viewport_width + xi) + rd + 0],
                    d_rendered_image[nd * (yi * viewport_width + xi) + rd + 1],
                    d_rendered_image[nd * (yi * viewport_width + xi) + rd + 2]
                };
            }
            // The weight is the length of edge divided by the probability
            // of selecting this edge, divided by the length of gradients
            // of the edge equation w.r.t. screen coordinate.
            // For perspective projection the length of edge and gradients
            // cancel each other out.
            // For fisheye & panorama we need to compute the Jacobians
            auto upper_weight = d_color / edges_pmf[edge_id];
            auto lower_weight = -d_color / edges_pmf[edge_id];

            assert(isfinite(d_color));
            assert(isfinite(upper_weight));

            throughputs[2 * idx + 0] = upper_weight;
            throughputs[2 * idx + 1] = lower_weight;

            for (int d = 0; d < nd; d++) {
                auto viewport_width = camera.viewport_end.x - camera.viewport_beg.x;
                auto d_channel = d_rendered_image[nd * (yi * viewport_width + xi) + d];
                channel_multipliers[2 * nd * idx + d] = d_channel / edges_pmf[edge_id];
                channel_multipliers[2 * nd * idx + d + nd] = -d_channel / edges_pmf[edge_id];
            }
        } else {
            assert(camera.camera_type == CameraType::Fisheye ||
                   camera.camera_type == CameraType::Panorama ||
                   camera.distortion_params.defined);
            // Fisheye or Panorama

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

            // The 3D edge equation for the fisheye & panorama camera is:
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
            auto xi = clamp(int(edge_pt[0] * camera.width - camera.viewport_beg.x),
                            0, camera.viewport_end.x - camera.viewport_beg.x);
            auto yi = clamp(int(edge_pt[1] * camera.height - camera.viewport_beg.y),
                            0, camera.viewport_end.y - camera.viewport_beg.y);
            auto rd = channel_info.radiance_dimension;
            auto d_color = Vector3{0, 0, 0};
            if (rd != -1) {
                auto viewport_width = camera.viewport_end.x - camera.viewport_beg.x;
                d_color = Vector3{
                    d_rendered_image[nd * (yi * viewport_width + xi) + rd + 0],
                    d_rendered_image[nd * (yi * viewport_width + xi) + rd + 1],
                    d_rendered_image[nd * (yi * viewport_width + xi) + rd + 2]
                };
            }
            // The weight is the length of edge divided by the probability
            // of selecting this edge, divided by the length of gradients
            // of the edge equation w.r.t. screen coordinate.
            // For perspective projection the length of edge and gradients
            // cancel each other out.
            // For fisheye & Panorama we need to compute the Jacobians
            auto upper_weight = d_color / edges_pmf[edge_id];
            auto lower_weight = -d_color / edges_pmf[edge_id];

            // alpha(p(x, y)) = dot(p(x, y), cross(v0_dir, v1_dir))
            // p = screen_to_camera(x, y)
            auto d_edge_pt = Vector2{0, 0};
            // dalpha/dx & dalpha/dy (d alpha / d p = cross(v0_dir, v1_dir))
            d_screen_to_camera(camera, edge_pt, cross(v0_dir, v1_dir), d_edge_pt);
            auto dirac_jacobian = 1.f / sqrt(square(d_edge_pt.x) + square(d_edge_pt.y));
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
                auto viewport_width = camera.viewport_end.x - camera.viewport_beg.x;
                auto d_channel = d_rendered_image[nd * (yi * viewport_width + xi) + d];
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
        auto d_edge_pt = Vector2{0, 0};
        auto edge_pt = edge_record.edge_pt;
        if ((camera.camera_type == CameraType::Perspective ||
                camera.camera_type == CameraType::Orthographic) &&
                !camera.distortion_params.defined) {
            // Equation 8 in the paper
            d_v0_ss.x = v1_ss.y - edge_pt.y;
            d_v0_ss.y = edge_pt.x - v1_ss.x;
            d_v1_ss.x = edge_pt.y - v0_ss.y;
            d_v1_ss.y = v0_ss.x - edge_pt.x;
            d_edge_pt.x = v0_ss.y - v1_ss.y;
            d_edge_pt.y = v1_ss.x - v0_ss.x;
        } else {
            assert(camera.camera_type == CameraType::Fisheye ||
                   camera.camera_type == CameraType::Panorama ||
                   camera.distortion_params.defined);

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
            d_screen_to_camera(camera, v0_ss, cross(v1_dir, edge_dir), d_v0_ss);
            d_screen_to_camera(camera, v1_ss, cross(edge_dir, v0_dir), d_v1_ss);
            d_screen_to_camera(camera, v1_ss, cross(v0_dir, v1_dir), d_edge_pt);
        }
        d_v0_ss *= edge_contrib;
        d_v1_ss *= edge_contrib;
        d_edge_pt *= edge_contrib;

        // v0_ss, v1_ss = project(camera, v0, v1)
        auto d_v0 = Vector3{0, 0, 0};
        auto d_v1 = Vector3{0, 0, 0};
        d_project(camera, v0, v1,
            d_v0_ss.x, d_v0_ss.y,
            d_v1_ss.x, d_v1_ss.y,
            d_camera, d_v0, d_v1);
        atomic_add(&d_shapes[edge_record.edge.shape_id].vertices[3 * edge_record.edge.v0], d_v0);
        atomic_add(&d_shapes[edge_record.edge.shape_id].vertices[3 * edge_record.edge.v1], d_v1);
        if (screen_gradient_image != nullptr) {
            auto xi = clamp(int(edge_pt[0] * camera.width - camera.viewport_beg.x),
                            0, camera.viewport_end.x - camera.viewport_beg.x);
            auto yi = clamp(int(edge_pt[1] * camera.height - camera.viewport_beg.y),
                            0, camera.viewport_end.y - camera.viewport_beg.y);
            auto pixel_idx = yi * (camera.viewport_end.x - camera.viewport_beg.x) + xi;
            atomic_add(&screen_gradient_image[2 * pixel_idx + 0], d_edge_pt[0]);
            atomic_add(&screen_gradient_image[2 * pixel_idx + 1], d_edge_pt[1]);
        }
    }

    const Camera camera;
    const Shape *shapes;
    const PrimaryEdgeRecord *edge_records;
    const Real *edge_contribs;
    DShape *d_shapes;
    DCamera d_camera;
    float *screen_gradient_image;
};

void compute_primary_edge_derivatives(const Scene &scene,
                                      const BufferView<PrimaryEdgeRecord> &edge_records,
                                      const BufferView<Real> &edge_contribs,
                                      BufferView<DShape> d_shapes,
                                      DCamera d_camera,
                                      float *screen_gradient_image) {
    parallel_for(primary_edge_derivatives_computer{
        scene.camera,
        scene.shapes.data,
        edge_records.begin(),
        edge_contribs.begin(),
        d_shapes.begin(),
        d_camera,
        screen_gradient_image
    }, edge_records.size(), scene.use_gpu);
}

DEVICE
inline Matrix3x3 get_ltc_matrix(const SurfacePoint &surface_point,
                                const Vector3 &wi,
                                Real roughness,
                                const float *tabM) {
    auto cos_theta = dot(wi, surface_point.shading_frame.n);
    auto theta = acos(cos_theta);
    // search lookup table
    auto rid = clamp(int(roughness * (ltc::size - 1)), 0, ltc::size - 1);
    auto tid = clamp(int((theta / (M_PI / 2.f)) * (ltc::size - 1)), 0, ltc::size - 1);
    // TODO: linear interpolation?
    return Matrix3x3(&tabM[9 * (rid + tid * ltc::size)]);
}

struct BVHStackItemH {
    BVHNodePtr node_ptr;
    int num_samples;
    Real pmf;
};

struct BVHStackItemL {
    BVHNodePtr node_ptr;
};

struct secondary_edge_sampler {
    DEVICE inline Real min_abs_bound(Real min, Real max) {
        if (min <= 0.f && max >= 0.f) {
            return Real(0);
        }
        if (min <= 0.f && max <= 0.f) {
            return max;
        }
        assert(min >= 0.f && max >= 0.f);
        return min;
    }

    DEVICE inline Real ltc_bound(const AABB3 &bounds,
                                 const SurfacePoint &p,
                                 const Matrix3x3 &m,
                                 const Matrix3x3 &m_inv) {
        // Due to the linear invariance, the maximum remains
        // the same after applying M^{-1}
        // Therefore we transform the bounds using M^{-1},
        // find the largest possible z and smallest possible
        // x, y in terms of magnitude.
        auto dir = Vector3{0, 0, 1};
        if (!::inside(bounds, p.position)) {
            AABB3 b;
            for (int i = 0; i < 8; i++) {
                b = merge(b, m_inv * (corner(bounds, i) - p.position));
            }
            if (b.p_max.z < 0) {
                return 0;
            }

            dir.x = min_abs_bound(b.p_min.x, b.p_max.x);
            dir.y = min_abs_bound(b.p_min.y, b.p_max.y);
            dir.z = b.p_max.z;
            auto dir_len = length(dir);
            if (dir_len <= 0) {
                dir = Vector3{0, 0, 1};
            } else {
                dir = dir / dir_len;
            }
        }

        auto max_dir = normalize(m * dir);
        auto max_dir_local = m_inv * max_dir;
        if (max_dir_local.z <= 0) {
            return 0;
        }
        auto n = square(length_squared(max_dir_local));
        return max_dir_local.z / n;
    }

    DEVICE inline Real ltc_bound(const AABB6 &bounds,
            const SurfacePoint &p,
            const Matrix3x3 &m,
            const Matrix3x3 &m_inv) {
        auto p_bounds = AABB3{bounds.p_min, bounds.p_max};
        return ltc_bound(p_bounds, p, m, m_inv);
    }

    DEVICE Real importance(const BVHNode3 &node,
                           const SurfacePoint &p,
                           const Matrix3x3 &m,
                           const Matrix3x3 &m_inv) {
        // importance = BRDF * weighted length / dist
        // For BRDF we estimate the bound using linearly transformed cosine distribution
        auto brdf_term = ltc_bound(node.bounds, p, m, m_inv);
        auto center = 0.5f * (node.bounds.p_min + node.bounds.p_max);
        return brdf_term * node.weighted_total_length
            / max(distance(center, p.position), Real(1e-3));
    }

    DEVICE Real importance(const BVHNode6 &node,
                           const SurfacePoint &p,
                           const Matrix3x3 &m,
                           const Matrix3x3 &m_inv) {
        // importance = BRDF * weighted length / dist
        // Except if the sphere centered at 0.5 * (p - cam_org),
        // which has radius of 0.5 * distance(p, cam_org)
        // does not intersect the directional bounding box of node, 
        // the importance is zero (see Olson and Zhang 2006)
        auto d_bounds = AABB3{node.bounds.d_min, node.bounds.d_max};
        if (!intersect(Sphere{0.5f * (p.position - cam_org),
                    0.5f * distance(p.position, cam_org)}, d_bounds)) {
            // Not silhouette
            return 0;
        }
        auto p_bounds = AABB3{node.bounds.p_min, node.bounds.p_max};
        auto brdf_term = ltc_bound(p_bounds, p, m, m_inv);
        auto center = 0.5f * (p_bounds.p_min + p_bounds.p_max);
        return brdf_term * node.weighted_total_length
            / max(distance(center, p.position), Real(1e-3));
    }

    DEVICE Real importance(const BVHNodePtr &node_ptr,
                           const SurfacePoint &p,
                           const Matrix3x3 &m,
                           const Matrix3x3 &m_inv) {
        if (node_ptr.is_bvh_node3) {
            return importance(*node_ptr.ptr3, p, m, m_inv);
        } else {
            return importance(*node_ptr.ptr6, p, m, m_inv);
        }
    }

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
                                const Matrix3x3 &m,
                                const Matrix3x3 &m_inv) {
        const auto &edge = edges[node.edge_id];
        if (!is_silhouette(scene.shapes, p.position, edge)) {
            return 0;
        }
        auto v0 = Vector3{get_v0(scene.shapes, edge)};
        auto v1 = Vector3{get_v1(scene.shapes, edge)};
        // If degenerate, the weight is 0
        if (length_squared(v1 - v0) > 1e-10f) {
            // Transform the vertices to local coordinates
            auto v0o = m_inv * (v0 - p.position);
            auto v1o = m_inv * (v1 - p.position);
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
                return max(Il1 - Il0, Real(0));
            }
        }
        return 0;
    }

    DEVICE Real leaf_importance(const BVHNodePtr &node_ptr,
                                const SurfacePoint &p,
                                const Matrix3x3 &m,
                                const Matrix3x3 &m_inv) {
        if (node_ptr.is_bvh_node3) {
            return leaf_importance(*node_ptr.ptr3, p, m, m_inv);
        } else {
            return leaf_importance(*node_ptr.ptr6, p, m, m_inv);
        }
    }

    template <typename BVHNodeType>
    DEVICE Real leaf_importance(const BVHNodeType &node,
                                const SurfacePoint &p,
                                const Matrix3x3 &m,
                                const Matrix3x3 &m_inv,
                                const Ray &nee_ray,
                                const Intersection &nee_isect,
                                Real edge_billboard_size) {
        const auto &edge = edges[node.edge_id];
        if (!is_silhouette(scene.shapes, p.position, edge)) {
            return 0;
        }
        if (nee_isect.valid()) {
            auto nee_pt = nee_ray.org + nee_ray.tmax * nee_ray.dir;
            if (!is_silhouette(scene.shapes, nee_pt, edge)) {
                return 0;
            }
        } else {
            if (!is_silhouette(scene.shapes, nee_ray.dir, edge)) {
                return 0;
            }
        }
        // Intersect nee_ray with the edge billboard, reject if
        // the intersection is not in edge_billboard_size
        auto v0 = Vector3{get_v0(scene.shapes, edge)};
        auto v1 = Vector3{get_v1(scene.shapes, edge)};
        auto plane_pt = v0;
        auto plane_normal = nee_ray.dir;
        auto t = -(dot(nee_ray.org, plane_normal) - dot(plane_pt, plane_normal)) /
            dot(nee_ray.dir, plane_normal);
        auto isect_pt = nee_ray.org + nee_ray.dir * t;
        // Project isect_pt to corresponding point on the edge
        auto v0_p = v0 - isect_pt;
        auto v0_v1 = normalize(v1 - v0);
        auto edge_pt = isect_pt + v0_p - (dot(v0_p, v0_v1)) * v0_v1;
        if (distance_squared(edge_pt, isect_pt) > square(edge_billboard_size)) {
            return 0;
        }

        // If degenerate, the weight is 0
        if (length_squared(v1 - v0) > 1e-10f) {
            // Transform the vertices to local coordinates
            auto v0o = m_inv * (v0 - p.position);
            auto v1o = m_inv * (v1 - p.position);
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
                return max(Il1 - Il0, Real(0));
            }
        }
        return 0;
    }

    DEVICE Real leaf_importance(const BVHNodePtr &node_ptr,
                                const SurfacePoint &p,
                                const Matrix3x3 &m,
                                const Matrix3x3 &m_inv,
                                const Ray &nee_ray,
                                const Intersection &nee_isect,
                                Real edge_bounds_expand) {
        if (node_ptr.is_bvh_node3) {
            return leaf_importance(*node_ptr.ptr3,
                p, m, m_inv, nee_ray, nee_isect, edge_bounds_expand);
        } else {
            return leaf_importance(*node_ptr.ptr6,
                p, m, m_inv, nee_ray, nee_isect, edge_bounds_expand);
        }
    }

    DEVICE bool inside(const BVHNodePtr &node_ptr, const SurfacePoint &p) {
        if (node_ptr.is_bvh_node3) {
            return ::inside(node_ptr.ptr3->bounds, p.position);
        } else {
            return ::inside(node_ptr.ptr6->bounds, p.position);
        }
    }

    static constexpr auto num_h_samples = 16;

    DEVICE bool intersect_edge(const Edge &edge,
                               const Ray &nee_ray) {
        // Does nee_ray hit the edge billboard?
        auto v0 = Vector3{get_v0(scene.shapes, edge)};
        auto v1 = Vector3{get_v1(scene.shapes, edge)};
        auto plane_pt = v0;
        auto plane_normal = nee_ray.dir;
        auto t = -(dot(nee_ray.org, plane_normal) - dot(plane_pt, plane_normal)) /
            dot(nee_ray.dir, plane_normal);
        if (t < nee_ray.tmin || t > nee_ray.tmax) {
            return false;
        }
        auto isect_pt = nee_ray.org + nee_ray.dir * t;
        // Project isect_pt to corresponding point on the edge
        auto v0_p = v0 - isect_pt;
        auto v0_v1 = normalize(v1 - v0);
        auto edge_pt = isect_pt + v0_p - (dot(v0_p, v0_v1)) * v0_v1;
        return distance_squared(edge_pt, isect_pt) < square(edge_bounds_expand);
    }

    DEVICE int sample_edge_h(const EdgeTreeRoots &edge_tree_roots,
                             const SurfacePoint &p,
                             const Matrix3x3 &m,
                             const Matrix3x3 &m_inv,
                             const Ray &nee_ray,
                             Real sample,
                             Real resample_sample,
                             Real &sample_weight) {
        constexpr auto buffer_size = 128;
        BVHStackItemH buffer[buffer_size];
        auto selected_edge = -1;
        auto edge_weight = Real(0);
        auto wsum = Real(0);

        auto stack_ptr = &buffer[0];

        // randomly sample an edge using edge hierarchy
        // push both nodes into stack
        auto imp_cs = Real(0);
        auto imp_ncs = Real(0);
        if (edge_tree_roots.cs_bvh_root != nullptr) {
            imp_cs = 1;
        }
        if (edge_tree_roots.ncs_bvh_root != nullptr) {
            imp_ncs = 1;
        }
        if (imp_cs <= 0 && imp_ncs <= 0) {
            return -1;
        }
        auto prob_cs = imp_cs / (imp_cs + imp_ncs);
        auto prob_ncs = 1 - prob_cs;
        auto expected_cs = num_h_samples * prob_cs;
        auto expected_ncs = num_h_samples * prob_ncs;
        auto samples_cs = int(floor(expected_cs));
        auto samples_ncs = int(floor(expected_ncs));
        if (samples_cs + samples_ncs < num_h_samples) {
            auto prob = expected_cs - samples_cs;
            if (sample < prob) {
                samples_cs++;
                sample /= prob;
            } else {
                samples_ncs++;
                sample = (sample - prob) / (1 - prob);
            }
        }

        if (samples_cs > 0) {
            *stack_ptr++ = BVHStackItemH{
                BVHNodePtr{edge_tree_roots.cs_bvh_root}, samples_cs, prob_cs};
        }
        if (samples_ncs > 0) {
            *stack_ptr++ = BVHStackItemH{
                BVHNodePtr{edge_tree_roots.ncs_bvh_root}, samples_ncs, prob_ncs};
        }
        while (stack_ptr != &buffer[0]) {
            assert(stack_ptr > &buffer[0] && stack_ptr < &buffer[buffer_size]);
            // pop from stack
            const auto &stack_item = *--stack_ptr;
            if (is_leaf(stack_item.node_ptr)) {
                auto w = stack_item.num_samples *
                    leaf_importance(stack_item.node_ptr, p, m, m_inv) /
                    stack_item.pmf;
                if (w > 0) {
                    auto prev_wsum = wsum;
                    wsum += w;
                    auto normalized_w = w / wsum;
                    if (resample_sample <= normalized_w || prev_wsum == 0) {
                        selected_edge = get_edge_id(stack_item.node_ptr);
                        edge_weight = w * stack_item.pmf;
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
                auto imp0 = Real(0), imp1 = Real(0);
                if (inside(stack_item.node_ptr, p)) {
                    imp0 = imp1 = Real(1);
                } else {
                    imp0 = importance(children[0], p, m, m_inv);
                    imp1 = importance(children[1], p, m, m_inv);
                }
                if (imp0 > 0 || imp1 > 0) {
                    auto prob0 = imp0 / (imp0 + imp1);
                    auto prob1 = 1 - prob0;
                    auto expected0 = stack_item.num_samples * prob0;
                    auto expected1 = stack_item.num_samples * prob1;
                    auto samples0 = int(floor(expected0));
                    auto samples1 = int(floor(expected1));
                    if (samples0 + samples1 < stack_item.num_samples) {
                        auto prob = expected0 - samples0;
                        if (sample < prob) {
                            samples0++;
                            sample /= prob;
                        } else {
                            samples1++;
                            sample = (sample - prob) / (1 - prob);
                        }
                    }
                    auto current_pmf = stack_item.pmf;
                    if (samples0 > 0) {
                        *stack_ptr++ = BVHStackItemH{
                            BVHNodePtr(children[0]), samples0, current_pmf * prob0};
                    }
                    if (samples1 > 0) {
                        *stack_ptr++ = BVHStackItemH{
                            BVHNodePtr(children[1]), samples1, current_pmf * prob1};
                    }
                }
            }
        }
        if (edge_weight <= 0 || wsum <= 0) {
            return -1;
        }

        auto pmf_h = edge_weight * num_h_samples / wsum;
        sample_weight = 1 / pmf_h;
        return selected_edge;
    }

    DEVICE int sample_edge_l(const EdgeTreeRoots &edge_tree_roots,
                             const SurfacePoint &p,
                             const Matrix3x3 &m,
                             const Matrix3x3 &m_inv,
                             const Ray &nee_ray,
                             const Intersection &nee_isect,
                             const SurfacePoint &nee_point,
                             Real resample_sample,
                             Real &sample_weight,
                             Vector3 &edge_pt,
                             Vector3 &mwt) {
        constexpr auto buffer_size = 128;
        BVHStackItemL buffer[buffer_size];
        auto selected_edge = -1;
        auto edge_weight = Real(0);
        auto wsum = Real(0);

        auto stack_ptr = &buffer[0];
        // randomly sample a point on an edge by collecting 
        // all edges intersect with nee_ray
        // push both nodes into stack
        if (edge_tree_roots.cs_bvh_root != nullptr) {
            *stack_ptr++ = BVHStackItemL{
                BVHNodePtr{edge_tree_roots.cs_bvh_root}};
        }
        if (edge_tree_roots.ncs_bvh_root != nullptr) {
            *stack_ptr++ = BVHStackItemL{
                BVHNodePtr{edge_tree_roots.ncs_bvh_root}};
        }
        while (stack_ptr != &buffer[0]) {
            assert(stack_ptr > &buffer[0] && stack_ptr < &buffer[buffer_size]);
            // pop from stack
            const auto &stack_item = *--stack_ptr;
            if (is_leaf(stack_item.node_ptr)) {
                auto w = leaf_importance(stack_item.node_ptr, p, m, m_inv,
                    nee_ray, nee_isect, edge_bounds_expand);
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
                if (nee_isect.valid()) {
                    auto nee_pt = nee_point.position;
                    if (contains_silhouette(children[0], p.position) &&
                            contains_silhouette(children[0], nee_pt) &&
                            intersect(children[0], nee_ray, edge_bounds_expand)) {
                        *stack_ptr++ = BVHStackItemL{BVHNodePtr(children[0])};
                    }
                    if (contains_silhouette(children[1], p.position) &&
                            contains_silhouette(children[1], nee_pt) &&
                            intersect(children[1], nee_ray, edge_bounds_expand)) {
                        *stack_ptr++ = BVHStackItemL{BVHNodePtr(children[1])};
                    }
                } else {
                    // Infinitely far nee rays
                    // TODO: silhouette detection for infinitely far positions
                    if (contains_silhouette(children[0], p.position) &&
                            intersect(children[0], nee_ray, edge_bounds_expand)) {
                        *stack_ptr++ = BVHStackItemL{BVHNodePtr(children[0])};
                    }
                    if (contains_silhouette(children[1], p.position) &&
                            intersect(children[1], nee_ray, edge_bounds_expand)) {
                        *stack_ptr++ = BVHStackItemL{BVHNodePtr(children[1])};
                    }
                }
            }
        }
        if (selected_edge == -1) {
            return -1;
        }

        auto pmf = edge_weight / wsum;
        // Intersect nee_ray with the edge billboard
        const auto &edge = edges[selected_edge];
        auto v0 = Vector3{get_v0(scene.shapes, edge)};
        auto v1 = Vector3{get_v1(scene.shapes, edge)};
        auto plane_pt = v0;
        auto plane_normal = nee_ray.dir;
        auto t = -(dot(nee_ray.org, plane_normal) - dot(plane_pt, plane_normal)) /
            dot(nee_ray.dir, plane_normal);
        if (t < nee_ray.tmin || t > nee_ray.tmax) {
            return -1;
        }
        auto isect_pt = nee_ray.org + nee_ray.dir * t;
        auto isect_jac = Real(0);
        auto pdf_nee = Real(0);
        if (nee_isect.valid()) {
            // Area light
            auto nee_normal = nee_point.geom_normal;
            auto nee_pt = nee_point.position;
            auto tau = dot(nee_pt - nee_ray.org, nee_normal) / dot(isect_pt - nee_ray.org, nee_normal);
            auto omega = isect_pt - nee_ray.org;
            isect_jac = length(tau * ((v1 - v0) -
                        omega * (dot(v1 - v0, nee_normal) / dot(omega, nee_normal))));
            const auto &light_shape = scene.shapes[nee_isect.shape_id];
            auto light_pmf = scene.light_pmf[light_shape.light_id];
            auto light_area = scene.light_areas[light_shape.light_id];
            pdf_nee = light_pmf / light_area;
        } else {
            // Environment map sampling
            isect_jac = 1 / distance_squared(isect_pt, nee_ray.org);
            pdf_nee = envmap_pdf(*scene.envmap, nee_ray.dir);
        }
        if (pmf <= 0 || isect_jac <= 0 || pdf_nee <= 0) {
            return -1;
        }
        sample_weight = 1 / (2 * edge_bounds_expand * pmf * isect_jac * pdf_nee);
        // Project isect_pt to corresponding point on the edge
        auto v0_p = v0 - isect_pt;
        auto v0_v1 = normalize(v1 - v0);
        edge_pt = isect_pt + v0_p - (dot(v0_p, v0_v1)) * v0_v1 - nee_ray.org;
        mwt = v1 - v0;
        return selected_edge;
    }
    
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        const auto &edge_sample = edge_samples[idx];
        const auto &wi = -incoming_rays[pixel_id].dir;
        const auto &shading_isect = shading_isects[pixel_id];
        const auto &shading_point = shading_points[pixel_id];
        const auto &throughput = throughputs[pixel_id];
        const auto &min_rough = min_roughness[pixel_id];
        const auto &nee_isect = nee_isects[pixel_id];
        const auto &nee_point = nee_points[pixel_id];

        auto nee_ray = nee_rays[pixel_id];
        // nee_ray.tmax is used for marking occluded rays, so we need to recompute
        // it here
        // TODO: there is probably a more elegant solution
        if (nee_isect.valid()) {
            nee_ray.tmax = length(nee_point.position - nee_ray.org);
        } else {
            nee_ray.tmax = infinity<Real>();
        }

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
        const Shape &shape = scene.shapes[shading_isect.shape_id];
        const Material &material = scene.materials[shape.material_id];
        // First decide which component of BRDF to sample
        auto diffuse_reflectance = get_diffuse_reflectance(material, shading_point);
        auto specular_reflectance = get_specular_reflectance(material, shading_point);
        auto diffuse_weight = luminance(diffuse_reflectance);
        auto specular_weight = luminance(specular_reflectance);
        if (c_uniform_sampling) {
            diffuse_weight = 1;
            specular_weight = 0;
        }
        auto weight_sum = diffuse_weight + specular_weight;
        if (weight_sum <= 0.f) {
            // black material
            return;
        }
        auto diffuse_pmf = diffuse_weight / weight_sum;
        auto specular_pmf = specular_weight / weight_sum;
        auto m_pmf = Real(0);
        auto n = shading_point.shading_frame.n;
        if (material.two_sided) {
            if (dot(wi, n) < 0.f) {
                n = -n;
            }
        }
        auto frame_x = normalize(wi - n * dot(wi, n));
        auto frame_y = cross(n, frame_x);
        if (dot(wi, n) > 1 - 1e-6f) {
            coordinate_system(n, frame_x, frame_y);
        }
        auto isotropic_frame = Frame{frame_x, frame_y, n};
        auto m = Matrix3x3{};
        auto m_inv = Matrix3x3{};
        auto roughness = max(get_roughness(material, shading_point), min_rough);
        if (edge_sample.bsdf_component <= diffuse_pmf) {
            // M is shading frame * identity
            m_inv = Matrix3x3(isotropic_frame);
            m = inverse(m_inv);
            m_pmf = diffuse_pmf;
        } else {
            m_inv = inverse(get_ltc_matrix(shading_point, wi, roughness, tabM)) *
                    Matrix3x3(isotropic_frame);
            m = inverse(m_inv);
            m_pmf = specular_pmf;
        }

        auto edge_id = -1;
        auto edge_weight = Real(0);
        auto sample_p = Vector3{};
        auto mwt = Vector3{};

        auto edge_sel = edge_sample.edge_sel;
        auto use_nee_ray = false;
        auto nee_ray_pmf = Real(1);
        auto is_diffuse_or_glossy =
            edge_sample.bsdf_component <= diffuse_pmf || roughness > Real(0.1);
        // Turn off nee edge sampling when roughness is low
        if (c_use_nee_ray && is_diffuse_or_glossy) {
            use_nee_ray = edge_sel < Real(0.5) ? true : false;
            if (roughness > Real(0.1)) {
                nee_ray_pmf = 0.5f;
            } else {
                nee_ray_pmf = use_nee_ray ? diffuse_pmf * 0.5f : 1.f - diffuse_pmf * 0.5f;
            }
        }
        if (!use_nee_ray) {
            if (c_use_nee_ray && is_diffuse_or_glossy) {
                edge_sel = (edge_sel - 0.5) * 2;
            }
            if (edges_pmf != nullptr) {
                if (c_uniform_sampling) {
                    const Real *edge_ptr = thrust::upper_bound(thrust::seq,
                            edges_cdf, edges_cdf + num_edges, edge_sel);
                    edge_id = clamp((int)(edge_ptr - edges_cdf - 1), 0, num_edges - 1);
                    edge_weight = 1 / edges_pmf[edge_id];
                } else {
                    // Sample an edge by importance resampling:
                    // We randomly sample M edges, estimate contribution based on LTC, 
                    // then sample based on the estimated contribution.
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
                                modulo(edge_sel + Real(sample_id) / M, Real(1)));
                        auto edge_id = clamp((int)(edge_ptr - edges_cdf - 1), 0, num_edges - 1);
                        edge_ids[sample_id] = edge_id;
                        edge_weights[sample_id] = 0;
                        const auto &edge = edges[edge_id];
                        // If the edge lies on the same triangle of shading isects, the weight is 0
                        // If not a silhouette edge, the weight is 0
                        bool same_tri = edge.shape_id == shading_isect.shape_id &&
                            (edge.v0 == shading_isect.tri_id || edge.v1 == shading_isect.tri_id);
                        if (edges_pmf[edge_id] > 0 &&
                                is_silhouette(scene.shapes, shading_point.position, edge) &&
                                !same_tri) {
                            auto v0 = Vector3{get_v0(scene.shapes, edge)};
                            auto v1 = Vector3{get_v1(scene.shapes, edge)};
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
                    edge_weight = (resample_cdf[M - 1] / M) /
                        (edge_weights[resample_id] * edges_pmf[edge_ids[resample_id]]);
                    edge_id = edge_ids[resample_id];
                }
            } else {
                // sample using a tree traversal
                edge_id = sample_edge_h(edge_tree_roots,
                    shading_point, m, m_inv, nee_ray,
                    edge_sel, edge_sample.resample_sel, edge_weight);
                if (edge_id == -1 || edge_weight <= 0) {
                    return;
                }
            }

            const auto &edge = edges[edge_id];
            if (!is_silhouette(scene.shapes, shading_point.position, edge)) {
                return;
            }

            auto v0 = Vector3{get_v0(scene.shapes, edge)};
            auto v1 = Vector3{get_v1(scene.shapes, edge)};
            // Transform the vertices to local coordinates
            auto v0o = m_inv * (v0 - shading_point.position);
            auto v1o = m_inv * (v1 - shading_point.position);
            if (v0o[2] <= 0.f && v1o[2] <= 0.f) {
                // Edge is below the shading point
                return;
            }

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
                swap_(lb, ub);
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
            sample_p = m * (vo + l * wt);
            auto edge_pdf = m_pmf * line_pdf(l);
            assert(edge_pdf > 0);
            edge_weight /= (m_pmf * line_pdf(l));
            mwt = m * wt;
        } else {
            // edge_sel *= 2;
            edge_id = sample_edge_l(edge_tree_roots,
                 shading_point, m, m_inv, nee_ray, nee_isect, nee_point,
                 edge_sample.resample_sel, edge_weight, sample_p,
                 mwt);
            if (edge_id == -1 || edge_weight <= 0) {
                return;
            }
        }

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
        auto d_color = Vector3{0, 0, 0};
        if (rd != -1) {
            d_color = Vector3{
                d_rendered_image[nd * pixel_id + rd + 0],
                d_rendered_image[nd * pixel_id + rd + 1],
                d_rendered_image[nd * pixel_id + rd + 2]
            };
        }
        edge_records[idx].edge = edge;
        edge_records[idx].edge_pt = sample_p; // for Jacobian computation 
        edge_records[idx].mwt = mwt; // for Jacobian computation
        edge_records[idx].use_nee_ray = use_nee_ray;
        edge_records[idx].is_diffuse_or_glossy = is_diffuse_or_glossy;
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
        assert(nee_ray_pmf > 0);
        auto nt = throughput * eval_bsdf * d_color * edge_weight / nee_ray_pmf;
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
    const Real *edges_pmf;
    const Real *edges_cdf;
    const EdgeTreeRoots edge_tree_roots;
    const Real edge_bounds_expand;
    const int *active_pixels;
    const SecondaryEdgeSample *edge_samples;
    const Ray *incoming_rays;
    const RayDifferential *incoming_ray_differentials;
    const Intersection *shading_isects;
    const SurfacePoint *shading_points;
    const Ray *nee_rays;
    const Intersection *nee_isects;
    const SurfacePoint *nee_points;
    const Vector3 *throughputs;
    const Real *min_roughness;
    const float *d_rendered_image;
    const ChannelInfo channel_info;
    const float *tabM;
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
                            const BufferView<Ray> &nee_rays,
                            const BufferView<Intersection> &nee_isects,
                            const BufferView<SurfacePoint> &nee_points,
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
        scene.edge_sampler.secondary_edges_pmf.begin(),
        scene.edge_sampler.secondary_edges_cdf.begin(),
        get_edge_tree_roots(edge_tree),
        edge_tree != nullptr ? edge_tree->edge_bounds_expand : Real(0),
        active_pixels.begin(),
        samples.begin(),
        incoming_rays.begin(),
        incoming_ray_differentials.begin(),
        shading_isects.begin(),
        shading_points.begin(),
        nee_rays.begin(),
        nee_isects.begin(),
        nee_points.begin(),
        throughputs.begin(),
        min_roughness.begin(),
        d_rendered_image,
        channel_info,
        ltc::tabM,
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

        if (c_use_nee_ray) {
            auto light_id0 = -1, light_id1 = -1;
            if (edge_isect0.valid()) {
                const auto &shape = scene.shapes[edge_isect0.shape_id];
                light_id0 = shape.light_id;
            }
            if (edge_isect1.valid()) {
                const auto &shape = scene.shapes[edge_isect1.shape_id];
                light_id1 = shape.light_id;
            }
            
            auto hit_light = light_id0 != -1 || light_id1 != -1;
            if (!hit_light && scene.envmap != nullptr) {
                auto wo = normalize(edge_record.edge_pt - shading_point.position);
                hit_light = envmap_pdf(*scene.envmap, wo) > 0;
            }

            if (edge_record.use_nee_ray) {
                if (hit_light) {
                    edge_throughputs[2 * idx + 0] *= 0.5f;
                    edge_throughputs[2 * idx + 1] *= 0.5f;
                } else {
                    edge_throughputs[2 * idx + 0] = Vector3{0, 0, 0};
                    edge_throughputs[2 * idx + 1] = Vector3{0, 0, 0};
                }
            } else {
                if (hit_light && edge_record.is_diffuse_or_glossy) {
                    edge_throughputs[2 * idx + 0] *= 0.5f;
                    edge_throughputs[2 * idx + 1] *= 0.5f;
                }
            }
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
        //assert(isfinite(edge_contrib0));
        //assert(isfinite(edge_contrib1));
        assert(isfinite(dcolor_dp));

        d_points[pixel_id].position += dcolor_dp;
        atomic_add(&(d_shapes[edge_record.edge.shape_id].vertices[3 * edge_record.edge.v0]), dcolor_dv0);
        atomic_add(&(d_shapes[edge_record.edge.shape_id].vertices[3 * edge_record.edge.v1]), dcolor_dv1);
    }

    const Shape *shapes;
    const int *active_pixels;
    const SurfacePoint *shading_points;
    const SecondaryEdgeRecord *edge_records;
    const Vector3 *edge_surface_points;
    const Real *edge_contribs;
    SurfacePoint *d_points;
    DShape *d_shapes;
};

void accumulate_secondary_edge_derivatives(const Scene &scene,
                                           const BufferView<int> &active_pixels,
                                           const BufferView<SurfacePoint> &shading_points,
                                           const BufferView<SecondaryEdgeRecord> &edge_records,
                                           const BufferView<Vector3> &edge_surface_points,
                                           const BufferView<Real> &edge_contribs,
                                           BufferView<SurfacePoint> d_points,
                                           BufferView<DShape> d_shapes) {
    parallel_for(secondary_edge_derivatives_accumulator{
        scene.shapes.data,
        active_pixels.begin(),
        shading_points.begin(),
        edge_records.begin(),
        edge_surface_points.begin(),
        edge_contribs.begin(),
        d_points.begin(),
        d_shapes.begin()
    }, active_pixels.size(), scene.use_gpu);
}
