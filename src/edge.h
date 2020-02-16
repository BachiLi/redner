#pragma once

#include "redner.h"
#include "shape.h"
#include "camera.h"
#include "channels.h"
#include "edge_tree.h"

#include <memory>

struct Scene;

struct Edge {
   DEVICE Edge() : shape_id(-1), v0(0), v1(0), f0(0), f1(0) {}
   DEVICE Edge(int si, int v0, int v1, int f0, int f1) :
      shape_id(si),
      v0(v0),
      v1(v1),
      f0(f0),
      f1(f1) {}

    int shape_id;
    int v0, v1;
    int f0, f1;

    DEVICE inline bool operator==(const Edge &other) const {
        return shape_id == other.shape_id &&
            v0 == other.v0 && v1 == other.v1 &&
            f0 == other.f0 && f1 == other.f1;
    }
};

struct PrimaryEdgeRecord {
    Edge edge;
    Vector2 edge_pt;
};

struct SecondaryEdgeRecord {
    Edge edge;
    Vector3 edge_pt;
    Vector3 mwt;
    bool use_nee_ray;
    bool is_diffuse_or_glossy;
};

template <typename T>
struct TPrimaryEdgeSample {
    T edge_sel;
    T t;
};

template <typename T>
struct TSecondaryEdgeSample {
    T edge_sel;
    T resample_sel;
    T bsdf_component;
    T t;
};

struct EdgeSampler {
    EdgeSampler() {}
    EdgeSampler(const std::vector<const Shape*> &shapes,
                const Scene &scene);

    Buffer<Edge> edges;
    Buffer<Real> primary_edges_pmf;
    Buffer<Real> primary_edges_cdf;
    Buffer<Real> secondary_edges_pmf;
    Buffer<Real> secondary_edges_cdf;
    // For secondary edges
    std::unique_ptr<EdgeTree> edge_tree;
};

using PrimaryEdgeSample = TPrimaryEdgeSample<Real>;
using SecondaryEdgeSample = TSecondaryEdgeSample<Real>;

DEVICE
inline Vector3f get_v0(const Shape *shapes, const Edge &edge) {
    return get_vertex(shapes[edge.shape_id], edge.v0);
}

DEVICE
inline Vector3f get_v1(const Shape *shapes, const Edge &edge) {
    return get_vertex(shapes[edge.shape_id], edge.v1);
}

DEVICE
inline Vector3f get_non_shared_v0(
        const Shape *shapes, const Edge &edge) {
    auto ind = get_indices(shapes[edge.shape_id], edge.f0);
    for (int i = 0; i < 3; i++) {
        if (ind[i] != edge.v0 && ind[i] != edge.v1) {
            return get_vertex(shapes[edge.shape_id], ind[i]);
        }
    }
    return get_v0(shapes, edge);
}

DEVICE
inline Vector3f get_non_shared_v1(
        const Shape *shapes, const Edge &edge) {
    // Unfotunately the below wouldn't work because we sometimes
    // merge edge's faces when there are duplicated edges

    // auto ind = get_indices(shapes[edge.shape_id], edge.f1);
    // for (int i = 0; i < 3; i++) {
    //     if (ind[i] != edge.v0 && ind[i] != edge.v1) {
    //         return get_vertex(shapes[edge.shape_id], ind[i]);
    //     }
    // }
    // return Vector3{0.f, 0.f, 0.f};

    // So we have to go for a slightly more expensive alternative
    auto ind = get_indices(shapes[edge.shape_id], edge.f1);
    auto v0 = get_v0(shapes, edge);
    auto v1 = get_v1(shapes, edge);
    for (int i = 0; i < 3; i++) {
        auto v = get_vertex(shapes[edge.shape_id], ind[i]);
        if (v != v0 && v != v1) {
            return v;
        }
    }
    return v1;
}

DEVICE
inline Vector3 get_n0(const Shape *shapes, const Edge &edge) {
    auto v0 = Vector3{get_v0(shapes, edge)};
    auto v1 = Vector3{get_v1(shapes, edge)};
    auto ns_v0 = Vector3{get_non_shared_v0(shapes, edge)};
    auto n = cross(v0 - ns_v0, v1 - ns_v0);
    auto n_len_sq = length_squared(n);
    if (n_len_sq < Real(1e-20)) {
        return Vector3{0, 0, 0};
    }
    n /= sqrt(n_len_sq);
    return n;
}

DEVICE
inline Vector3 get_n1(const Shape *shapes, const Edge &edge) {
    auto v0 = Vector3{get_v0(shapes, edge)};
    auto v1 = Vector3{get_v1(shapes, edge)};
    auto ns_v1 = Vector3{get_non_shared_v1(shapes, edge)};
    auto n = cross(v1 - ns_v1, v0 - ns_v1);
    auto n_len_sq = length_squared(n);
    if (n_len_sq < Real(1e-20)) {
        return Vector3{0, 0, 0};
    }
    n /= sqrt(n_len_sq);
    return n;

}

DEVICE
inline bool is_silhouette(const Shape *shapes, const Vector3 &p, const Edge &edge) {
    auto v0 = Vector3{get_v0(shapes, edge)};
    auto v1 = Vector3{get_v1(shapes, edge)};
    if (edge.f0 == -1 || edge.f1 == -1) {
        // Only adjacent to one face
        if (edge.f0 != -1) {
            auto ns_v0 = Vector3{get_non_shared_v0(shapes, edge)};
            auto n0 = cross(v0 - ns_v0, v1 - ns_v0);
            auto n0_len_sq = length_squared(n0);
            if (n0_len_sq < Real(1e-20)) {
                // Degenerate vertices
                return false;
            }
        }
        if (edge.f1 != -1) {
            auto ns_v1 = Vector3{get_non_shared_v1(shapes, edge)};
            auto n1 = cross(v1 - ns_v1, v0 - ns_v1);
            auto n1_len_sq = length_squared(n1);
            if (n1_len_sq < Real(1e-20)) {
                // Degenerate vertices
                return false;
            }
        }
        return true;
    }
    auto ns_v0 = Vector3{get_non_shared_v0(shapes, edge)};
    auto ns_v1 = Vector3{get_non_shared_v1(shapes, edge)};
    auto n0 = cross(v0 - ns_v0, v1 - ns_v0);
    auto n1 = cross(v1 - ns_v1, v0 - ns_v1);
    auto n0_len_sq = length_squared(n0);
    auto n1_len_sq = length_squared(n1);
    if (n0_len_sq < Real(1e-20) || n1_len_sq < Real(1e-20)) {
        // Degenerate vertices
        return false;
    }
    n0 /= sqrt(n0_len_sq);
    n1 /= sqrt(n1_len_sq);
    if (!has_shading_normals(shapes[edge.shape_id])) {
        // If we are not using Phong normal, every edge is silhouette,
        // except edges with dihedral angle of 0
        if (dot(n0, n1) >= 1 - 1e-6f) {
            return false;
        }
        return true;
    }
    auto frontfacing0 = dot(p - ns_v0, n0) > 0.f;
    auto frontfacing1 = dot(p - ns_v1, n1) > 0.f;
    return (frontfacing0 && !frontfacing1) || (!frontfacing0 && frontfacing1);
}

DEVICE
inline bool is_silhouette_dir(const Shape *shapes, const Vector3 &dir, const Edge &edge) {
    if (edge.f0 == -1 || edge.f1 == -1) {
        // Only adjacent to one face
        return true;
    }
    auto v0 = Vector3{get_v0(shapes, edge)};
    auto v1 = Vector3{get_v1(shapes, edge)};
    auto ns_v0 = Vector3{get_non_shared_v0(shapes, edge)};
    auto ns_v1 = Vector3{get_non_shared_v1(shapes, edge)};
    auto n0 = normalize(cross(v0 - ns_v0, v1 - ns_v0));
    auto n1 = normalize(cross(v1 - ns_v1, v0 - ns_v1));
    if (!has_shading_normals(shapes[edge.shape_id])) {
        // If we are not using Phong normal, every edge is silhouette,
        // except edges with dihedral angle of 0
        if (dot(n0, n1) >= 1 - 1e-6f) {
            return false;
        }
        return true;
    }
    auto frontfacing0 = dot(dir, n0) > 0.f;
    auto frontfacing1 = dot(dir, n1) > 0.f;
    return (frontfacing0 && !frontfacing1) || (!frontfacing0 && frontfacing1);
}


DEVICE
inline Real compute_exterior_dihedral_angle(const Shape *shapes, const Edge &edge) {
    auto exterior_dihedral = Real(M_PI);
    if (edge.f1 != -1) {
        auto n0 = get_n0(shapes, edge);
        auto n1 = get_n1(shapes, edge);
        exterior_dihedral = acos(clamp(dot(n0, n1), Real(-1), Real(1)));
    }
    return exterior_dihedral;
}

void initialize_ltc_table(bool use_gpu);

void sample_primary_edges(const Scene &scene,
                          const BufferView<PrimaryEdgeSample> &samples,
                          const float *d_rendered_image,
                          const ChannelInfo &channel_info,
                          BufferView<PrimaryEdgeRecord> edge_records,
                          BufferView<Ray> rays,
                          BufferView<RayDifferential> primary_ray_differentials,
                          BufferView<Vector3> throughputs,
                          BufferView<Real> channel_multipliers);

void update_primary_edge_weights(const Scene &scene,
                                 const BufferView<PrimaryEdgeRecord> &edge_records,
                                 const BufferView<Intersection> &shading_isects,
                                 const ChannelInfo &channel_info,
                                 BufferView<Vector3> throughputs,
                                 BufferView<Real> channel_multipliers);

void compute_primary_edge_derivatives(const Scene &scene,
                                      const BufferView<PrimaryEdgeRecord> &edge_records,
                                      const BufferView<Real> &edge_contribs,
                                      BufferView<DShape> d_shapes,
                                      DCamera d_camera,
                                      float *screen_gradient_image);

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
                            BufferView<RayDifferential> &bsdf_ray_differentials,
                            BufferView<Vector3> new_throughputs,
                            BufferView<Real> edge_min_roughness);

void update_secondary_edge_weights(const Scene &scene,
                                   const BufferView<int> &active_pixels,
                                   const BufferView<SurfacePoint> &shading_points,
                                   const BufferView<Intersection> &edge_isects,
                                   const BufferView<SurfacePoint> &edge_surface_points,
                                   const BufferView<SecondaryEdgeRecord> &edge_records,
                                   BufferView<Vector3> edge_throughputs);

void accumulate_secondary_edge_derivatives(const Scene &scene,
                                           const BufferView<int> &active_pixels,
                                           const BufferView<SurfacePoint> &shading_points,
                                           const BufferView<SecondaryEdgeRecord> &edge_records,
                                           const BufferView<Vector3> &edge_surface_points,
                                           const BufferView<Real> &edge_contribs,
                                           BufferView<SurfacePoint> d_points,
                                           BufferView<DShape> d_shapes);
