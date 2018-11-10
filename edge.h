#pragma once

#include "redner.h"
#include "shape.h"
#include "camera.h"

struct Scene;

struct Edge {
    int shape_id = -1;
    int v0, v1;
    int f0, f1;

    DEVICE inline bool operator==(const Edge &other) const {
        return shape_id == other.shape_id &&
            v0 == other.v0 && v1 == other.v1 &&
            f0 == other.f0 && f1 == other.f1;
    }
};

struct PrimaryEdgeRecord {
    int shape_id = -1;
    int v0, v1;
    Vector2 edge_pt;
};

struct SecondaryEdgeRecord {
    int shape_id = -1;
    int v0, v1;
    Vector3 edge_pt;
    Vector3 mwt;
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

using PrimaryEdgeSample = TPrimaryEdgeSample<Real>;
using SecondaryEdgeSample = TSecondaryEdgeSample<Real>;

template <typename EdgeType>
DEVICE
inline Vector3f get_v0(const Shape *shapes, const EdgeType &edge) {
    return get_vertex(shapes[edge.shape_id], edge.v0);
}

template <typename EdgeType>
DEVICE
inline Vector3f get_v1(const Shape *shapes, const EdgeType &edge) {
    return get_vertex(shapes[edge.shape_id], edge.v1);
}

DEVICE
inline Vector3 get_n0(const Shape *shapes, const Edge &edge) {
    return get_normal(shapes[edge.shape_id], edge.f0);
}

DEVICE
inline Vector3 get_n1(const Shape *shapes, const Edge &edge) {
    return get_normal(shapes[edge.shape_id], edge.f1);
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
    return Vector3{0.f, 0.f, 0.f};
}

DEVICE
inline Vector3f get_non_shared_v1(
        const Shape *shapes, const Edge &edge) {
    auto ind = get_indices(shapes[edge.shape_id], edge.f1);
    for (int i = 0; i < 3; i++) {
        if (ind[i] != edge.v0 && ind[i] != edge.v1) {
            return get_vertex(shapes[edge.shape_id], ind[i]);
        }
    }
    return Vector3{0.f, 0.f, 0.f};
}

DEVICE
inline bool is_silhouette(const Shape *shapes,
                          const Vector3 &p,
                          const Edge &edge) {
    if (!has_shading_normals(shapes[edge.shape_id])) {
        // If we are not using Phong normal, every edge is silhouette
        return true;
    }
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
    auto frontfacing0 = dot(p - ns_v0, n0) > 0.f;
    auto frontfacing1 = dot(p - ns_v1, n1) > 0.f;
    return (frontfacing0 && !frontfacing1) || (!frontfacing0 && frontfacing1);
}

struct EdgeSampler {
    EdgeSampler() {}
    EdgeSampler(const std::vector<const Shape*> &shapes,
                const Scene &scene);

    Buffer<Edge> edges;
    Buffer<Real> primary_edges_pmf;
    Buffer<Real> primary_edges_cdf;
    Buffer<Real> secondary_edges_pmf;
    Buffer<Real> secondary_edges_cdf;
};

void sample_primary_edges(const Scene &scene,
                          const BufferView<PrimaryEdgeSample> &samples,
                          const float *d_rendered_image,
                          BufferView<PrimaryEdgeRecord> edge_records,
                          BufferView<Ray> rays,
                          BufferView<Vector3> throughputs);

void compute_primary_edge_derivatives(const Scene &scene,
                                      const BufferView<PrimaryEdgeRecord> &edge_records,
                                      const BufferView<Real> &edge_contribs,
                                      BufferView<DVertex> d_vertices,
                                      BufferView<DCameraInst> d_cameras);

void sample_secondary_edges(const Scene &scene,
                            const BufferView<int> &active_pixels,
                            const BufferView<SecondaryEdgeSample> &samples,
                            const BufferView<Ray> &incoming_rays,
                            const BufferView<Intersection> &shading_isects,
                            const BufferView<SurfacePoint> &shading_points,
                            const BufferView<Vector3> &throughputs,
                            const BufferView<Real> &min_roughness,
                            const float *d_rendered_image,
                            BufferView<SecondaryEdgeRecord> edge_records,
                            BufferView<Ray> rays,
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
                                           BufferView<DVertex> d_vertices);
