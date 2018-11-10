#pragma once

#include "redner.h"
#include "vector.h"
#include "intersection.h"
#include "buffer.h"
#include "ptr.h"

struct Shape {
    Shape() {}
    Shape(ptr<float> vertices,
          ptr<int> indices,
          ptr<float> uvs,
          ptr<float> normals,
          int num_vertices,
          int num_triangles,
          int material_id,
          int light_id) :
        vertices(vertices.get()),
        indices(indices.get()),
        uvs(uvs.get()),
        normals(normals.get()),
        num_vertices(num_vertices),
        num_triangles(num_triangles),
        material_id(material_id),
        light_id(light_id) {}

    inline bool has_uvs() const {
        return uvs != nullptr;
    }

    inline bool has_normals() const {
        return normals != nullptr;
    }

    float *vertices;
    int *indices;
    float *uvs;
    float *normals;
    int num_vertices;
    int num_triangles;
    int material_id;
    int light_id;
};

struct DShape {
    DShape() {}
    DShape(ptr<float> vertices,
           ptr<float> uvs,
           ptr<float> normals)
        : vertices(vertices.get()),
          uvs(uvs.get()),
          normals(normals.get()) {}

    float *vertices;
    float *uvs;
    float *normals;
};

struct DVertex {
    int shape_id = -1, vertex_id = -1;
    Vector3 d_v = Vector3{0, 0, 0};
    Vector2 d_uv = Vector2{0, 0};
    Vector3 d_n = Vector3{0, 0, 0};

    DEVICE inline bool operator<(const DVertex &other) const {
        if (shape_id != other.shape_id) {
            return shape_id < other.shape_id;
        } else {
            return vertex_id < other.vertex_id;
        }
    }

    DEVICE inline bool operator==(const DVertex &other) const {
        return shape_id == other.shape_id && vertex_id == other.vertex_id;
    }

    DEVICE inline DVertex operator+(const DVertex &other) const {
        return DVertex{shape_id, vertex_id,
                       d_v + other.d_v,
                       d_uv + other.d_uv,
                       d_n + other.d_n};
    }
};

DEVICE
inline Vector3f get_vertex(const Shape &shape, int index) {
    return Vector3f{shape.vertices[3 * index + 0],
                    shape.vertices[3 * index + 1],
                    shape.vertices[3 * index + 2]};
}

DEVICE
inline Vector3i get_indices(const Shape &shape, int index) {
    return Vector3i{shape.indices[3 * index + 0],
                    shape.indices[3 * index + 1],
                    shape.indices[3 * index + 2]};
}

DEVICE
inline bool has_uvs(const Shape &shape) {
    return shape.uvs != nullptr;
}

DEVICE
inline Vector2f get_uv(const Shape &shape, int index) {
    return Vector2f{shape.uvs[2 * index + 0],
                    shape.uvs[2 * index + 1]};
}

DEVICE
inline void accumulate_uv(DShape &d_shape, int index, const Vector2 &d) {
    d_shape.uvs[2 * index + 0] += d[0];
    d_shape.uvs[2 * index + 1] += d[1];
}

DEVICE
inline bool has_shading_normals(const Shape &shape) {
    return shape.normals != nullptr;
}

DEVICE
inline Vector3f get_shading_normal(const Shape &shape, int index) {
    return Vector3f{shape.normals[3 * index + 0],
                    shape.normals[3 * index + 1],
                    shape.normals[3 * index + 2]};
}

DEVICE
inline Vector3 get_normal(const Shape &shape, int tri_index) {
    auto indices = get_indices(shape, tri_index);
    auto v0 = Vector3{get_vertex(shape, indices[0])};
    auto v1 = Vector3{get_vertex(shape, indices[1])};
    auto v2 = Vector3{get_vertex(shape, indices[2])};
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    return normalize(cross(e1, e2));
}

DEVICE
inline void accumulate_shading_normal(DShape &d_shape, int index, const Vector3 &d) {
    d_shape.normals[3 * index + 0] += d[0];
    d_shape.normals[3 * index + 1] += d[1];
    d_shape.normals[3 * index + 2] += d[2];
}

DEVICE
inline Real get_area(const Shape &shape, int index) {
    auto ind = get_indices(shape, index);
    auto v0 = Vector3{get_vertex(shape, ind[0])};
    auto v1 = Vector3{get_vertex(shape, ind[1])};
    auto v2 = Vector3{get_vertex(shape, ind[2])};
    return 0.5f * length(cross(v1 - v0, v2 - v0));
}

DEVICE
inline void d_get_area(const Shape &shape, int index,
                       const Real d_area, DVertex *d_vertices) {
    auto ind = get_indices(shape, index);
    auto v0 = Vector3{get_vertex(shape, ind[0])};
    auto v1 = Vector3{get_vertex(shape, ind[1])};
    auto v2 = Vector3{get_vertex(shape, ind[2])};
    auto dir = cross(v1 - v0, v2 - v0);
    // auto area = 0.5f * length(dir);
    auto d_len = d_area * 0.5f;
    auto d_dir = d_length(dir, d_len);
    auto d_e1 = Vector3{0, 0, 0};
    auto d_e2 = Vector3{0, 0, 0};
    d_cross(v1 - v0, v2 - v0, d_dir, d_e1, d_e2);
    d_vertices[0].d_v -= (d_e1 + d_e2);
    d_vertices[1].d_v += d_e1;
    d_vertices[2].d_v += d_e2;
}

DEVICE
inline SurfacePoint sample_shape(
        const Shape &shape, int index, const Vector2 &sample) {
    auto ind = get_indices(shape, index);
    auto v0 = Vector3{get_vertex(shape, ind[0])};
    auto v1 = Vector3{get_vertex(shape, ind[1])};
    auto v2 = Vector3{get_vertex(shape, ind[2])};
    auto a = sqrt(sample[0]);
    auto b1 = 1.f - a;
    auto b2 = a * sample[1];
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto n = cross(e1, e2);
    auto normalized_n = normalize(n);
    return SurfacePoint{
        v0 + e1 * b1 + e2 * b2,
        normalized_n,
        Frame(normalized_n), // TODO: phong interpolate this
        sample}; // TODO: give true light source uv
}

DEVICE
inline void d_sample_shape(
        const Shape &shape, int index, const Vector2 &sample,
        const SurfacePoint &d_point, DVertex *d_vertices) {
    auto ind = get_indices(shape, index);
    auto v0 = Vector3{get_vertex(shape, ind[0])};
    auto v1 = Vector3{get_vertex(shape, ind[1])};
    auto v2 = Vector3{get_vertex(shape, ind[2])};
    auto a = sqrt(sample[0]);
    auto b1 = 1.f - a;
    auto b2 = a * sample[1];
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto n = cross(e1, e2);
    auto normalized_n = normalize(n);
    // point = SurfacePoint{
    //     v0 + e1 * b1 + e2 * b2,
    //     normalized_n,
    //     Frame(normalized_n),
    //     sample};
    // No need to propagate to b1 b2
    auto d_v0 = d_point.position;
    auto d_e1 = d_point.position * b1;
    auto d_e2 = d_point.position * b2;
    auto d_normalized_n = d_point.geom_normal;
    d_normalized_n += d_point.shading_frame[2];
    d_coordinate_system(normalized_n, d_point.shading_frame[0], d_point.shading_frame[1],
        d_normalized_n);
    // auto normalized_n = normalize(n);
    auto d_n = d_normalize(n, d_normalized_n);
    // n = cross(e1, e2)
    d_cross(e1, e2, d_n, d_e1, d_e2);
    // e1 = v1 - v0
    auto d_v1 = d_e1;
    d_v0 -= d_e1;
    // e2 = v2 - v0
    auto d_v2 = d_e2;
    d_v0 -= d_e2;
    d_vertices[0].d_v += d_v0;
    d_vertices[1].d_v += d_v1;
    d_vertices[2].d_v += d_v2;
}

DEVICE
inline SurfacePoint intersect_shape(
        const Shape &shape, int index, const Ray &ray) {
    auto ind = get_indices(shape, index);
    auto v0 = Vector3{get_vertex(shape, ind[0])};
    auto v1 = Vector3{get_vertex(shape, ind[1])};
    auto v2 = Vector3{get_vertex(shape, ind[2])};
    Vector2 uvs0, uvs1, uvs2;
    if (has_uvs(shape)) {
        uvs0 = get_uv(shape, ind[0]);
        uvs1 = get_uv(shape, ind[1]);
        uvs2 = get_uv(shape, ind[2]);
    } else {
        uvs0 = Vector2{0.f, 0.f};
        uvs1 = Vector2{1.f, 0.f};
        uvs2 = Vector2{1.f, 1.f};
    }
    auto uvt = intersect(v0, v1, v2, ray);
    auto u = uvt[0];
    auto v = uvt[1];
    auto w = 1.f - (u + v);
    auto t = uvt[2];
    auto uv = w * uvs0 + u * uvs1 + v * uvs2;
    auto hit_pos = ray.org + ray.dir * t;
    auto geom_normal = normalize(cross(v1 - v0, v2 - v0));
    auto shading_normal = geom_normal;
    if (has_shading_normals(shape)) {
        auto n0 = get_shading_normal(shape, ind[0]);
        auto n1 = get_shading_normal(shape, ind[1]);
        auto n2 = get_shading_normal(shape, ind[2]);
        shading_normal = normalize(w * n0 + u * n1 + v * n2);
        // Flip geometric normal to the same side of shading normal
        if (dot(geom_normal, shading_normal) < 0.f) {
            geom_normal = -geom_normal;
        }
    }
    return SurfacePoint{
        hit_pos, geom_normal, Frame(shading_normal), uv};
}

DEVICE
inline void d_intersect_shape(
        const Shape &shape, int index, const Ray &ray,
        const SurfacePoint &d_point,
        DRay &d_ray,
        DVertex *d_vertices) {
    auto ind = get_indices(shape, index);
    auto v0 = Vector3{get_vertex(shape, ind[0])};
    auto v1 = Vector3{get_vertex(shape, ind[1])};
    auto v2 = Vector3{get_vertex(shape, ind[2])};
    Vector2 uvs0, uvs1, uvs2;
    if (has_uvs(shape)) {
        uvs0 = get_uv(shape, ind[0]);
        uvs1 = get_uv(shape, ind[1]);
        uvs2 = get_uv(shape, ind[2]);
    } else {
        uvs0 = Vector2{0.f, 0.f};
        uvs1 = Vector2{1.f, 0.f};
        uvs2 = Vector2{1.f, 1.f};
    }
    auto uvt = intersect(v0, v1, v2, ray);
    auto u = uvt[0];
    auto v = uvt[1];
    auto w = 1.f - (u + v);
    auto t = uvt[2];
    // auto uv = w * uvs0 + u * uvs1 + v * uvs2;
    // auto hit_pos = ray.org + ray.dir * t;
    auto unnormalized_geom_normal = cross(v1 - v0, v2 - v0);
    auto geom_normal = normalize(unnormalized_geom_normal);
    auto shading_normal = geom_normal;
    auto geom_normal_flipped = false;
    if (has_shading_normals(shape)) {
        auto n0 = get_shading_normal(shape, ind[0]);
        auto n1 = get_shading_normal(shape, ind[1]);
        auto n2 = get_shading_normal(shape, ind[2]);
        shading_normal = normalize(w * n0 + u * n1 + v * n2);
        // Flip geometric normal to the same side of shading normal
        if (dot(geom_normal, shading_normal) < 0.f) {
            geom_normal = -geom_normal;
            geom_normal_flipped = true;
        }
    }
    // point = SurfacePoint{hit_pos, geom_normal, Frame(shading_normal), uv}
    
    auto d_geom_normal = d_point.geom_normal;
    auto d_u = Real(0), d_v = Real(0), d_w = Real(0);
    if (has_shading_normals(shape)) {
        if (geom_normal_flipped) {
            d_geom_normal = -d_geom_normal;
        }
        // shading_normal = normalize(w * n0 + u * n1 + v * n2)
        auto n0 = get_shading_normal(shape, ind[0]);
        auto n1 = get_shading_normal(shape, ind[1]);
        auto n2 = get_shading_normal(shape, ind[2]);
        auto d_shading_normal = d_point.shading_frame[2];
        // differentiate through frame construction
        d_coordinate_system(shading_normal, d_point.shading_frame[0], d_point.shading_frame[1],
                            d_shading_normal);
        auto d_unnormalized_normal = d_normalize(w * n0 + u * n1 + v * n2, d_shading_normal);
        d_w += sum(d_unnormalized_normal * n0);
        d_u += sum(d_unnormalized_normal * n1);
        d_v += sum(d_unnormalized_normal * n2);
        auto d_n0 = d_unnormalized_normal * w;
        auto d_n1 = d_unnormalized_normal * u;
        auto d_n2 = d_unnormalized_normal * v;
        d_vertices[0].d_n += d_n0;
        d_vertices[1].d_n += d_n1;
        d_vertices[2].d_n += d_n2;
    } else {
        d_geom_normal += d_point.shading_frame[2];
        d_coordinate_system(shading_normal, d_point.shading_frame[0], d_point.shading_frame[1],
                            d_geom_normal);
    }
    // geom_normal = normalize(unnormalized_geom_normal)
    auto d_unnormalized_geom_normal = d_normalize(unnormalized_geom_normal, d_geom_normal);
    // unnormalized_geom_normal = cross(v1 - v0, v2 - v0)
    auto d_v1_v0 = Vector3{0, 0, 0};
    auto d_v2_v0 = Vector3{0, 0, 0};
    d_cross(v1 - v0, v2 - v0, d_unnormalized_geom_normal, d_v1_v0, d_v2_v0);
    auto d_v0 = - d_v1_v0 - d_v2_v0;
    auto d_v1 = d_v1_v0;
    auto d_v2 = d_v2_v0;
    // hit_pos = ray.org + ray.dir * t
    auto d_hit_pos = d_point.position;
    d_ray.org += d_hit_pos;
    d_ray.dir += d_hit_pos * t;
    auto d_t = sum(d_hit_pos * ray.dir);
    // uv = w * uvs0 + u * uvs1 + v * uvs2
    auto d_uv = d_point.uv;
    d_w += sum(d_uv * uvs0);
    d_u += sum(d_uv * uvs1);
    d_v += sum(d_uv * uvs2);
    auto d_uvs0 = d_uv * w;
    auto d_uvs1 = d_uv * u;
    auto d_uvs2 = d_uv * v;
    // auto t = uvt[2];
    auto d_uvt = Vector3{0, 0, 0};
    d_uvt[2] += d_t;
    // w = 1.f - (u + v)
    d_u -= d_w;
    d_v -= d_w;
    // u = uvt[0]
    // v = uvt[1]
    d_uvt[0] += d_u;
    d_uvt[1] += d_v;
    // uvt = intersect(v0, v1, v2, ray)
    d_intersect(v0, v1, v2, ray, d_uvt, d_v0, d_v1, d_v2, d_ray);
    if (has_uvs(shape)) {
        d_vertices[0].d_uv += d_uvs0;
        d_vertices[1].d_uv += d_uvs1;
        d_vertices[2].d_uv += d_uvs2;
    }
    d_vertices[0].d_v += d_v0;
    d_vertices[1].d_v += d_v1;
    d_vertices[2].d_v += d_v2;
}

void accumulate_vertex(const BufferView<DVertex> &d_vertices,
                       BufferView<DShape> shapes,
                       bool use_gpu);

void test_d_intersect();
void test_d_sample_shape();
