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
inline SurfacePoint sample_shape(const Shape &shape, int index, const Vector2 &sample) {
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
        sample, // TODO: give true light source uv
        Vector2{0, 0}, // TODO: inherit derivatives from previous path vertex
    	Vector2{0, 0}}; 
}

DEVICE
inline void d_sample_shape(const Shape &shape, int index, const Vector2 &sample,
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
    //     sample,
	//     Vector2{0, 0},
	//     Vector2{0, 0}};
    // No need to propagate to b1 b2
    auto d_v0 = d_point.position;
    auto d_e1 = d_point.position * b1;
    auto d_e2 = d_point.position * b2;
    auto d_normalized_n = d_point.geom_normal;
    d_normalized_n += d_point.shading_frame[2];
    d_coordinate_system(
    	normalized_n, d_point.shading_frame[0], d_point.shading_frame[1], d_normalized_n);
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

// Derivatives of projection of a point to barycentric coordinate
// http://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
DEVICE
inline
void barycentric(const Vector3 &p0,
                 const Vector3 &p1,
                 const Vector3 &p2,
                 Vector3 &d_b0_d_p,
                 Vector3 &d_b1_d_p) {
    auto e1 = p1 - p0;
    auto e2 = p2 - p0;
    // auto e0 = p - p0;
    auto e0_dp = Vector3{1, 1, 1};
    auto d11 = dot(e1, e1);
    auto d12 = dot(e1, e2);
    auto d22 = dot(e2, e2);
    // auto d01 = dot(e0, e1);
    auto d01_dp = e0_dp * e1;
    // auto d02 = dot(e0, e2);
    auto d02_dp = e0_dp * e2;
    auto inv_denom = Real(1) / (d11 * d22 - d12 * d12);
    // auto b0 = (d22 * d01 - d12 * d02) * inv_denom;
    // auto b1 = (d11 * d02 - d12 * d01) * inv_denom;
    d_b0_d_p = (d22 * d01_dp - d12 * d02_dp) * inv_denom;
    d_b1_d_p = (d11 * d02_dp - d12 * d01_dp) * inv_denom;
}

DEVICE
inline
void d_barycentric(const Vector3 &p0,
                   const Vector3 &p1,
                   const Vector3 &p2,
                   const Vector3 &d_d_b0_d_p,
                   const Vector3 &d_d_b1_d_p,
                   Vector3 &d_p0,
                   Vector3 &d_p1,
                   Vector3 &d_p2) {
    auto e1 = p1 - p0;
    auto e2 = p2 - p0;
    // auto e0 = p - p0;
    auto e0_dp = Vector3{1, 1, 1};
    auto d11 = dot(e1, e1);
    auto d12 = dot(e1, e2);
    auto d22 = dot(e2, e2);
    // auto d01 = dot(e0, e1);
    auto d01_dp = dot(e0_dp, e1);
    // auto d02 = dot(e0, e2);
    auto d02_dp = dot(e0_dp, e2);
    auto inv_denom = Real(1) / (d11 * d22 - d12 * d12);
    // auto b0 = (d22 * d01 - d12 * d02) * inv_denom;
    // auto b1 = (d11 * d02 - d12 * d01) * inv_denom;
    // d_b0_d_p = (d22 * d01_dp - d12 * d02_dp) * inv_denom;
    // d_b1_d_p = (d11 * d02_dp - d12 * d01_dp) * inv_denom;
    auto d_d22 = d_d_b0_d_p * d01_dp * inv_denom;
    auto d_d01_dp = d_d_b0_d_p * d22 * inv_denom;
    auto d_d12 = -d_d_b0_d_p * d02_dp * inv_denom;
    auto d_d02_dp = -d_d_b0_d_p * d12 * inv_denom;
    auto d_d11 = d_d_b1_d_p * d02_dp * inv_denom;
    d_d02_dp += d_d_b1_d_p * d11 * inv_denom;
    d_d12 += (-d_d_b1_d_p * d01_dp * inv_denom);
    d_d01_dp += (-d_d_b1_d_p * d12 * inv_denom);
    auto d_inv_denom = d_d_b0_d_p * (d22 * d01_dp - d12 * d02_dp) +
                       d_d_b1_d_p * (d11 * d02_dp - d12 * d01_dp);
    // inv_denom = 1 / (d11 * d22 - d12 * d12)
    d_d11 += d_inv_denom * (-square(inv_denom) * d22);
    d_d22 += d_inv_denom * (-square(inv_denom) * d11);
    d_d12 += d_inv_denom * (2 * square(inv_denom) * d12);
    // ignore d_e0_dp
    // d02_dp = dot(e0_dp, e2)
    auto d_e2 = d_d02_dp * e0_dp;
    // d01_dp = dot(e0_dp, e1)
    auto d_e1 = d_d01_dp * e0_dp;
    // d11 = dot(e1, e1)
    d_e1 += 2 * d_d11 * e1;
    // d12 = dot(e1, e2)
    d_e1 += d_d12 * e2;
    d_e2 += d_d12 * e1;
    // d22 = dot(e2, e2)
    d_e2 += 2 * d_d22 * e2;
    // e1 = p1 - p0
    d_p1 += d_e1;
    d_p0 -= d_e1;
    // e2 = p2 - p0
    d_p2 += d_e2;
    d_p0 -= d_e2;
}

DEVICE
inline SurfacePoint intersect_shape(const Shape &shape,
                                    int index,
                                    const Ray &ray,
                                    const RayDifferential &ray_differential,
                                    RayDifferential &new_ray_differential) {
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

    // Surface derivative for ray differentials
    // Want to know duv/dx through duv/dp * dp/dx
    // We project p to barycentric coordinate, then reconstruct
    // p -> u, v -> uv
    auto dudp = Vector3{0, 0, 0};
    auto dvdp = Vector3{0, 0, 0};
    barycentric(v0, v1, v2, dudp, dvdp);
    auto duv_dpx = (1.f - dudp[0] - dvdp[0]) * uvs0 + dudp[0] * uvs1 + dvdp[0] * uvs2;
    auto duv_dpy = (1.f - dudp[1] - dvdp[1]) * uvs0 + dudp[1] * uvs1 + dvdp[1] * uvs2;
    auto duv_dpz = (1.f - dudp[2] - dvdp[2]) * uvs0 + dudp[2] * uvs1 + dvdp[2] * uvs2;
    // Igehy 1999 Eq. 10
    auto org_dx = ray_differential.org_dx;
    auto dir_dx = ray_differential.dir_dx;
    auto dtdx = -dot((org_dx + t * dir_dx), geom_normal) / dot(ray.dir, geom_normal);
    auto dpdx = (org_dx + t * dir_dx) + dtdx * ray.dir;
    auto org_dy = ray_differential.org_dy;
    auto dir_dy = ray_differential.dir_dy;
    auto dtdy = -dot((org_dy + t * dir_dy), geom_normal) / dot(ray.dir, geom_normal);
    auto dpdy = (org_dy + t * dir_dy) + dtdy * ray.dir;
    auto du_dxy = Vector2{duv_dpx[0] * dpdx.x + duv_dpy[0] * dpdx.y + duv_dpz[0] * dpdx.z,
                          duv_dpx[0] * dpdy.x + duv_dpy[0] * dpdy.y + duv_dpz[0] * dpdy.z};
    auto dv_dxy = Vector2{duv_dpx[1] * dpdx.x + duv_dpy[1] * dpdx.y + duv_dpz[1] * dpdx.z,
                          duv_dpx[1] * dpdy.x + duv_dpy[1] * dpdy.y + duv_dpz[1] * dpdy.z};
    auto shading_normal = geom_normal;
    auto dn_dx = Vector3{0, 0, 0};
    auto dn_dy = Vector3{0, 0, 0};
    if (has_shading_normals(shape)) {
        auto n0 = get_shading_normal(shape, ind[0]);
        auto n1 = get_shading_normal(shape, ind[1]);
        auto n2 = get_shading_normal(shape, ind[2]);

        auto nn = w * n0 + u * n1 + v * n2;

        // Compute dndx & dndy
        auto dnn_dpx = (1.f - dudp[0] - dvdp[0]) * n0 + dudp[0] * n1 + dvdp[0] * n2;
        auto dnn_dpy = (1.f - dudp[1] - dvdp[1]) * n0 + dudp[1] * n1 + dvdp[1] * n2;
        auto dnn_dpz = (1.f - dudp[2] - dvdp[2]) * n0 + dudp[2] * n1 + dvdp[2] * n2;
        auto dnn_dx = dnn_dpx * dpdx.x + dnn_dpy * dpdx.y + dnn_dpz * dpdx.z;
        auto dnn_dy = dnn_dpx * dpdy.x + dnn_dpy * dpdy.y + dnn_dpz * dpdy.z;
        // normalization derivative
        auto nn_len_sq = dot(nn, nn);
        auto nn_denom = pow(nn_len_sq, Real(3.0/2.0));
        dn_dx = (nn_len_sq * dnn_dx - dot(nn, dnn_dx) * nn) / nn_denom;
        dn_dy = (nn_len_sq * dnn_dy - dot(nn, dnn_dy) * nn) / nn_denom;

        // Shading normal computation
        shading_normal = normalize(nn);
        // Flip geometric normal to the same side of shading normal
        if (dot(geom_normal, shading_normal) < 0.f) {
            geom_normal = -geom_normal;
        }
    }
    // Update ray differential
    new_ray_differential.org_dx = dpdx;
    new_ray_differential.org_dy = dpdy;
    new_ray_differential.dir_dx = dir_dx;
    new_ray_differential.dir_dy = dir_dy;
    return SurfacePoint{hit_pos, geom_normal, Frame(shading_normal), uv,
        du_dxy, dv_dxy, dn_dx, dn_dy};
}

DEVICE
inline void d_intersect_shape(
        const Shape &shape,
        int index,
        const Ray &ray,
        const RayDifferential &ray_differential,
        const SurfacePoint &d_point,
        const RayDifferential &d_new_ray_differential,
        DRay &d_ray,
        RayDifferential &d_ray_differential,
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
    // uv = w * uvs0 + u * uvs1 + v * uvs2
    // hit_pos = ray.org + ray.dir * t
    auto unnormalized_geom_normal = cross(v1 - v0, v2 - v0);
    auto geom_normal = normalize(unnormalized_geom_normal);
    // Surface derivative for ray differential
    // Want to know duv/dp
    // We project p to barycentric coordinate, then reconstruct
    // p -> u, v -> uv
    auto dudp = Vector3{0, 0, 0};
    auto dvdp = Vector3{0, 0, 0};
    barycentric(v0, v1, v2, dudp, dvdp);
    auto duv_dpx = (1.f - dudp[0] - dvdp[0]) * uvs0 + dudp[0] * uvs1 + dvdp[0] * uvs2;
    auto duv_dpy = (1.f - dudp[1] - dvdp[1]) * uvs0 + dudp[1] * uvs1 + dvdp[1] * uvs2;
    auto duv_dpz = (1.f - dudp[2] - dvdp[2]) * uvs0 + dudp[2] * uvs1 + dvdp[2] * uvs2;
    // Ighey 1999 Eq. 10
    auto org_dx = ray_differential.org_dx;
    auto dir_dx = ray_differential.dir_dx;
    auto dtdx = -dot((org_dx + t * dir_dx), geom_normal) / dot(ray.dir, geom_normal);
    auto dpdx = (org_dx + t * dir_dx) + dtdx * ray.dir;
    auto org_dy = ray_differential.org_dy;
    auto dir_dy = ray_differential.dir_dy;
    auto dtdy = -dot((org_dy + t * dir_dy), geom_normal) / dot(ray.dir, geom_normal);
    auto dpdy = (org_dy + t * dir_dy) + dtdy * ray.dir;
    // auto du_dxy = Vector2{duv_dpx[0] * dpdx.x + duv_dpy[0] * dpdx.y + duv_dpz[0] * dpdx.z,
    //                       duv_dpx[0] * dpdy.x + duv_dpy[0] * dpdy.y + duv_dpz[0] * dpdy.z};
    // auto dv_dxy = Vector2{duv_dpx[1] * dpdx.x + duv_dpy[1] * dpdx.y + duv_dpz[1] * dpdx.z,
    //                       duv_dpx[1] * dpdy.x + duv_dpy[1] * dpdy.y + duv_dpz[1] * dpdy.z};
    auto shading_normal = geom_normal;
    auto geom_normal_flipped = false;
    auto dn_dx = Vector3{0, 0, 0};
    auto dn_dy = Vector3{0, 0, 0};
    if (has_shading_normals(shape)) {
        auto n0 = get_shading_normal(shape, ind[0]);
        auto n1 = get_shading_normal(shape, ind[1]);
        auto n2 = get_shading_normal(shape, ind[2]);
        auto nn = w * n0 + u * n1 + v * n2;

        // dndx & dndy computation. might be useful for bump mapping in the future
        auto dnn_dpx = (1.f - dudp[0] - dvdp[0]) * n0 + dudp[0] * n1 + dvdp[0] * n2;
        auto dnn_dpy = (1.f - dudp[1] - dvdp[1]) * n0 + dudp[1] * n1 + dvdp[1] * n2;
        auto dnn_dpz = (1.f - dudp[2] - dvdp[2]) * n0 + dudp[2] * n1 + dvdp[2] * n2;
        auto dnn_dx = dnn_dpx * dpdx.x + dnn_dpy * dpdx.y + dnn_dpz * dpdx.z;
        auto dnn_dy = dnn_dpx * dpdy.x + dnn_dpy * dpdy.y + dnn_dpz * dpdy.z;
        // normalization derivative
        auto nn_len_sq = dot(nn, nn);
        auto nn_denom = pow(nn_len_sq, Real(3.0/2.0));
        dn_dx = (nn_len_sq * dnn_dx - dot(nn, dnn_dx) * nn) / nn_denom;
        dn_dy = (nn_len_sq * dnn_dy - dot(nn, dnn_dy) * nn) / nn_denom;

        // Shading normal computation
        shading_normal = normalize(nn);
        // Flip geometric normal to the same side of shading normal
        if (dot(geom_normal, shading_normal) < 0.f) {
            geom_normal = -geom_normal;
            geom_normal_flipped = true;
        }
    }
    // point = SurfacePoint{hit_pos, geom_normal, Frame(shading_normal), uv, du_dxy, dv_dxy}

    // Backprop
    auto d_geom_normal = d_point.geom_normal;
    // new_ray_differential.org_dx = dpdx;
    // new_ray_differential.org_dy = dpdy;
    // new_ray_differential.dir_dx = dir_dx;
    // new_ray_differential.dir_dy = dir_dy;
    auto d_dpdx = d_new_ray_differential.org_dx;
    auto d_dpdy = d_new_ray_differential.org_dy;
    d_ray_differential.dir_dx += d_new_ray_differential.dir_dx;
    d_ray_differential.dir_dy += d_new_ray_differential.dir_dy;
    auto d_u = Real(0), d_v = Real(0), d_w = Real(0);
    auto d_dudp = Vector3{0, 0, 0};
    auto d_dvdp = Vector3{0, 0, 0};
    if (has_shading_normals(shape)) {
        if (geom_normal_flipped) {
            d_geom_normal = -d_geom_normal;
        }
        auto n0 = get_shading_normal(shape, ind[0]);
        auto n1 = get_shading_normal(shape, ind[1]);
        auto n2 = get_shading_normal(shape, ind[2]);
        auto d_shading_normal = d_point.shading_frame[2];
        // differentiate through frame construction
        d_coordinate_system(shading_normal, d_point.shading_frame[0], d_point.shading_frame[1],
                            d_shading_normal);

        auto nn = w * n0 + u * n1 + v * n2;

        auto dnn_dpx = (1.f - dudp[0] - dvdp[0]) * n0 + dudp[0] * n1 + dvdp[0] * n2;
        auto dnn_dpy = (1.f - dudp[1] - dvdp[1]) * n0 + dudp[1] * n1 + dvdp[1] * n2;
        auto dnn_dpz = (1.f - dudp[2] - dvdp[2]) * n0 + dudp[2] * n1 + dvdp[2] * n2;
        auto dnn_dx = dnn_dpx * dpdx.x + dnn_dpy * dpdx.y + dnn_dpz * dpdx.z;
        auto dnn_dy = dnn_dpx * dpdy.x + dnn_dpy * dpdy.y + dnn_dpz * dpdy.z;
        // normalization derivative
        auto nn_len_sq = dot(nn, nn);
        auto nn_denom = pow(nn_len_sq, Real(3.0/2.0));
        dn_dx = (nn_len_sq * dnn_dx - dot(nn, dnn_dx) * nn) / nn_denom;
        dn_dy = (nn_len_sq * dnn_dy - dot(nn, dnn_dy) * nn) / nn_denom;

        // shading_normal = normalize(nn)
        auto d_nn = d_normalize(nn, d_shading_normal);

        // dn_dx = (nn_len_sq * dnn_dx - dot(nn, dnn_dx) * nn) / nn_denom
        // dn_dy = (nn_len_sq * dnn_dy - dot(nn, dnn_dy) * nn) / nn_denom
        auto d_dn_dx = d_point.dn_dx;
        auto d_dn_dy = d_point.dn_dy;
        auto d_nn_len_sq = (d_dn_dx * dnn_dx + d_dn_dy * dnn_dy) / nn_denom;
        auto d_dnn_dx = d_dn_dx * nn_len_sq / nn_denom;
        auto d_dnn_dy = d_dn_dy * nn_len_sq / nn_denom;
        auto d_dot_nn_dnn_dx = sum(d_dn_dx * nn) / nn_denom;
        auto d_dot_nn_dnn_dy = sum(d_dn_dy * nn) / nn_denom;
        d_nn += (d_dn_dx * dot(nn, dnn_dx) + d_dn_dy * dot(nn, dnn_dy)) / nn_denom;
        auto d_nn_denom = (d_dn_dx * (-dn_dx) + d_dn_dx * (-dn_dy)) / nn_denom;
        // dot(nn, dnn_dx) & dot(nn, dnn_dy)
        d_nn += d_dot_nn_dnn_dx * dnn_dx + d_dot_nn_dnn_dy * dnn_dy;
        d_dnn_dx += (d_dot_nn_dnn_dx + d_dot_nn_dnn_dy) * nn;

        // nn_denom = pow(nn_len_sq, Real(3.0/2.0))
        d_nn_len_sq += d_nn_denom * sqrt(nn_len_sq) * Real(3.0 / 2.0);
        // nn_len_sq = dot(nn, nn)
        d_nn += 2 * d_nn_len_sq * nn;

        // dnn_dx = dnn_dpx * dpdx.x + dnn_dpy * dpdx.y + dnn_dpz * dpdx.z;
        // dnn_dy = dnn_dpx * dpdy.x + dnn_dpy * dpdy.y + dnn_dpz * dpdy.z;
        auto d_dnn_dpx = d_dnn_dx * dpdx.x + d_dnn_dy * dpdy.x;
        auto d_dnn_dpy = d_dnn_dx * dpdx.y + d_dnn_dy * dpdy.y;
        auto d_dnn_dpz = d_dnn_dx * dpdx.z + d_dnn_dy * dpdy.z;
        d_dpdx.x += sum(d_dnn_dx * dnn_dpx);
        d_dpdx.y += sum(d_dnn_dx * dnn_dpy);
        d_dpdx.z += sum(d_dnn_dx * dnn_dpz);
        d_dpdy.x += sum(d_dnn_dy * dnn_dpx);
        d_dpdy.y += sum(d_dnn_dy * dnn_dpy);
        d_dpdy.z += sum(d_dnn_dy * dnn_dpz);
        // dnn_dpx = (1.f - dudp[0] - dvdp[0]) * n0 + dudp[0] * n1 + dvdp[0] * n2
        // dnn_dpy = (1.f - dudp[1] - dvdp[1]) * n0 + dudp[1] * n1 + dvdp[1] * n2
        // dnn_dpz = (1.f - dudp[2] - dvdp[2]) * n0 + dudp[2] * n1 + dvdp[2] * n2
        d_dudp[0] += sum(d_dnn_dpx * (n1 - n0));
        d_dvdp[0] += sum(d_dnn_dpx * (n2 - n0));
        d_dudp[1] += sum(d_dnn_dpy * (n1 - n0));
        d_dvdp[1] += sum(d_dnn_dpy * (n2 - n0));
        d_dudp[2] += sum(d_dnn_dpz * (n1 - n0));
        d_dvdp[2] += sum(d_dnn_dpz * (n2 - n0));
        auto d_n0 = d_dnn_dpx * (1.f - dudp[0] - dvdp[0]) +
                    d_dnn_dpy * (1.f - dudp[1] - dvdp[1]) +
                    d_dnn_dpz * (1.f - dudp[2] - dvdp[2]);
        auto d_n1 = d_dnn_dpx * dudp[0] +
                    d_dnn_dpy * dudp[1] +
                    d_dnn_dpz * dudp[2];
        auto d_n2 = d_dnn_dpx * dvdp[0] +
                    d_dnn_dpy * dvdp[1] +
                    d_dnn_dpz * dvdp[2];

        // nn = w * n0 + u * n1 + v * n2
        d_w += sum(d_nn * n0);
        d_u += sum(d_nn * n1);
        d_v += sum(d_nn * n2);
        d_n0 += d_nn * w;
        d_n1 += d_nn * u;
        d_n2 += d_nn * v;
        d_vertices[0].d_n += d_n0;
        d_vertices[1].d_n += d_n1;
        d_vertices[2].d_n += d_n2;
    } else {
        d_geom_normal += d_point.shading_frame[2];
        d_coordinate_system(shading_normal, d_point.shading_frame[0], d_point.shading_frame[1],
                            d_geom_normal);
    }

    // du_dxy = Vector2{duv_dpx[0] * dpdx.x + duv_dpy[0] * dpdx.y + duv_dpz[0] * dpdx.z,
    //                  duv_dpx[0] * dpdy.x + duv_dpy[0] * dpdy.y + duv_dpz[0] * dpdy.z}
    // dv_dxy = Vector2{duv_dpx[1] * dpdx.x + duv_dpy[1] * dpdx.y + duv_dpz[1] * dpdx.z,
    //                  duv_dpx[1] * dpdy.x + duv_dpy[1] * dpdy.y + duv_dpz[1] * dpdy.z}
    auto d_du_dxy = d_point.du_dxy;
    auto d_dv_dxy = d_point.dv_dxy;
    auto d_duv_dpx = Vector2{d_du_dxy[0] * dpdx.x + d_du_dxy[1] * dpdy.x,
                             d_dv_dxy[0] * dpdx.x + d_du_dxy[1] * dpdy.x};
    auto d_duv_dpy = Vector2{d_du_dxy[0] * dpdx.y + d_du_dxy[1] * dpdy.y,
                             d_dv_dxy[0] * dpdx.y + d_du_dxy[1] * dpdy.y};
    auto d_duv_dpz = Vector2{d_du_dxy[0] * dpdx.z + d_du_dxy[1] * dpdy.z,
                             d_dv_dxy[0] * dpdx.z + d_du_dxy[1] * dpdy.z};
    d_dpdx += Vector3{d_du_dxy[0] * duv_dpx[0] + d_dv_dxy[0] * duv_dpx[1],
                      d_du_dxy[0] * duv_dpy[0] + d_dv_dxy[0] * duv_dpy[1],
                      d_du_dxy[0] * duv_dpz[0] + d_dv_dxy[0] * duv_dpz[1]};
    d_dpdy += Vector3{d_du_dxy[1] * duv_dpx[0] + d_dv_dxy[1] * duv_dpx[1],
                      d_du_dxy[1] * duv_dpy[0] + d_dv_dxy[1] * duv_dpy[1],
                      d_du_dxy[1] * duv_dpz[0] + d_dv_dxy[1] * duv_dpz[1]};
    // dpdy = (org_dy + t * dir_dy) + dtdy * ray.dir
    d_ray_differential.org_dy += d_dpdy;
    auto d_t = sum(d_dpdy * dir_dy);
    d_ray_differential.dir_dy += d_dpdy * t;
    auto d_dtdy = sum(d_dpdy * ray.dir);
    d_ray.dir += d_dpdy * dtdy;
    // dtdy = -dot((org_dy + t * dir_dy), geom_normal) / dot(ray.dir, geom_normal)
    // -> dtdy = dtdy_numerator / dtdy_denominator
    auto d_dtdy_numerator = d_dtdy / dot(ray.dir, geom_normal);
    auto d_dtdy_denominator = -d_dtdy * dtdy / dot(ray.dir, geom_normal);
    // dtdy_numerator = -dot((org_dy + t * dir_dy), geom_normal)
    d_ray_differential.org_dy += (-d_dtdy_numerator * geom_normal);
    d_ray_differential.dir_dy += (-d_dtdy_numerator * geom_normal * t);
    d_t += -sum(d_dtdy_numerator * geom_normal * dir_dy);
    d_geom_normal += -d_dtdy_numerator * (org_dy + t * dir_dy);
    // dtdy_denominator = dot(ray.dir, geom_normal)
    d_ray.dir += d_dtdy_denominator * geom_normal;
    d_geom_normal += d_dtdy_denominator * ray.dir;

    // dpdx = (org_dx + t * dir_dx) + dtdx * ray.dir
    d_ray_differential.org_dx += d_dpdx;
    d_t += sum(d_dpdx * dir_dx);
    d_ray_differential.dir_dx += d_dpdx * t;
    auto d_dtdx = sum(d_dpdx * ray.dir);
    d_ray.dir += d_dpdx * dtdx;
    // dtdx = -dot((org_dx + t * dir_dx), geom_normal) / dot(ray.dir, geom_normal)
    // -> dtdx = dtdx_numerator / dtdx_denominator
    auto d_dtdx_numerator = d_dtdx / dot(ray.dir, geom_normal);
    auto d_dtdx_denominator = -d_dtdx * dtdx / dot(ray.dir, geom_normal);
    // dtdx_numerator = -dot((org_dx + t * dir_dx), geom_normal)
    d_ray_differential.org_dx += (-d_dtdx_numerator * geom_normal);
    d_ray_differential.dir_dx += (-d_dtdx_numerator * geom_normal * t);
    d_t += -sum(d_dtdx_numerator * geom_normal * dir_dx);
    d_geom_normal += -d_dtdx_numerator * (org_dx + t * dir_dx);
    // dtdx_denominator = dot(ray.dir, geom_normal)
    d_ray.dir += d_dtdx_denominator * geom_normal;
    d_geom_normal += d_dtdx_denominator * ray.dir;

    // duv_dpx = (1.f - dudp[0] - dvdp[0]) * uvs0 + dudp[0] * uvs1 + dvdp[0] * uvs2
    // duv_dpy = (1.f - dudp[1] - dvdp[1]) * uvs0 + dudp[1] * uvs1 + dvdp[1] * uvs2
    // duv_dpz = (1.f - dudp[2] - dvdp[2]) * uvs0 + dudp[2] * uvs1 + dvdp[2] * uvs2
    d_dudp[0] += sum(d_duv_dpx * (uvs1 - uvs0));
    d_dudp[1] += sum(d_duv_dpy * (uvs1 - uvs0));
    d_dudp[2] += sum(d_duv_dpz * (uvs1 - uvs0));
    d_dvdp[0] += sum(d_duv_dpx * (uvs2 - uvs0));
    d_dvdp[1] += sum(d_duv_dpy * (uvs2 - uvs0));
    d_dvdp[2] += sum(d_duv_dpz * (uvs2 - uvs0));
    auto d_v0 = Vector3{0, 0, 0};
    auto d_v1 = Vector3{0, 0, 0};
    auto d_v2 = Vector3{0, 0, 0};
    d_barycentric(v0, v1, v2, d_dudp, d_dvdp, d_v0, d_v1, d_v2);

    // geom_normal = normalize(unnormalized_geom_normal)
    auto d_unnormalized_geom_normal = d_normalize(unnormalized_geom_normal, d_geom_normal);
    // unnormalized_geom_normal = cross(v1 - v0, v2 - v0)
    auto d_v1_v0 = Vector3{0, 0, 0};
    auto d_v2_v0 = Vector3{0, 0, 0};
    d_cross(v1 - v0, v2 - v0, d_unnormalized_geom_normal, d_v1_v0, d_v2_v0);
    d_v0 += (- d_v1_v0 - d_v2_v0);
    d_v1 += d_v1_v0;
    d_v2 += d_v2_v0;
    // hit_pos = ray.org + ray.dir * t
    auto d_hit_pos = d_point.position;
    d_ray.org += d_hit_pos;
    d_ray.dir += d_hit_pos * t;
    d_t += sum(d_hit_pos * ray.dir);
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
