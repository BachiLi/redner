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
          ptr<float> uvs, // optional
          ptr<float> normals, // optional
          ptr<int> uv_indices, // optional, overrides uv index access
          ptr<int> normal_indices, // optional, overrides normal index access
          ptr<float> colors, // optional, used when the material specifies so.
          int num_vertices,
          int num_uv_vertices,
          int num_normal_vertices,
          int num_triangles,
          int material_id,
          int light_id) :
        vertices(vertices.get()),
        indices(indices.get()),
        uvs(uvs.get()),
        normals(normals.get()),
        uv_indices(uv_indices.get()),
        normal_indices(normal_indices.get()),
        colors(colors.get()),
        num_vertices(num_vertices),
        num_uv_vertices(num_uv_vertices),
        num_normal_vertices(num_normal_vertices),
        num_triangles(num_triangles),
        material_id(material_id),
        light_id(light_id) {}

    inline bool has_uvs() const {
        return uvs != nullptr;
    }

    inline bool has_normals() const {
        return normals != nullptr;
    }

    inline bool has_colors() const {
        return colors != nullptr;
    }

    float *vertices;
    int *indices;
    float *uvs;
    float *normals;
    int *uv_indices;
    int *normal_indices;
    float *colors;
    int num_vertices;
    int num_uv_vertices;
    int num_normal_vertices;
    int num_triangles;
    int material_id;
    int light_id;
};

struct DShape {
    DShape() {}
    DShape(ptr<float> vertices,
           ptr<float> uvs,
           ptr<float> normals,
           ptr<float> colors)
        : vertices(vertices.get()),
          uvs(uvs.get()),
          normals(normals.get()),
          colors(colors.get()) {}

    float *vertices;
    float *uvs;
    float *normals;
    float *colors;
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
inline Vector3i get_uv_indices(const Shape &shape, int index) {
    return Vector3i{shape.uv_indices[3 * index + 0],
                    shape.uv_indices[3 * index + 1],
                    shape.uv_indices[3 * index + 2]};
}

DEVICE
inline Vector3i get_normal_indices(const Shape &shape, int index) {
    return Vector3i{shape.normal_indices[3 * index + 0],
                    shape.normal_indices[3 * index + 1],
                    shape.normal_indices[3 * index + 2]};
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
inline bool has_colors(const Shape &shape) {
    return shape.colors != nullptr;
}

DEVICE
inline Vector3f get_color(const Shape &shape, int index) {
    return Vector3f{shape.colors[3 * index + 0],
                    shape.colors[3 * index + 1],
                    shape.colors[3 * index + 2]};
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
                       const Real d_area, Vector3 d_v[3]) {
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
    d_v[0] -= (d_e1 + d_e2);
    d_v[1] += d_e1;
    d_v[2] += d_e2;
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
        Vector3{0, 0, 0}, // TODO: compute proper dpdu
        sample, // TODO: give true light source uv
        Vector2{0, 0}, // TODO: inherit derivatives from previous path vertex
        Vector2{0, 0},
        Vector3{0, 0, 0} // color
    }; 
}

DEVICE
inline void d_sample_shape(const Shape &shape, int index, const Vector2 &sample,
                           const SurfacePoint &d_point, Vector3 d_v[3]) {
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
    //     Vector2{0, 0},
    //     Vector3{0, 0, 0}};
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
    d_v[0] += d_v0;
    d_v[1] += d_v1;
    d_v[2] += d_v2;
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

    // Backprop
    // d_b0_d_p = (d22 * d01_dp - d12 * d02_dp) * inv_denom
    auto d_d22 = d_d_b0_d_p * d01_dp * inv_denom;
    auto d_d01_dp = d_d_b0_d_p * d22 * inv_denom;
    auto d_d12 = -d_d_b0_d_p * d02_dp * inv_denom;
    auto d_d02_dp = -d_d_b0_d_p * d12 * inv_denom;
    auto d_inv_denom = d_d_b0_d_p * (d22 * d01_dp - d12 * d02_dp);
    // d_b1_d_p = (d11 * d02_dp - d12 * d01_dp) * inv_denom
    auto d_d11 = d_d_b1_d_p * d02_dp * inv_denom;
    d_d02_dp += d_d_b1_d_p * d11 * inv_denom;
    d_d12 += (-d_d_b1_d_p * d01_dp * inv_denom);
    d_d01_dp += (-d_d_b1_d_p * d12 * inv_denom);
    d_inv_denom += d_d_b1_d_p * (d11 * d02_dp - d12 * d01_dp);
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
    auto uv_ind = ind;
    if (shape.uv_indices != nullptr) {
        uv_ind = get_uv_indices(shape, index);
    }
    auto normal_ind = ind;
    if (shape.normal_indices != nullptr) {
        normal_ind = get_normal_indices(shape, index);
    }
    Vector2 uvs0, uvs1, uvs2;
    if (has_uvs(shape)) {
        uvs0 = get_uv(shape, uv_ind[0]);
        uvs1 = get_uv(shape, uv_ind[1]);
        uvs2 = get_uv(shape, uv_ind[2]);
    } else {
        uvs0 = Vector2{0.f, 0.f};
        uvs1 = Vector2{1.f, 0.f};
        uvs2 = Vector2{1.f, 1.f};
    }
    auto u_dxy = Vector2{0, 0};
    auto v_dxy = Vector2{0, 0};
    auto t_dxy = Vector2{0, 0};
    auto uvt = intersect(v0, v1, v2, ray, ray_differential, u_dxy, v_dxy, t_dxy);
    auto u = uvt[0];
    auto v = uvt[1];
    auto w = 1.f - (u + v);
    auto t = uvt[2];
    auto uv = w * uvs0 + u * uvs1 + v * uvs2;
    auto hit_pos = ray.org + ray.dir * t;
    auto geom_normal = normalize(cross(v1 - v0, v2 - v0));

    // Compute triangle derivatives (for shading frame)
    auto uvs02 = uvs0 - uvs2;
    auto uvs12 = uvs1 - uvs2;
    auto uv_det = uvs02[0] * uvs12[1] - uvs02[1] * uvs12[0];
    auto dpdu = Vector3{0, 0, 0};
    auto dpdv = Vector3{0, 0, 0};
    if (uv_det == 0) {
        coordinate_system(geom_normal, dpdu, dpdv);
    } else {
        auto inv_det = 1 / uv_det;
        auto v02 = v0 - v2;
        auto v12 = v1 - v2;
        dpdu = ( uvs12[1] * v02 - uvs02[1] * v12) * inv_det;
        dpdv = (-uvs12[0] * v02 + uvs02[0] * v12) * inv_det;
    }

    // Surface derivative for ray differentials
    auto du_dxy = (- u_dxy - v_dxy) * uvs0[0] + u_dxy * uvs1[0] + v_dxy * uvs2[0];
    auto dv_dxy = (- u_dxy - v_dxy) * uvs0[1] + u_dxy * uvs1[1] + v_dxy * uvs2[1];
    auto dpdx = ray_differential.org_dx + ray.dir * t_dxy.x + ray_differential.dir_dx * t;
    auto dpdy = ray_differential.org_dy + ray.dir * t_dxy.y + ray_differential.dir_dy * t;
    auto shading_normal = geom_normal;
    auto dn_dx = Vector3{0, 0, 0};
    auto dn_dy = Vector3{0, 0, 0};
    if (has_shading_normals(shape)) {
        auto n0 = get_shading_normal(shape, normal_ind[0]);
        auto n1 = get_shading_normal(shape, normal_ind[1]);
        auto n2 = get_shading_normal(shape, normal_ind[2]);

        auto nn = w * n0 + u * n1 + v * n2;

        // Compute dndx & dndy
        auto dnn_dx = (- u_dxy.x - v_dxy.x) * n0 + u_dxy.x * n1 + v_dxy.x * n2;
        auto dnn_dy = (- u_dxy.y - v_dxy.y) * n0 + u_dxy.y * n1 + v_dxy.y * n2;
        // normalization derivatives
        auto nn_len_sq = dot(nn, nn);
        auto nn_len = sqrt(nn_len_sq);
        dn_dx = (nn_len_sq * dnn_dx - dot(nn, dnn_dx) * nn) / (nn_len_sq * nn_len);
        dn_dy = (nn_len_sq * dnn_dy - dot(nn, dnn_dy) * nn) / (nn_len_sq * nn_len);

        // Shading normal computation
        shading_normal = normalize(nn);
        // Flip geometric normal to the same side of shading normal
        if (dot(geom_normal, shading_normal) < 0.f) {
            geom_normal = -geom_normal;
        }
    }

    auto frame_x = normalize(dpdu);
    auto frame_y = cross(shading_normal, frame_x);
    if (length_squared(frame_y) > 0) {
        frame_y = normalize(frame_y);
        frame_x = cross(frame_y, shading_normal);
    } else {
        coordinate_system(shading_normal, frame_x, frame_y);
    }
    auto frame = Frame(frame_x, frame_y, shading_normal);

    // Update ray differential
    new_ray_differential.org_dx = dpdx;
    new_ray_differential.org_dy = dpdy;
    new_ray_differential.dir_dx = ray_differential.dir_dx;
    new_ray_differential.dir_dy = ray_differential.dir_dy;

    // Interpolate color
    auto cc = Vector3{0, 0, 0};
    if (has_colors(shape)) {
        auto c0 = get_color(shape, ind[0]);
        auto c1 = get_color(shape, ind[1]);
        auto c2 = get_color(shape, ind[2]);
        cc = w * c0 + u * c1 + v * c2;
    }

    return SurfacePoint{hit_pos,
                        geom_normal,
                        frame,
                        dpdu,
                        uv,
                        du_dxy,
                        dv_dxy,
                        dn_dx,
                        dn_dy,
                        cc};
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
        Vector3 d_v_p[3],
        Vector3 d_v_n[3],
        Vector2 d_v_uv[3],
        Vector3 d_v_c[3]) {
    auto ind = get_indices(shape, index);
    auto v0 = Vector3{get_vertex(shape, ind[0])};
    auto v1 = Vector3{get_vertex(shape, ind[1])};
    auto v2 = Vector3{get_vertex(shape, ind[2])};
    auto uv_ind = ind;
    if (shape.uv_indices != nullptr) {
        uv_ind = get_uv_indices(shape, index);
    }
    auto normal_ind = ind;
    if (shape.normal_indices != nullptr) {
        normal_ind = get_normal_indices(shape, index);
    }
    Vector2 uvs0, uvs1, uvs2;
    if (has_uvs(shape)) {
        uvs0 = get_uv(shape, uv_ind[0]);
        uvs1 = get_uv(shape, uv_ind[1]);
        uvs2 = get_uv(shape, uv_ind[2]);
    } else {
        uvs0 = Vector2{0.f, 0.f};
        uvs1 = Vector2{1.f, 0.f};
        uvs2 = Vector2{1.f, 1.f};
    }
    auto u_dxy = Vector2{0, 0};
    auto v_dxy = Vector2{0, 0};
    auto t_dxy = Vector2{0, 0};
    auto uvt = intersect(v0, v1, v2, ray, ray_differential, u_dxy, v_dxy, t_dxy);
    auto u = uvt[0];
    auto v = uvt[1];
    auto w = 1.f - (u + v);
    auto t = uvt[2];
    // uv = w * uvs0 + u * uvs1 + v * uvs2
    // hit_pos = ray.org + ray.dir * t
    auto unnormalized_geom_normal = cross(v1 - v0, v2 - v0);
    auto geom_normal = normalize(unnormalized_geom_normal);

    // Compute triangle derivatives (for shading frame)
    auto uvs02 = uvs0 - uvs2;
    auto uvs12 = uvs1 - uvs2;
    auto uv_det = uvs02[0] * uvs12[1] - uvs02[1] * uvs12[0];
    auto dpdu = Vector3{0, 0, 0};
    auto dpdv = Vector3{0, 0, 0};
    if (uv_det == 0) {
        coordinate_system(geom_normal, dpdu, dpdv);
    } else {
        auto inv_det = 1 / uv_det;
        auto v02 = v0 - v2;
        auto v12 = v1 - v2;
        dpdu = ( uvs12[1] * v02 - uvs02[1] * v12) * inv_det;
        dpdv = (-uvs12[0] * v02 + uvs02[0] * v12) * inv_det;
    }

    // Surface derivative for ray differentials
    // du_dxy = (- u_dxy - v_dxy) * uvs0[0] + u_dxy * uvs1[0] + v_dxy * uvs2[0]
    // dv_dxy = (- u_dxy - v_dxy) * uvs0[1] + u_dxy * uvs1[1] + v_dxy * uvs2[1]
    // dpdx = ray_differential.org_dx + ray.dir * t_dxy.x + ray_differential.dir_dx * t
    // dpdy = ray_differential.org_dy + ray.dir * t_dxy.y + ray_differential.dir_dy * t

    auto shading_normal = geom_normal;
    auto geom_normal_flipped = false;
    auto dn_dx = Vector3{0, 0, 0};
    auto dn_dy = Vector3{0, 0, 0};
    if (has_shading_normals(shape)) {
        auto n0 = get_shading_normal(shape, normal_ind[0]);
        auto n1 = get_shading_normal(shape, normal_ind[1]);
        auto n2 = get_shading_normal(shape, normal_ind[2]);
        auto nn = w * n0 + u * n1 + v * n2;

        // Compute dndx & dndy
        auto dnn_dx = (- u_dxy.x - v_dxy.x) * n0 + u_dxy.x * n1 + v_dxy.x * n2;
        auto dnn_dy = (- u_dxy.y - v_dxy.y) * n0 + u_dxy.y * n1 + v_dxy.y * n2;
        // normalization derivatives
        auto nn_len_sq = dot(nn, nn);
        auto nn_len = sqrt(nn_len_sq);
        dn_dx = (nn_len_sq * dnn_dx - dot(nn, dnn_dx) * nn) / (nn_len_sq * nn_len);
        dn_dy = (nn_len_sq * dnn_dy - dot(nn, dnn_dy) * nn) / (nn_len_sq * nn_len);

        // Shading normal computation
        shading_normal = normalize(nn);
        // Flip geometric normal to the same side of shading normal
        if (dot(geom_normal, shading_normal) < 0.f) {
            geom_normal = -geom_normal;
            geom_normal_flipped = true;
        }
    }

    auto frame_x_org = normalize(dpdu);
    auto frame_y_org = cross(shading_normal, frame_x_org);
    auto frame_y_org_not_degenerated = length_squared(frame_y_org) > 0;
    auto frame_x = Vector3{0, 0, 0};
    auto frame_y = Vector3{0, 0, 0};
    if (frame_y_org_not_degenerated) {
        frame_y = normalize(frame_y_org);
        frame_x = cross(frame_y, shading_normal);
    } else {
        coordinate_system(shading_normal, frame_x, frame_y);
    }
    // auto frame = Frame(frame_x, frame_y, shading_normal);

    // Interpolate color
    auto cc = Vector3{0, 0, 0};
    if (has_colors(shape)) {
        auto c0 = get_color(shape, ind[0]);
        auto c1 = get_color(shape, ind[1]);
        auto c2 = get_color(shape, ind[2]);
        cc = w * c0 + u * c1 + v * c2;
    }

    // point = SurfacePoint{hit_pos,
    //                      geom_normal,
    //                      frame,
    //                      uv,
    //                      du_dxy,
    //                      dv_dxy,
    //                      dn_dx,
    //                      dn_dy,
    //                      cc}

    // Backprop
    auto d_u = Real(0), d_v = Real(0), d_w = Real(0);

    if (has_colors(shape)) {
        auto c0 = get_color(shape, ind[0]);
        auto c1 = get_color(shape, ind[1]);
        auto c2 = get_color(shape, ind[2]);
        d_v_c[0] += d_point.color * w;
        d_v_c[1] += d_point.color * u;
        d_v_c[2] += d_point.color * v;
        d_w += sum(d_point.color * c0);
        d_u += sum(d_point.color * c1);
        d_v += sum(d_point.color * c2);
    }

    auto d_frame_x = d_point.shading_frame[0];
    auto d_frame_y = d_point.shading_frame[1];
    auto d_shading_normal = d_point.shading_frame[2];
    auto d_dpdu = d_point.dpdu;
    if (frame_y_org_not_degenerated) {
        // frame_y = normalize(frame_y_org);
        // frame_x = cross(frame_y, shading_normal);
        d_cross(frame_y, shading_normal, d_frame_x, d_frame_y, d_shading_normal);
        auto d_frame_y_org = d_normalize(frame_y_org, d_frame_y);
        // frame_x_org = normalize(dpdu)
        // frame_y_org = cross(shading_normal, frame_x_org)
        auto d_frame_x_org = Vector3{0, 0, 0};
        d_cross(shading_normal, frame_x_org, d_frame_y_org, d_shading_normal, d_frame_x_org);
        d_dpdu = d_normalize(dpdu, d_frame_x_org);
    } else {
        d_coordinate_system(shading_normal, d_frame_x, d_frame_y, d_shading_normal);
    }

    auto d_geom_normal = d_point.geom_normal;
    // new_ray_differential.org_dx = dpdx;
    // new_ray_differential.org_dy = dpdy;
    // new_ray_differential.dir_dx = dir_dx;
    // new_ray_differential.dir_dy = dir_dy;
    auto d_dpdx = d_new_ray_differential.org_dx;
    auto d_dpdy = d_new_ray_differential.org_dy;
    d_ray_differential.dir_dx += d_new_ray_differential.dir_dx;
    d_ray_differential.dir_dy += d_new_ray_differential.dir_dy;
    auto d_u_dxy = Vector2{0, 0};
    auto d_v_dxy = Vector2{0, 0};
    auto d_v0 = Vector3{0, 0, 0};
    auto d_v1 = Vector3{0, 0, 0};
    auto d_v2 = Vector3{0, 0, 0};
    if (has_shading_normals(shape)) {
        if (geom_normal_flipped) {
            d_geom_normal = -d_geom_normal;
        }
        auto n0 = get_shading_normal(shape, normal_ind[0]);
        auto n1 = get_shading_normal(shape, normal_ind[1]);
        auto n2 = get_shading_normal(shape, normal_ind[2]);
        auto d_shading_normal = d_point.shading_frame[2];
        // differentiate through frame construction
        d_coordinate_system(shading_normal, d_point.shading_frame[0], d_point.shading_frame[1],
                            d_shading_normal);

        auto nn = w * n0 + u * n1 + v * n2;

        auto dnn_dx = (- u_dxy.x - v_dxy.x) * n0 + u_dxy.x * n1 + v_dxy.x * n2;
        auto dnn_dy = (- u_dxy.y - v_dxy.y) * n0 + u_dxy.y * n1 + v_dxy.y * n2;
        // normalization derivatives
        auto nn_len_sq = dot(nn, nn);
        auto nn_len = sqrt(nn_len_sq);
        // dn_dx = (nn_len_sq * dnn_dx - dot(nn, dnn_dx) * nn) / (nn_len_sq * nn_len)
        // dn_dy = (nn_len_sq * dnn_dy - dot(nn, dnn_dy) * nn) / (nn_len_sq * nn_len)

        if (nn_len_sq > 0) { // <= 0 means degenerate normal
            // shading_normal = normalize(nn)
            auto d_nn = d_normalize(nn, d_shading_normal);
            
            // dn_dx = (nn_len_sq * dnn_dx - dot(nn, dnn_dx) * nn) / nn_denom
            // dn_dy = (nn_len_sq * dnn_dy - dot(nn, dnn_dy) * nn) / nn_denom
            auto nn_denom = (nn_len_sq * nn_len);
            auto d_dn_dx = d_point.dn_dx;
            auto d_dn_dy = d_point.dn_dy;
            auto d_nn_len_sq = (d_dn_dx * dnn_dx + d_dn_dy * dnn_dy) / nn_denom;
            auto d_dnn_dx = d_dn_dx * nn_len_sq / nn_denom;
            auto d_dnn_dy = d_dn_dy * nn_len_sq / nn_denom;
            auto d_dot_nn_dnn_dx = sum(d_dn_dx * nn) / nn_denom;
            auto d_dot_nn_dnn_dy = sum(d_dn_dy * nn) / nn_denom;
            d_nn += (d_dn_dx * dot(nn, dnn_dx) + d_dn_dy * dot(nn, dnn_dy)) / nn_denom;
            auto d_nn_denom = (d_dn_dx * (-dn_dx) + d_dn_dy * (-dn_dy)) / nn_denom;
            // dot(nn, dnn_dx) & dot(nn, dnn_dy)
            d_nn += d_dot_nn_dnn_dx * dnn_dx + d_dot_nn_dnn_dy * dnn_dy;
            d_dnn_dx += d_dot_nn_dnn_dx * nn;
            d_dnn_dy += d_dot_nn_dnn_dy * nn;

            // nn_denom = pow(nn_len_sq, Real(3.0/2.0))
            d_nn_len_sq += d_nn_denom * nn_len * Real(3.0 / 2.0);
            // nn_len_sq = dot(nn, nn)
            d_nn += 2 * d_nn_len_sq * nn;

            // dnn_dx = (- u_dxy.x - v_dxy.x) * n0 + u_dxy.x * n1 + v_dxy.x * n2
            // dnn_dy = (- u_dxy.y - v_dxy.y) * n0 + u_dxy.y * n1 + v_dxy.y * n2
            d_u_dxy.x += sum(d_dnn_dx * (n1 - n0));
            d_u_dxy.y += sum(d_dnn_dy * (n1 - n0));
            d_v_dxy.x += sum(d_dnn_dx * (n2 - n0));
            d_v_dxy.y += sum(d_dnn_dy * (n2 - n0));
            auto d_n0 = d_dnn_dx * (- u_dxy.x - v_dxy.x) +
                        d_dnn_dy * (- u_dxy.y - v_dxy.y);
            auto d_n1 = d_dnn_dx * u_dxy.x + d_dnn_dy * u_dxy.y;
            auto d_n2 = d_dnn_dx * v_dxy.x + d_dnn_dy * v_dxy.y;

            // nn = w * n0 + u * n1 + v * n2
            d_w += sum(d_nn * n0);
            d_u += sum(d_nn * n1);
            d_v += sum(d_nn * n2);
            d_n0 += d_nn * w;
            d_n1 += d_nn * u;
            d_n2 += d_nn * v;
            d_v_n[0] += d_n0;
            d_v_n[1] += d_n1;
            d_v_n[2] += d_n2;
        }
    } else {
        d_geom_normal += d_point.shading_frame[2];
        d_coordinate_system(shading_normal, d_point.shading_frame[0], d_point.shading_frame[1],
                            d_geom_normal);
    }

    // dpdx = ray_differential.org_dx + ray.dir * t_dxy.x + ray_differential.dir_dx * t
    // dpdy = ray_differential.org_dy + ray.dir * t_dxy.y + ray_differential.dir_dy * t
    auto d_t_dxy = Vector2{0, 0};
    d_ray_differential.org_dx += d_dpdx;
    d_ray.dir += d_dpdx * t_dxy.x;
    d_t_dxy.x += sum(d_dpdx * ray.dir);
    d_ray_differential.dir_dx += d_dpdx * t;
    auto d_t = sum(d_dpdx * ray_differential.dir_dx);
    d_ray_differential.org_dy += d_dpdy;
    d_ray.dir += d_dpdy * t_dxy.y;
    d_t_dxy.y += sum(d_dpdy * ray.dir);
    d_ray_differential.dir_dy += d_dpdy * t;
    d_t += sum(d_dpdy * ray_differential.dir_dy);

    // Partial derivatives
    auto d_uvs0 = Vector2{0, 0};
    auto d_uvs1 = Vector2{0, 0};
    auto d_uvs2 = Vector2{0, 0};
    if (uv_det == 0) {
        // coordinate_system(geom_normal, dpdu, dpdv)
        d_coordinate_system(geom_normal, d_dpdu, Vector3{0, 0, 0}, d_geom_normal);
    } else {
        // dpdu = ( uvs12[1] * v02 - uvs02[1] * v12) * inv_det
        auto inv_det = 1 / uv_det;
        auto v02 = v0 - v2;
        auto v12 = v1 - v2;
        auto d_uvs02 = Vector2{0, 0};
        auto d_uvs12 = Vector2{0, 0};
        d_uvs12[1] += sum(d_dpdu * v02) * inv_det;
        auto d_v02 = d_dpdu * uvs12[1] * inv_det;
        d_uvs02[1] += sum(d_dpdu * v12) * inv_det;
        auto d_v12 = d_dpdu * uvs02[1] * inv_det;
        auto d_inv_det = sum(d_dpdu * (uvs12[1] * v02 - uvs02[1] * v12));
        // inv_det = 1 / uv_det
        auto d_uv_det = -d_inv_det * inv_det * inv_det;
        // uv_det = uvs02[0] * uvs12[1] - uvs02[1] * uvs12[0]
        d_uvs02[0] += d_uv_det * uvs12[1];
        d_uvs12[1] += d_uv_det * uvs02[0];
        d_uvs02[1] -= d_uv_det * uvs12[0];
        d_uvs12[0] -= d_uv_det * uvs02[1];
        // uvs02 = uvs0 - uvs2
        // uvs12 = uvs1 - uvs2
        d_uvs0 += d_uvs02;
        d_uvs1 += d_uvs12;
        d_uvs2 -= (d_uvs02 + d_uvs12);
        // v02 = v0 - v2
        // v12 = v1 - v2
        d_v0 += d_v02;
        d_v1 += d_v12;
        d_v2 -= (d_v02 + d_v12);
    }
    // du_dxy = (- u_dxy - v_dxy) * uvs0[0] + u_dxy * uvs1[0] + v_dxy * uvs2[0]
    // dv_dxy = (- u_dxy - v_dxy) * uvs0[1] + u_dxy * uvs1[1] + v_dxy * uvs2[1]
    auto d_du_dxy = d_point.du_dxy;
    auto d_dv_dxy = d_point.dv_dxy;
    d_u_dxy += d_du_dxy * (uvs1[0] - uvs0[0]) + d_dv_dxy * (uvs1[1] - uvs0[1]);
    d_v_dxy += d_du_dxy * (uvs2[0] - uvs0[0]) + d_dv_dxy * (uvs2[1] - uvs0[1]);
    d_uvs0[0] += sum(d_du_dxy * (- u_dxy - v_dxy));
    d_uvs0[1] += sum(d_dv_dxy * (- u_dxy - v_dxy));
    d_uvs1[0] += sum(d_du_dxy * u_dxy);
    d_uvs1[1] += sum(d_dv_dxy * u_dxy);
    d_uvs2[0] += sum(d_du_dxy * v_dxy);
    d_uvs2[1] += sum(d_dv_dxy * v_dxy);

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
    d_uvs0 += d_uv * w;
    d_uvs1 += d_uv * u;
    d_uvs2 += d_uv * v;
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
    // uvt = intersect(v0, v1, v2, ray, ray_differential, u_dxy, v_dxy, t_dxy)
    d_intersect(v0, v1, v2, ray, ray_differential,
        d_uvt, d_u_dxy, d_v_dxy, d_t_dxy, d_v0, d_v1, d_v2, d_ray, d_ray_differential);
    if (has_uvs(shape)) {
        d_v_uv[0] += d_uvs0;
        d_v_uv[1] += d_uvs1;
        d_v_uv[2] += d_uvs2;
    }
    d_v_p[0] += d_v0;
    d_v_p[1] += d_v1;
    d_v_p[2] += d_v2;
}

void test_d_intersect();
void test_d_sample_shape();
