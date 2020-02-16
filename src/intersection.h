#pragma once

#include "redner.h"
#include "vector.h"
#include "frame.h"
#include "ray.h"

struct Intersection {
    Intersection(int shape_id = -1, int tri_id = -1)
        : shape_id(shape_id), tri_id(tri_id) {}

    int shape_id = -1;
    int tri_id = -1;

    DEVICE
    bool valid() const {
        return shape_id >= 0 && tri_id >= 0;
    }
};

template <typename T>
struct TSurfacePoint {
    TVector3<T> position;
    TVector3<T> geom_normal;
    TFrame<T> shading_frame;
    TVector3<T> dpdu;
    TVector2<T> uv;

    // Ray differential information
    TVector2<T> du_dxy, dv_dxy;
    TVector3<T> dn_dx, dn_dy;

    TVector3<T> color; // vertex color information

    DEVICE static TSurfacePoint<T> zero() {
        return TSurfacePoint<T>{
            TVector3<T>{0, 0, 0}, // position
            TVector3<T>{0, 0, 0}, // geom_normal
            TFrame<T>{TVector3<T>{0, 0, 0},
                      TVector3<T>{0, 0, 0},
                      TVector3<T>{0, 0, 0}}, // shading_frame
            TVector3<T>{0, 0, 0}, // dpdu
            TVector2<T>{0, 0}, // uv
            TVector2<T>{0, 0}, TVector2<T>{0, 0}, // du_dxy, dv_dxy
            TVector3<T>{0, 0, 0}, TVector3<T>{0, 0, 0}, // dn_dx, dn_dy
            TVector3<T>{0, 0, 0} // color
        };
    }
};

using SurfacePoint = TSurfacePoint<Real>;

template <typename T>
DEVICE
inline TVector3<T> intersect(const TVector3<T> &v0,
                             const TVector3<T> &v1,
                             const TVector3<T> &v2,
                             const TRay<T> &ray,
                             const TRayDifferential<T> &ray_differential,
                             TVector2<T> &u_dxy,
                             TVector2<T> &v_dxy,
                             TVector2<T> &t_dxy) {
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto pvec = cross(ray.dir, e2);
    auto pvec_dx = cross(ray_differential.dir_dx, e2);
    auto pvec_dy = cross(ray_differential.dir_dy, e2);
    auto divisor = dot(pvec, e1);
    auto divisor_dx = dot(pvec_dx, e1);
    auto divisor_dy = dot(pvec_dy, e1);
    if (fabs(divisor) < Real(1e-8f)) {
        // XXX HACK!!! XXX
        if (divisor > 0) {
            divisor = 1e-8f;
        } else {
            divisor = -1e-8f;
        }
    }
    auto s = ray.org - v0;
    auto s_dx = ray_differential.org_dx;
    auto s_dy = ray_differential.org_dy;
    auto dot_s_pvec = dot(s, pvec);
    auto dot_s_pvec_dx = dot(s_dx, pvec) + dot(s, pvec_dx);
    auto dot_s_pvec_dy = dot(s_dy, pvec) + dot(s, pvec_dy);
    auto u = dot_s_pvec / divisor;
    auto u_dx = (dot_s_pvec_dx * divisor - dot_s_pvec * divisor_dx) / square(divisor);
    auto u_dy = (dot_s_pvec_dy * divisor - dot_s_pvec * divisor_dy) / square(divisor);
    auto qvec = cross(s, e1);
    auto qvec_dx = cross(s_dx, e1);
    auto qvec_dy = cross(s_dy, e1);
    auto dot_dir_qvec = dot(ray.dir, qvec);
    auto dot_dir_qvec_dx = dot(ray_differential.dir_dx, qvec) + dot(ray.dir, qvec_dx);
    auto dot_dir_qvec_dy = dot(ray_differential.dir_dy, qvec) + dot(ray.dir, qvec_dy);
    auto v = dot_dir_qvec / divisor;
    auto v_dx = (dot_dir_qvec_dx * divisor - dot_dir_qvec * divisor_dx) / square(divisor);
    auto v_dy = (dot_dir_qvec_dy * divisor - dot_dir_qvec * divisor_dy) / square(divisor);
    auto dot_e2_qvec = dot(e2, qvec);
    auto dot_e2_qvec_dx = dot(e2, qvec_dx);
    auto dot_e2_qvec_dy = dot(e2, qvec_dy);
    auto t = dot_e2_qvec / divisor;
    auto t_dx = (dot_e2_qvec_dx * divisor - dot_e2_qvec * divisor_dx) / square(divisor);
    auto t_dy = (dot_e2_qvec_dy * divisor - dot_e2_qvec * divisor_dy) / square(divisor);
    u_dxy = Vector2{u_dx, u_dy};
    v_dxy = Vector2{v_dx, v_dy};
    t_dxy = Vector2{t_dx, t_dy};
    return TVector3<T>{u, v, t};
}

template <typename T>
DEVICE
inline void d_intersect(const TVector3<T> &v0,
                        const TVector3<T> &v1,
                        const TVector3<T> &v2,
                        const TRay<T> &ray,
                        const TRayDifferential<T> &ray_differential,
                        const TVector3<T> &d_uvt,
                        const TVector2<T> &d_u_dxy,
                        const TVector2<T> &d_v_dxy,
                        const TVector2<T> &d_t_dxy,
                        TVector3<T> &d_v0,
                        TVector3<T> &d_v1,
                        TVector3<T> &d_v2,
                        DTRay<T> &d_ray,
                        TRayDifferential<T> &d_ray_differential) {
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto pvec = cross(ray.dir, e2);
    auto pvec_dx = cross(ray_differential.dir_dx, e2);
    auto pvec_dy = cross(ray_differential.dir_dy, e2);
    auto divisor = dot(pvec, e1);
    auto divisor_dx = dot(pvec_dx, e1);
    auto divisor_dy = dot(pvec_dy, e1);
    if (fabs(divisor) < Real(1e-8f)) {
        // XXX HACK!!! XXX
        if (divisor > 0) {
            divisor = 1e-8f;
        } else {
            divisor = -1e-8f;
        }
    }
    auto s = ray.org - v0;
    auto s_dx = ray_differential.org_dx;
    auto s_dy = ray_differential.org_dy;
    auto dot_s_pvec = dot(s, pvec);
    auto dot_s_pvec_dx = dot(s_dx, pvec) + dot(s, pvec_dx);
    auto dot_s_pvec_dy = dot(s_dy, pvec) + dot(s, pvec_dy);
    auto u = dot_s_pvec / divisor;
    // auto u_dx = (dot_s_pvec_dx * divisor - dot_s_pvec * divisor_dx) / square(divisor);
    // auto u_dy = (dot_s_pvec_dy * divisor - dot_s_pvec * divisor_dy) / square(divisor);
    auto qvec = cross(s, e1);
    auto qvec_dx = cross(s_dx, e1);
    auto qvec_dy = cross(s_dy, e1);
    auto dot_dir_qvec = dot(ray.dir, qvec);
    auto dot_dir_qvec_dx = dot(ray_differential.dir_dx, qvec) + dot(ray.dir, qvec_dx);
    auto dot_dir_qvec_dy = dot(ray_differential.dir_dy, qvec) + dot(ray.dir, qvec_dy);
    auto v = dot_dir_qvec / divisor;
    // auto v_dx = (dot_dir_qvec_dx * divisor - dot_dir_qvec * divisor_dx) / square(divisor);
    // auto v_dy = (dot_dir_qvec_dy * divisor - dot_dir_qvec * divisor_dy) / square(divisor);
    auto dot_e2_qvec = dot(e2, qvec);
    auto dot_e2_qvec_dx = dot(e2, qvec_dx);
    auto dot_e2_qvec_dy = dot(e2, qvec_dy);
    auto t = dot_e2_qvec / divisor;
    // auto t_dx = (dot_e2_qvec_dx * divisor - dot_e2_qvec * divisor_dx) / square(divisor);
    // auto t_dy = (dot_e2_qvec_dy * divisor - dot_e2_qvec * divisor_dy) / square(divisor);
    auto divisor_sq = square(divisor);
    auto divisor_cu = divisor_sq * divisor;

    // Backprop
    // t_dx = (dot_e2_qvec_dx * divisor - dot_e2_qvec * divisor_dx) / square(divisor)
    auto d_dot_e2_qvec_dx = d_t_dxy.x / divisor;
    auto d_divisor = -d_t_dxy.x * (dot_e2_qvec_dx / divisor_sq -
                     (2 * dot_e2_qvec * divisor_dx / divisor_cu));
    auto d_dot_e2_qvec = -d_t_dxy.x * divisor_dx / divisor_sq;
    auto d_divisor_dx = -d_t_dxy.x * dot_e2_qvec / divisor_sq;
    // t_dy = (dot_e2_qvec_dy * divisor - dot_e2_qvec * divisor_dy) / square(divisor)
    auto d_dot_e2_qvec_dy = d_t_dxy.y / divisor;
    d_divisor += (-d_t_dxy.y * (dot_e2_qvec_dy / divisor_sq -
                  (2 * dot_e2_qvec * divisor_dy / divisor_cu)));
    d_dot_e2_qvec += (-d_t_dxy.y * divisor_dy / divisor_sq);
    auto d_divisor_dy = -d_t_dxy.y * dot_e2_qvec / divisor_sq;
    // t = dot_e2_qvec / divisor
    d_dot_e2_qvec += d_uvt[2] / divisor;
    d_divisor += (-d_uvt[2] * t / divisor);
    // dot_e2_qvec_dx = dot(e2, qvec_dx)
    auto d_e2 = d_dot_e2_qvec_dx * qvec_dx;
    auto d_qvec_dx = d_dot_e2_qvec_dx * e2;
    // dot_e2_qvec_dy = dot(e2, qvec_dy)
    d_e2 += d_dot_e2_qvec_dy * qvec_dy;
    auto d_qvec_dy = d_dot_e2_qvec_dy * e2;
    // dot_e2_qvec = dot(e2, qvec)
    d_e2 += d_dot_e2_qvec * qvec;
    auto d_qvec = d_dot_e2_qvec * e2;
    // v_dx = (dot_dir_qvec_dx * divisor - dot_dir_qvec * divisor_dx) / square(divisor)
    auto d_dot_dir_qvec_dx = d_v_dxy.x / divisor;
    d_divisor += (-d_v_dxy.x * (dot_dir_qvec_dx / divisor_sq -
                  (2 * dot_dir_qvec * divisor_dx) / divisor_cu));
    auto d_dot_dir_qvec = -d_v_dxy.x * divisor_dx / divisor_sq;
    d_divisor_dx += (-d_v_dxy.x * dot_dir_qvec / divisor_sq);
    // v_dy = (dot_dir_qvec_dy * divisor - dot_dir_qvec * divisor_dy) / square(divisor)
    auto d_dot_dir_qvec_dy = d_v_dxy.y / divisor;
    d_divisor += (-d_v_dxy.y * (dot_dir_qvec_dy / divisor_sq -
                  (2 * dot_dir_qvec * divisor_dy) / divisor_cu));
    d_dot_dir_qvec += (-d_v_dxy.y * divisor_dy / divisor_sq);
    d_divisor_dy += (-d_v_dxy.y * dot_dir_qvec / divisor_sq);
    // v = dot_dir_qvec / divisor
    d_dot_dir_qvec += d_uvt[1] / divisor;
    d_divisor -= d_uvt[1] * v / divisor;
    // dot_dir_qvec_dx = dot(ray_differential.dir_dx, qvec) + dot(ray.dir, qvec_dx)
    d_ray_differential.dir_dx += d_dot_dir_qvec_dx * qvec;
    d_qvec += d_dot_dir_qvec_dx * ray_differential.dir_dx;
    d_ray.dir += d_dot_dir_qvec_dx * qvec_dx;
    d_qvec_dx += d_dot_dir_qvec_dx * ray.dir;
    // dot_dir_qvec_dy = dot(ray_differential.dir_dy, qvec) + dot(ray.dir, qvec_dy)
    d_ray_differential.dir_dy += d_dot_dir_qvec_dy * qvec;
    d_qvec += d_dot_dir_qvec_dy * ray_differential.dir_dy;
    d_ray.dir += d_dot_dir_qvec_dy * qvec_dy;
    d_qvec_dy += d_dot_dir_qvec_dy * ray.dir;
    // dot_dir_qvec = dot(ray.dir, qvec)
    d_ray.dir += d_dot_dir_qvec * qvec;
    d_qvec += d_dot_dir_qvec * ray.dir;
    auto d_s = TVector3<T>{0, 0, 0};
    auto d_s_dx = TVector3<T>{0, 0, 0};
    auto d_s_dy = TVector3<T>{0, 0, 0};
    auto d_e1 = TVector3<T>{0, 0, 0};
    // qvec_dx = cross(s_dx, e1)
    d_cross(s_dx, e1, d_qvec_dx, d_s_dx, d_e1);
    // qvec_dy = cross(s_dy, e1)
    d_cross(s_dy, e1, d_qvec_dy, d_s_dy, d_e1);
    // qvec = cross(s, e1)
    d_cross(s, e1, d_qvec, d_s, d_e1);
    // u_dx = (dot_s_pvec_dx * divisor - dot_s_pvec * divisor_dx) / square(divisor)
    auto d_dot_s_pvec_dx = d_u_dxy.x / divisor;
    d_divisor += (-d_u_dxy.x * (dot_s_pvec_dx / divisor_sq -
                  (2 * dot_s_pvec * divisor_dx) / divisor_cu));
    auto d_dot_s_pvec = -d_u_dxy.x * divisor_dx / divisor_sq;
    d_divisor_dx += (-d_u_dxy.x * dot_s_pvec / divisor_sq);
    // u_dy = (dot_s_pvec_dy * divisor - dot_s_pvec * divisor_dy) / square(divisor)
    auto d_dot_s_pvec_dy = d_u_dxy.y / divisor;
    d_divisor += (-d_u_dxy.y * (dot_s_pvec_dy / divisor_sq -
                  (2 * dot_s_pvec * divisor_dy) / divisor_cu));
    d_dot_s_pvec += (-d_u_dxy.y * divisor_dy / divisor_sq);
    d_divisor_dy += (-d_u_dxy.y * dot_s_pvec / divisor_sq);
    // u = dot_s_pvec / divisor
    d_dot_s_pvec += d_uvt[0] / divisor;
    d_divisor -= d_uvt[0] * u / divisor;
    // dot_s_pvec_dx = dot(s_dx, pvec) + dot(s, pvec_dx)
    d_s_dx += d_dot_s_pvec_dx * pvec;
    auto d_pvec = d_dot_s_pvec_dx * s_dx;
    d_s += d_dot_s_pvec_dx * pvec_dx;
    auto d_pvec_dx = d_dot_s_pvec_dx * s;
    // dot_s_pvec_dy = dot(s_dy, pvec) + dot(s, pvec_dy)
    d_s_dy += d_dot_s_pvec_dy * pvec;
    d_pvec += d_dot_s_pvec_dy * s_dy;
    d_s += d_dot_s_pvec_dy * pvec_dy;
    auto d_pvec_dy = d_dot_s_pvec_dy * s;
    // dot_s_pvec = dot(s, pvec)
    d_s += d_dot_s_pvec * pvec;
    d_pvec += d_dot_s_pvec * s;
    // s_dx = ray_differential.org_dx
    d_ray_differential.org_dx += d_s_dx;
    // s_dy = ray_differential.org_dy
    d_ray_differential.org_dy += d_s_dy;
    // s = ray.org - v0
    d_ray.org += d_s;
    d_v0 -= d_s;
    // divisor_dx = dot(pvec_dx, e1)
    d_pvec_dx += d_divisor_dx * e1;
    d_e1 += d_divisor_dx * pvec_dx;
    // divisor_dy = dot(pvec_dy, e1)
    d_pvec_dy += d_divisor_dy * e1;
    d_e1 += d_divisor_dy * pvec_dy;
    // divisor = dot(pvec, e1)
    d_pvec += d_divisor * e1;
    d_e1 += d_divisor * pvec;
    // pvec_dx = cross(ray_differential.dir_dx, e2)
    d_cross(ray_differential.dir_dx, e2, d_pvec_dx, d_ray_differential.dir_dx, d_e2);
    // pvec_dy = cross(ray_differential.dir_dy, e2)
    d_cross(ray_differential.dir_dy, e2, d_pvec_dy, d_ray_differential.dir_dy, d_e2);
    // pvec = cross(ray.dir, e2)
    d_cross(ray.dir, e2, d_pvec, d_ray.dir, d_e2);
    // e2 = v2 - v0
    d_v2 += d_e2;
    d_v0 -= d_e2;
    // e1 = v1 - v0
    d_v1 += d_e1;
    d_v0 -= d_e1;
}

