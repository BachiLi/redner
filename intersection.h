#pragma once

#include "redner.h"
#include "vector.h"
#include "frame.h"
#include "ray.h"

struct Intersection {
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
    TVector2<T> uv;

    DEVICE static TSurfacePoint<T> zero() {
        return TSurfacePoint<T>{
            TVector3<T>{0, 0, 0},
            TVector3<T>{0, 0, 0},
            TFrame<T>{TVector3<T>{0, 0, 0},
                      TVector3<T>{0, 0, 0},
                      TVector3<T>{0, 0, 0}},
            TVector2<T>{0, 0}
        };
    }
};

using SurfacePoint = TSurfacePoint<Real>;

template <typename T>
DEVICE
inline TVector3<T> intersect(const TVector3<T> &v0,
                             const TVector3<T> &v1,
                             const TVector3<T> &v2,
                             const TRay<T> &ray) {
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto pvec = cross(ray.dir, e2);
    auto divisor = dot(pvec, e1);
    if (divisor == 0.f) {
        // XXX HACK!!! XXX
        divisor = 1e-8f;
    }
    auto s = ray.org - v0;
    auto u = dot(s, pvec) / divisor;
    auto qvec = cross(s, e1);
    auto v = dot(ray.dir, qvec) / divisor;
    auto t = dot(e2, qvec) / divisor;
    return TVector3<T>{u, v, t};
}

template <typename T>
DEVICE
inline void d_intersect(const TVector3<T> &v0,
                        const TVector3<T> &v1,
                        const TVector3<T> &v2,
                        const TRay<T> &ray,
                        const TVector3<T> &d_uvt,
                        TVector3<T> &d_v0,
                        TVector3<T> &d_v1,
                        TVector3<T> &d_v2,
                        DTRay<T> &d_ray) {
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    auto pvec = cross(ray.dir, e2);
    auto divisor = dot(pvec, e1);
    if (divisor == 0.f) {
        // XXX HACK!!! XXX
        divisor = 1e-8f;
    }
    auto s = ray.org - v0;
    auto u = dot(s, pvec) / divisor;
    auto qvec = cross(s, e1);
    auto v = dot(ray.dir, qvec) / divisor;
    auto t = dot(e2, qvec) / divisor;

    // t = dot(e2, qvec) / divisor
    auto d_dot_e2_qvec = d_uvt[2] / divisor;
    auto d_divisor = -d_uvt[2] * t / divisor;
    // dot(e2, qvec)
    auto d_e2 = d_dot_e2_qvec * qvec;
    auto d_qvec = d_dot_e2_qvec * e2;
    // v = dot(ray.dir, qvec) / divisor
    auto d_dot_dir_qvec = d_uvt[1] / divisor;
    d_divisor -= d_uvt[1] * v / divisor;
    // dot(ray.dir, qvec)
    d_ray.dir += d_dot_dir_qvec * qvec;
    d_qvec += d_dot_dir_qvec * ray.dir;
    // qvec = cross(s, e1)
    auto d_s = TVector3<T>{0, 0, 0};
    auto d_e1 = TVector3<T>{0, 0, 0};
    d_cross(s, e1, d_qvec, d_s, d_e1);
    // u = dot(s, pvec) / divisor
    auto d_dot_s_pvec = d_uvt[0] / divisor;
    d_divisor -= d_uvt[0] * u / divisor;
    // dot(s, pvec)
    d_s += d_dot_s_pvec * pvec;
    auto d_pvec = d_dot_s_pvec * s;
    // s = ray.org - v0
    d_ray.org += d_s;
    d_v0 -= d_s;
    // divisor = dot(pvec, e1)
    d_pvec += d_divisor * e1;
    d_e1 += d_divisor * pvec;
    // pvec = cross(ray.dir, e2)
    d_cross(ray.dir, e2, d_pvec, d_ray.dir, d_e2);
    // e2 = v2 - v0
    d_v2 += d_e2;
    d_v0 -= d_e2;
    // e1 = v1 - v0
    d_v1 += d_e1;
    d_v0 -= d_e1;
}

template <typename T>
DEVICE
inline TSurfacePoint<T> operator-(const TSurfacePoint<T> &p) {
    return TSurfacePoint<T>{p.position, -p.geom_normal, -p.shading_frame, p.uv};
}
