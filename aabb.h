#pragma once

#include "redner.h"
#include "vector.h"
#include "cuda_utils.h"
#include "ray.h"

#include <iostream>

struct AABB3 {
    DEVICE AABB3(
        const Vector3 &p_min = Vector3{ infinity<Real>(),  infinity<Real>(),  infinity<Real>()},
        const Vector3 &p_max = Vector3{-infinity<Real>(), -infinity<Real>(), -infinity<Real>()})
            : p_min(p_min), p_max(p_max) {}

    DEVICE
    const Vector3& operator[](int i) const {
        return i == 0 ? p_min : p_max;
    }

    Vector3 p_min;
    Vector3 p_max;
};

struct AABB6 {
    DEVICE AABB6(
        const Vector3 &p_min = Vector3{ infinity<Real>(),  infinity<Real>(),  infinity<Real>()},
        const Vector3 &d_min = Vector3{ infinity<Real>(),  infinity<Real>(),  infinity<Real>()},
        const Vector3 &p_max = Vector3{-infinity<Real>(), -infinity<Real>(), -infinity<Real>()},
        const Vector3 &d_max = Vector3{-infinity<Real>(), -infinity<Real>(), -infinity<Real>()})
            : p_min(p_min), d_min(d_min), p_max(p_max), d_max(d_max) {}

    Vector3 p_min, d_min;
    Vector3 p_max, d_max;
};

template<typename T>
DEVICE
inline T convert_aabb(const AABB6 &b) {
    assert(false);
}

template<>
DEVICE
inline AABB3 convert_aabb(const AABB6 &b) {
    return AABB3{b.p_min, b.p_max};
}

template<>
DEVICE
inline AABB6 convert_aabb(const AABB6 &b) {
    return b;
}

struct Sphere {
    Vector3 center;
    Real radius;
};

DEVICE
inline Vector3 corner(const AABB3 &b, int i) {
    Vector3 ret;
    ret[0] = ((i & 1) == 0) ? b.p_min[0] : b.p_max[0];
    ret[1] = ((i & 2) == 0) ? b.p_min[1] : b.p_max[1];
    ret[2] = ((i & 4) == 0) ? b.p_min[2] : b.p_max[2];
    return ret;
}

DEVICE
inline AABB3 merge(const AABB3 &b, const Vector3 &p) {
    return AABB3{
        Vector3{
            min(b.p_min[0], p[0]),
            min(b.p_min[1], p[1]),
            min(b.p_min[2], p[2])},
        Vector3{
            max(b.p_max[0], p[0]),
            max(b.p_max[1], p[1]),
            max(b.p_max[2], p[2])}};
}

DEVICE
inline AABB3 merge(const AABB3 &b0, const AABB3 &b1) {
    return AABB3{
        Vector3{
            min(b0.p_min[0], b1.p_min[0]),
            min(b0.p_min[1], b1.p_min[1]),
            min(b0.p_min[2], b1.p_min[2])},
        Vector3{
            max(b0.p_max[0], b1.p_max[0]),
            max(b0.p_max[1], b1.p_max[1]),
            max(b0.p_max[2], b1.p_max[2])}};
}

DEVICE
inline AABB6 merge(const AABB6 &b0, const AABB6 &b1) {
    return AABB6{
        Vector3{
            min(b0.p_min[0], b1.p_min[0]),
            min(b0.p_min[1], b1.p_min[1]),
            min(b0.p_min[2], b1.p_min[2])},
        Vector3{
            min(b0.d_min[0], b1.d_min[0]),
            min(b0.d_min[1], b1.d_min[1]),
            min(b0.d_min[2], b1.d_min[2])},
        Vector3{
            max(b0.p_max[0], b1.p_max[0]),
            max(b0.p_max[1], b1.p_max[1]),
            max(b0.p_max[2], b1.p_max[2])},
        Vector3{
            max(b0.d_max[0], b1.d_max[0]),
            max(b0.d_max[1], b1.d_max[1]),
            max(b0.d_max[2], b1.d_max[2])}};
}

DEVICE
inline Sphere compute_bounding_sphere(const AABB3 &b) {
    auto c = 0.5f * (b.p_max + b.p_min);
    auto r = distance(c, b.p_max);
    return Sphere{c, r};
}

DEVICE
inline Sphere compute_bounding_sphere(const AABB6 &b) {
    auto c = 0.5f * (b.p_max + b.p_min);
    auto r = distance(c, b.p_max);
    return Sphere{c, r};
}

DEVICE
inline bool inside(const AABB3 &b, const Vector3 &p) {
    return p.x >= b.p_min.x && p.x <= b.p_max.x &&
           p.y >= b.p_min.y && p.y <= b.p_max.y &&
           p.z >= b.p_min.z && p.z <= b.p_max.z;
}

DEVICE
inline bool inside(const AABB6 &b, const Vector3 &p) {
    return p.x >= b.p_min.x && p.x <= b.p_max.x &&
           p.y >= b.p_min.y && p.y <= b.p_max.y &&
           p.z >= b.p_min.z && p.z <= b.p_max.z;
}

DEVICE
inline bool inside(const Sphere &b, const Vector3 &p) {
    return distance(p, b.center) <= b.radius;
}

DEVICE
inline Vector3 center(const AABB3 &b) {
    return 0.5f * (b.p_max + b.p_min);
}


DEVICE
inline bool intersect(const Sphere &s, const AABB3 &b) {
    // "A Simple Method for Box-Sphere Intersection Testing", Jim Arvo
    // https://github.com/erich666/GraphicsGems/blob/master/gems/BoxSphere.c
    auto d_min = Real(0);
    auto r2 = square(s.radius);
    for(int i = 0; i < 3; i++) {
        if (s.center[i] < b.p_min[i]) {
            d_min += square(s.center[i] - b.p_min[i]);
        } else if (s.center[i] > b.p_max[i]) {
            d_min += square(s.center[i] - b.p_max[i]);
        }
        if (d_min <= r2) {
            return true;
        }
    }
    return false;
}

DEVICE
inline bool intersect(const AABB3 &b, const Ray &r, Real expand_dist = 0) {
    // From https://github.com/mmp/pbrt-v3/blob/master/src/core/geometry.h
    auto t0 = r.tmin, t1 = r.tmax;
    for (int i = 0; i < 3; i++) {
        // Update interval for _i_th bounding box slab
        auto inv_ray_dir = 1 / r.dir[i];
        auto t_near = (b.p_min[i] - expand_dist - r.org[i]) * inv_ray_dir;
        auto t_far = (b.p_max[i] + expand_dist - r.org[i]) * inv_ray_dir;

        // Update parametric interval from slab intersection $t$ values
        if (t_near > t_far) {
            swap_(t_near, t_far);
        }

        // Update t_far to ensure robust ray bounds intersection
        t_far *= (1 + 1e-6f);
        t0 = t_near > t0 ? t_near : t0;
        t1 = t_far < t1 ? t_far : t1;
        if (t0 > t1) {
            return false;
        }
    }
    return true;
}

DEVICE
inline bool intersect(const AABB6 &b, const Ray &r, Real expand_dist = 0) {
    return intersect(convert_aabb<AABB3>(b), r, expand_dist);
}

DEVICE
inline bool intersect(const AABB3 &b, const Ray &r,
                      const Vector3 &inv_dir, const TVector3<bool> dir_is_neg,
                      Real expand_dist = 0) {
    // From https://github.com/mmp/pbrt-v3/blob/master/src/core/geometry.h
    auto t_min = (b[dir_is_neg[0]].x - r.org.x) * inv_dir.x;
    auto t_max = (b[1 - dir_is_neg[0]].x - r.org.x) * inv_dir.x;
    auto ty_min = (b[dir_is_neg[1]].y - r.org.y) * inv_dir.y;
    auto ty_max = (b[1 - dir_is_neg[1]].y - r.org.y) * inv_dir.y;
    // Update tMax and tyMax to ensure robust bounds intersection
    t_max *= (1 + 1e-6f);
    t_min *= (1 + 1e-6f);
    if (t_min > ty_max || ty_min > t_max) {
        return false;
    }
    if (ty_min > t_min) {
        t_min = ty_min;
    }
    if (ty_max < t_max) {
        t_max = ty_max;
    }
    // Check for ray intersection against z slab
    auto tz_min = (b[dir_is_neg[2]].z - r.org.z) * inv_dir.z;
    auto tz_max = (b[1 - dir_is_neg[2]].z - r.org.z) * inv_dir.z;
    // Update tzMax to ensure robust bounds intersection
    tz_max *= (1 + 1e-6f);
    if (t_min > tz_max || tz_min > t_max) return false;
    if (tz_min > t_min) t_min = tz_min;
    if (tz_max < t_max) t_max = tz_max;
    return (t_min < r.tmax) && (t_max > r.tmin);
}

DEVICE
inline bool intersect(const AABB6 &b, const Ray &r,
                      const Vector3 &inv_dir, const TVector3<bool> dir_is_neg,
                      Real expand_dist = 0) {
    return intersect(convert_aabb<AABB3>(b), r,
        inv_dir, dir_is_neg, expand_dist);
}

std::ostream& operator<<(std::ostream &os, const AABB3 &bounds);
std::ostream& operator<<(std::ostream &os, const AABB6 &bounds);
