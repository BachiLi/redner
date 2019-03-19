#pragma once

#include "redner.h"
#include "vector.h"
#include "cuda_utils.h"

#include <iostream>

struct AABB3 {
    DEVICE AABB3(
        const Vector3 &p_min = Vector3{ infinity<Real>(),  infinity<Real>(),  infinity<Real>()},
        const Vector3 &p_max = Vector3{-infinity<Real>(), -infinity<Real>(), -infinity<Real>()})
            : p_min(p_min), p_max(p_max) {}

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

struct Sphere {
    Vector3 center;
    Real radius;
};

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
inline bool inside(const Sphere &b, const Vector3 &p) {
    return distance(p, b.center) <= b.radius;
}


DEVICE
inline bool intersect(const Sphere &s, const AABB3 &b) {
    // "A Simple Method for Box-Sphere Intersection Testing", Jim Arvo
    // https://github.com/erich666/GraphicsGems/blob/master/gems/BoxSphere.c
    auto d_min = Real(0);
    auto r2 = s.radius;
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

std::ostream& operator<<(std::ostream &os, const AABB3 &bounds);
std::ostream& operator<<(std::ostream &os, const AABB6 &bounds);