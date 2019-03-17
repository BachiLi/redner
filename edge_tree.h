#pragma once

#include "redner.h"
#include "vector.h"
#include "buffer.h"

struct Camera;
struct Shape;
struct Edge;

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

struct BVHNode3 {
    AABB3 bounds;
    BVHNode3 *parent;
    BVHNode3 *children[2];
    Real weighted_total_length; // \sum length * (pi - dihedral angle)
};

struct BVHNode6 {
    AABB6 bounds;
    BVHNode6 *parent;
    BVHNode6 *children[2];
    Real weighted_total_length; // \sum length * (pi - dihedral angle)
};

struct EdgeTree {
    EdgeTree(bool use_gpu,
             const Camera &camera,
             const BufferView<Shape> &shapes,
             const BufferView<Edge> &edges);

    Buffer<BVHNode3> cs_bvh_nodes;
    Buffer<BVHNode3> cs_bvh_leaves;
    Buffer<BVHNode6> ncs_bvh_nodes;
    Buffer<BVHNode6> ncs_bvh_leaves;
};
