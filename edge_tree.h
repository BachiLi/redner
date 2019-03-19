#pragma once

#include "redner.h"
#include "vector.h"
#include "buffer.h"
#include "aabb.h"

#include <iostream>

struct Camera;
struct Shape;
struct Edge;

struct BVHNode3 {
    AABB3 bounds;
    Real weighted_total_length; // \sum length * (pi - dihedral angle)
    BVHNode3 *parent;
    BVHNode3 *children[2];
    int edge_id;
};

struct BVHNode6 {
    AABB6 bounds;
    Real weighted_total_length; // \sum length * (pi - dihedral angle)
    BVHNode6 *parent;
    BVHNode6 *children[2];
    int edge_id;
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

struct EdgeTreeRoots {
    const BVHNode3 *cs_bvh_root;
    const BVHNode6 *ncs_bvh_root;
};

inline EdgeTreeRoots get_edge_tree_roots(const EdgeTree *edge_tree) {
    if (edge_tree == nullptr) {
        return EdgeTreeRoots{nullptr, nullptr};
    } else {
        return EdgeTreeRoots{
            edge_tree->cs_bvh_nodes.begin(),
            edge_tree->ncs_bvh_nodes.begin()};
    }
}
