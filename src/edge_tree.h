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
    Real cost;
};

struct BVHNode6 {
    AABB6 bounds;
    Real weighted_total_length; // \sum length * (pi - dihedral angle)
    BVHNode6 *parent;
    BVHNode6 *children[2];
    int edge_id;
    Real cost;
};

struct BVHNodePtr {
    DEVICE BVHNodePtr() {}
    DEVICE BVHNodePtr(const BVHNode3 *ptr3)
        : is_bvh_node3(true), ptr3(ptr3) {}
    DEVICE BVHNodePtr(const BVHNode6 *ptr6)
        : is_bvh_node3(false), ptr6(ptr6) {}

    bool is_bvh_node3;
    union {
        const BVHNode3 *ptr3;
        const BVHNode6 *ptr6;
    };
};

DEVICE
inline bool is_leaf(const BVHNodePtr &node_ptr) {
    if (node_ptr.is_bvh_node3) {
        return node_ptr.ptr3->edge_id != -1;
    } else {
        return node_ptr.ptr6->edge_id != -1;
    }
}

DEVICE
inline int get_edge_id(const BVHNodePtr &node_ptr) {
    if (node_ptr.is_bvh_node3) {
        return node_ptr.ptr3->edge_id;
    } else {
        return node_ptr.ptr6->edge_id;
    }
}

DEVICE
inline void get_children(const BVHNodePtr &node_ptr, BVHNodePtr children[2]) {
    if (node_ptr.is_bvh_node3) {
        children[0] = BVHNodePtr(node_ptr.ptr3->children[0]);
        children[1] = BVHNodePtr(node_ptr.ptr3->children[1]);
    } else {
        children[0] = BVHNodePtr(node_ptr.ptr6->children[0]);
        children[1] = BVHNodePtr(node_ptr.ptr6->children[1]);
    }
}

DEVICE
inline bool intersect(const BVHNodePtr &node_ptr, const Ray &ray,
                      const Real edge_bounds_expand = 0) {
    if (node_ptr.is_bvh_node3) {
        return intersect(node_ptr.ptr3->bounds, ray, edge_bounds_expand);
    } else {
        return intersect(node_ptr.ptr6->bounds, ray, edge_bounds_expand);
    }
}

DEVICE
inline bool intersect(const BVHNodePtr &node_ptr, const Ray &ray,
                      const Vector3 &inv_dir, const TVector3<bool> dir_is_neg,
                      const Real edge_bounds_expand = 0) {
    if (node_ptr.is_bvh_node3) {
        return intersect(node_ptr.ptr3->bounds, ray,
            inv_dir, dir_is_neg, edge_bounds_expand);
    } else {
        return intersect(node_ptr.ptr6->bounds, ray,
            inv_dir, dir_is_neg, edge_bounds_expand);
    }
}

struct EdgeTree {
    EdgeTree(bool use_gpu,
             const Camera &camera,
             const BufferView<Shape> &shapes,
             const BufferView<Edge> &edges);

    Buffer<BVHNode3> cs_bvh_nodes;
    Buffer<BVHNode3> cs_bvh_leaves;
    Buffer<BVHNode6> ncs_bvh_nodes;
    Buffer<BVHNode6> ncs_bvh_leaves;
    Real edge_bounds_expand;
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
