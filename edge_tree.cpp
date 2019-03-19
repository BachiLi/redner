#include "edge_tree.h"
#include "vector.h"
#include "cuda_utils.h"
#include "atomic.h"
#include "edge.h"
#include "parallel.h"
#include "thrust_utils.h"

#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/partition.h>

struct edge_partitioner {
    DEVICE bool operator()(int edge_id) const {
        bool result = is_silhouette(shapes, cam_org, edges[edge_id]);
        return result;
    }

    const Shape *shapes;
    Vector3 cam_org;
    const Edge *edges;
};

struct edge_6d_bounds_computer {
    DEVICE void operator()(int idx) {
        const auto &edge = edges[edge_ids[idx]];
        // Compute position bound
        auto v0 = get_v0(shapes, edge);
        auto v1 = get_v1(shapes, edge);
        auto p_min = Vector3{0, 0, 0};
        auto p_max = Vector3{0, 0, 0};
        for (int i = 0; i < 3; i++) {
            p_min[i] = std::min(v0[i], v1[i]);
            p_max[i] = std::max(v0[i], v1[i]);
        }
        edge_aabbs[idx].p_min = p_min;
        edge_aabbs[idx].p_max = p_max;

        // Compute directional bound
        auto n0 = get_n0(shapes, edge);
        auto n1 = Vector3{0, 0, 0};
        if (edge.f1 == -1) {
            n1 = -n0;
        } else {
            n1 = get_n1(shapes, edge);
        }
        auto p = 0.5f * (v0 + v1) - cam_org;
        // plane 0 is n0.x * x + n0.y * y + n0.z * z = dot(p, n0)
        auto p0d = dot(p, n0);
        auto p1d = dot(p, n1);
        // 3D Hough transform, see "Silhouette extraction in hough space", Olson and Zhang
        auto h0 = Vector3{n0.x * p0d, n0.y * p0d, n0.z * p0d};
        auto h1 = Vector3{n1.x * p1d, n1.y * p1d, n1.z * p1d};
        auto d_min = Vector3{0, 0, 0};
        auto d_max = Vector3{0, 0, 0};
        for (int i = 0; i < 3; i++) {
            d_min[i] = min(h0[i], h1[i]);
            d_max[i] = max(h0[i], h1[i]);
        }
        edge_aabbs[idx].d_min = d_min;
        edge_aabbs[idx].d_max = d_max;
    }

    const Shape *shapes;
    const Edge *edges;
    const int *edge_ids;
    const Vector3 cam_org;
    AABB6 *edge_aabbs;
};

void compute_edge_bounds(const Shape *shapes,
                         const BufferView<Edge> &edges,
                         const BufferView<int> &edge_ids,
                         const Vector3 cam_org,
                         BufferView<AABB6> edge_aabbs,
                         bool use_gpu) {
    parallel_for(edge_6d_bounds_computer{
                     shapes, edges.begin(), edge_ids.begin(), cam_org, edge_aabbs.begin()},
                 edge_ids.size(),
                 use_gpu);
}

struct id_to_aabb3 {
    DEVICE AABB3 operator()(int id) const {
        auto b = bounds[id];
        return AABB3{b.p_min, b.p_max};
    }

    const AABB6 *bounds;
};

struct id_to_aabb6 {
    DEVICE AABB6 operator()(int id) const {
        return bounds[id];
    }

    const AABB6 *bounds;
};

struct union_bounding_box {
    DEVICE AABB6 operator()(const AABB6 &b0, const AABB6 &b1) const {
        auto p_min = Vector3{min(b0.p_min[0], b1.p_min[0]),
                             min(b0.p_min[1], b1.p_min[1]),
                             min(b0.p_min[2], b1.p_min[2])};
        auto d_min = Vector3{min(b0.d_min[0], b1.d_min[0]),
                             min(b0.d_min[1], b1.d_min[1]),
                             min(b0.d_min[2], b1.d_min[2])};
        auto p_max = Vector3{max(b0.p_max[0], b1.p_max[0]),
                             max(b0.p_max[1], b1.p_max[1]),
                             max(b0.p_max[2], b1.p_max[2])};
        auto d_max = Vector3{max(b0.d_max[0], b1.d_max[0]),
                             max(b0.d_max[1], b1.d_max[1]),
                             max(b0.d_max[2], b1.d_max[2])};
        return AABB6{p_min, d_min, p_max, d_max};
    }

    DEVICE AABB3 operator()(const AABB3 &b0, const AABB3 &b1) const {
        auto p_min = Vector3{min(b0.p_min[0], b1.p_min[0]),
                             min(b0.p_min[1], b1.p_min[1]),
                             min(b0.p_min[2], b1.p_min[2])};
        auto p_max = Vector3{max(b0.p_max[0], b1.p_max[0]),
                             max(b0.p_max[1], b1.p_max[1]),
                             max(b0.p_max[2], b1.p_max[2])};
        return AABB3{p_min, p_max};
    }
};

struct morton_code_3d_computer {
    DEVICE uint64_t expand_bits(uint64_t x) {
        // Insert two zero after every bit given a 10-bit integer
        // https://devblogs.nvidia.com/thinking-parallel-part-iii-tree-construction-gpu/
        x = (x * 0x00010001u) & 0xFF0000FFu;
        x = (x * 0x00000101u) & 0x0F00F00Fu;
        x = (x * 0x00000011u) & 0xC30C30C3u;
        x = (x * 0x00000005u) & 0x49249249u;
        return x;
    }

    DEVICE uint64_t morton3D(const Vector3 &p) {
        auto pp = (p - scene_bounds.p_min) / (scene_bounds.p_max - scene_bounds.p_min);
        for (int i = 0; i < 3; i++) {
            if (scene_bounds.p_max[i] - scene_bounds.p_min[i] <= 0.f) {
                pp[i] = 0.5f;
            }
        }
        return (expand_bits(pp.x) << 2u) |
               (expand_bits(pp.y) << 1u) |
               (expand_bits(pp.z) << 0u);
    }

    DEVICE void operator()(int idx) {
        // This might be suboptimal -- should probably use raw edge information directly
        auto box = convert_aabb<AABB3>(edge_aabbs[idx]);
        morton_codes[idx] = morton3D(0.5f * (box.p_min + box.p_max));
    }

    const AABB3 scene_bounds;
    const AABB6 *edge_aabbs;
    uint64_t *morton_codes;
};

void compute_morton_codes(const AABB3 &scene_bounds,
                          const BufferView<AABB6> &edge_bounds,
                          BufferView<uint64_t> morton_codes,
                          bool use_gpu) {
    parallel_for(morton_code_3d_computer{
                     scene_bounds, edge_bounds.begin(), morton_codes.begin()},
                 morton_codes.size(),
                 use_gpu);
}

struct morton_code_6d_computer {
    // For 6D Morton code, insert 5 zeros between each bit of a 10-bit integer
    // I'm doing this in a very slow way by manipulating each bit.
    // This is not the bottleneck anyway and I want readability.
    DEVICE uint64_t expand_bits(uint64_t x) {
        constexpr uint64_t mask = 0x1u;
        // We start from LSB (bit 63)
        auto result = (x & (mask << 0u)) << 5u;
        result |= (x & (mask << 1u)) << 10u;
        result |= (x & (mask << 2u)) << 15u;
        result |= (x & (mask << 3u)) << 20u;
        result |= (x & (mask << 4u)) << 25u;
        result |= (x & (mask << 5u)) << 30u;
        result |= (x & (mask << 6u)) << 35u;
        result |= (x & (mask << 7u)) << 40u;
        result |= (x & (mask << 8u)) << 45u;
        result |= (x & (mask << 9u)) << 50u;
        return result;
    }

    DEVICE uint64_t morton6D(const Vector3 &p, const Vector3 &d) {
        Vector3 pp = (p - scene_bounds.p_min) / (scene_bounds.p_max - scene_bounds.p_min);
        Vector3 dd = (d - scene_bounds.d_min) / (scene_bounds.d_max - scene_bounds.d_min);
        for (int i = 0; i < 3; i++) {
            if (scene_bounds.p_max[i] - scene_bounds.p_min[i] <= 0.f) {
                pp[i] = 0.5f;
            }
            if (scene_bounds.d_max[i] - scene_bounds.d_min[i] <= 0.f) {
                dd[i] = 0.5f;
            }
        }
        return (expand_bits(pp.x) << 5u) |
               (expand_bits(pp.y) << 4u) |
               (expand_bits(pp.z) << 3u) |
               (expand_bits(dd.x) << 2u) |
               (expand_bits(dd.y) << 1u) |
               (expand_bits(dd.z) << 0u);
    }

    DEVICE void operator()(int idx) {
        // This might be suboptimal -- should probably use raw edge information directly
        const auto &box = edge_aabbs[idx];
        morton_codes[idx] = morton6D(0.5f * (box.p_min + box.p_max),
                                     0.5f * (box.d_min + box.d_max));
    }

    const AABB6 scene_bounds;
    const AABB6 *edge_aabbs;
    uint64_t *morton_codes;
};

void compute_morton_codes(const AABB6 &scene_bounds,
                          const BufferView<AABB6> &edge_aabbs,
                          BufferView<uint64_t> morton_codes,
                          bool use_gpu) {
    parallel_for(morton_code_6d_computer{scene_bounds, edge_aabbs.begin(), morton_codes.begin()},
                 morton_codes.size(),
                 use_gpu);
}

template <typename BVHNodeType>
struct radix_tree_builder {
    // https://github.com/henrikdahlberg/GPUPathTracer/blob/master/Source/Core/BVHConstruction.cu#L62
    DEVICE int longest_common_prefix(int idx0, int idx1) {
        if (idx0 < 0 || idx0 >= num_primitives || idx1 < 0 || idx1 >= num_primitives) {
            return -1;
        }
        auto mc0 = morton_codes[idx0];
        auto mc1 = morton_codes[idx1];
        if (mc0 == mc1) {
            // Break even when the Morton codes are the same
            auto id0 = (uint64_t)edge_ids[idx0];
            auto id1 = (uint64_t)edge_ids[idx1];
            return clz(mc0 ^ mc1) + clz(id0 ^ id1);
        }
        else {
            return clz(mc0 ^ mc1);
        }
    }

    DEVICE void operator()(int idx) {
        // Mostly adapted from 
        // https://github.com/henrikdahlberg/GPUPathTracer/blob/master/Source/Core/BVHConstruction.cu#L161
        // Also see Figure 4 in
        // https://devblogs.nvidia.com/wp-content/uploads/2012/11/karras2012hpg_paper.pdf

        if (idx >= num_primitives - 1) {
            if (num_primitives == 1) {
                // Special case: if there is only one primitive, set it as the root
                nodes[0] = leaves[0];
            }
            return;
        }

        // Compute upper bound for the length of the range
        auto d = longest_common_prefix(idx, idx + 1) -
                 longest_common_prefix(idx, idx - 1) >= 0 ? 1 : -1;
        auto delta_min = longest_common_prefix(idx, idx - d);
        auto lmax = 2;
        while (longest_common_prefix(idx, idx + lmax * d) > delta_min) {
            lmax *= 2;
        }
        // Find the other end using binary search
        auto l = 0;
        auto divider = 2;
        for (int t = lmax / divider; t >= 1; divider *= 2) {
            if (longest_common_prefix(idx, idx + (l + t) * d) > delta_min) {
                l += t;
            }
            if (t == 1) {
                break;
            }
            t = lmax / divider;
        }
        auto j = idx + l * d;
        // Find the split position using binary search
        auto delta_node = longest_common_prefix(idx, j);
        auto s = 0;
        divider = 2;
        for (int t = (l + (divider - 1)) / divider; t >= 1; divider *= 2) {
            if (longest_common_prefix(idx, idx + (s + t) * d) > delta_node) {
                s += t;
            }
            if (t == 1) {
                break;
            }
            t = (l + (divider - 1)) / divider;
        }
        auto gamma = idx + s * d + min(d, 0);
        auto &node = nodes[idx];
        if (min(idx, j) == gamma) {
            node.children[0] = &leaves[gamma];
            leaves[gamma].parent = &node;
        } else {
            node.children[0] = &nodes[gamma];
            nodes[gamma].parent = &node;
        }
        if (max(idx, j) == gamma + 1) {
            node.children[1] = &leaves[gamma + 1];
            leaves[gamma + 1].parent = &node;
        } else {
            node.children[1] = &nodes[gamma + 1];
            nodes[gamma + 1].parent = &node;
        }
    }

    const uint64_t *morton_codes;
    const int *edge_ids;
    const int num_primitives;
    BVHNodeType *nodes;
    BVHNodeType *leaves;
};

template <typename BVHNodeType>
void build_radix_tree(const BufferView<uint64_t> &morton_codes,
                      const BufferView<int> &edge_ids,
                      BufferView<BVHNodeType> nodes,
                      BufferView<BVHNodeType> leaves,
                      bool use_gpu) {
    parallel_for(radix_tree_builder<BVHNodeType>{
        morton_codes.begin(), edge_ids.begin(),
            morton_codes.size(), nodes.begin(), leaves.begin()},
        morton_codes.size(),
        use_gpu);
}

template <typename BVHNodeType>
struct bvh_computer {
    DEVICE void operator()(int idx) {
        const auto &edge = edges[edge_ids[idx]];
        auto leaf = &leaves[idx];
        leaf->bounds = convert_aabb<decltype(leaf->bounds)>(bounds[edge_ids[idx]]);
        // length * (pi - dihedral angle)
        auto v0 = get_v0(shapes, edge);
        auto v1 = get_v1(shapes, edge);
        auto exterior_dihedral = compute_exterior_dihedral_angle(shapes, edge);
        leaf->weighted_total_length = distance(v0, v1) * exterior_dihedral;
        leaf->edge_id = edge_ids[idx];

        // Trace from leaf to root and merge bounding boxes & length
        auto current = leaf->parent;
        auto node_idx = current - nodes;
        if (current != nullptr) {
            while(true) {
                auto res = atomic_increment(node_counters + node_idx);
                if (res == 1) {
                    // Terminate the first thread entering this node to avoid duplicate computation
                    // It is important to terminate the first not the second so we ensure all children
                    // are processed
                    return;
                }
                auto bbox = current->children[0]->bounds;
                auto weighted_length = current->children[0]->weighted_total_length;
                for (int i = 1; i < 2; i++) {
                    bbox = merge(bbox, current->children[i]->bounds);
                    weighted_length += current->children[i]->weighted_total_length;
                }
                current->bounds = bbox;
                current->weighted_total_length = weighted_length;
                if (current->parent == nullptr) {
                    return;
                }
                current = current->parent;
                node_idx = current - nodes;
            }
        }
    }

    const Shape *shapes;
    const Edge *edges;
    const int *edge_ids;
    const AABB6 *bounds;
    int *node_counters;
    BVHNodeType *nodes;
    BVHNodeType *leaves;
};

template <typename BVHNodeType>
void compute_bvh(const BufferView<Shape> &shapes,
                 const BufferView<Edge> &edges,
                 const BufferView<int> &edge_ids,
                 const BufferView<AABB6> &bounds,
                 BufferView<int> node_counters,
                 BufferView<BVHNodeType> nodes,
                 BufferView<BVHNodeType> leaves,
                 bool use_gpu) {
    parallel_for(bvh_computer<BVHNodeType>{
            shapes.begin(), edges.begin(), edge_ids.begin(), bounds.begin(),
            node_counters.begin(), nodes.begin(), leaves.begin()},
        leaves.size(),
        use_gpu);
}

EdgeTree::EdgeTree(bool use_gpu,
                   const Camera &camera,
                   const BufferView<Shape> &shapes,
                   const BufferView<Edge> &edges) {
    if (edges.size() == 0) {
        return;
    }
    // We construct a 6D LBVH for the edges using AABB, where the first 3 dimensions are the
    // spatial dimensions and the rest are the 3D hough space as described in 
    // "Silhouette extraction in Hough space", Olson and Zhang

    // We use the camera position as the origin for the 3D Hough transform.
    // First, we split the edges into two sets.
    // 1) The edges that are silhouette when looking from the camera
    // 2) The rest
    //
    // According to Olson and Zhang, set 1 is a small set (and it includes all
    // "boundary" edges that are always silhouettes), and set 2 is a silhouette iff
    // it has exactly one point inside the "v-sphere" (the sphere whose center is at the query
    // point and the radius is the distance between the query point and the origin)
    // in Hough space.
    // This means we can build a BVH over set 2 and discard edges whose two endpoints
    // are both not inside the v-sphere during traversal.
    Buffer<int> edge_ids(use_gpu, edges.size());
    DISPATCH(use_gpu, thrust::sequence, edge_ids.begin(), edge_ids.end());
    auto cam_org = xfm_point(camera.cam_to_world, Vector3{0, 0, 0});
    auto partition_result = DISPATCH(use_gpu,
        thrust::stable_partition, edge_ids.begin(), edge_ids.end(),
        edge_partitioner{shapes.begin(), cam_org, edges.begin()});
    // We call the set of edges in 1) "cs_edges" and the set 2) "ncs_edges"
    BufferView<int> cs_edge_ids(edge_ids.begin(), partition_result - edge_ids.begin());
    BufferView<int> ncs_edge_ids(partition_result, edge_ids.end() - partition_result);
    Buffer<int> node_counters(use_gpu, edge_ids.size());
    Buffer<AABB6> edge_bounds(use_gpu, edge_ids.size());
    compute_edge_bounds(shapes.begin(),
                        edges,
                        edge_ids.view(0, edge_ids.size()),
                        cam_org,
                        edge_bounds.view(0, edge_ids.size()),
                        use_gpu);

    // We build a 3D BVH over the camera silhouette edges, and build
    // a 6D BVH over the non camera silhouette edges
    // camera silhouette edges
    if (cs_edge_ids.size() > 0) {
        // Compute scene bounding box for BVH
        AABB3 cs_scene_bounds = DISPATCH(use_gpu,
            thrust::transform_reduce, cs_edge_ids.begin(), cs_edge_ids.end(),
            id_to_aabb3{edge_bounds.begin()}, AABB3(), union_bounding_box{});
        assert(cs_scene_bounds.p_max.x - cs_scene_bounds.p_min.x >= 0.f &&
               cs_scene_bounds.p_max.y - cs_scene_bounds.p_min.y >= 0.f &&
               cs_scene_bounds.p_max.z - cs_scene_bounds.p_min.z >= 0.f);
        // Compute Morton code for LBVH
        Buffer<uint64_t> cs_morton_codes(use_gpu, cs_edge_ids.size());
        compute_morton_codes(cs_scene_bounds,
                             edge_bounds.view(0, cs_edge_ids.size()),
                             cs_morton_codes.view(0, cs_edge_ids.size()),
                             use_gpu);
        // Sort by Morton code
        DISPATCH(use_gpu, thrust::stable_sort_by_key,
            cs_morton_codes.begin(), cs_morton_codes.end(), cs_edge_ids.begin());
        cs_bvh_nodes = Buffer<BVHNode3>(use_gpu, max(cs_morton_codes.size() - 1, 1));
        cs_bvh_leaves = Buffer<BVHNode3>(use_gpu, cs_morton_codes.size());
        // Initialize nodes
        BVHNode3 init_node{AABB3(), Real(0), nullptr, {nullptr, nullptr}, -1};
        DISPATCH(use_gpu, thrust::fill, cs_bvh_nodes.begin(), cs_bvh_nodes.end(), init_node);
        DISPATCH(use_gpu, thrust::fill, cs_bvh_leaves.begin(), cs_bvh_leaves.end(), init_node);
        // Build tree (see
        // "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees")
        build_radix_tree(cs_morton_codes.view(0, cs_morton_codes.size()),
                         cs_edge_ids,
                         cs_bvh_nodes.view(0, cs_bvh_nodes.size()),
                         cs_bvh_leaves.view(0, cs_bvh_leaves.size()),
                         use_gpu);
        // Compute BVH node information (bounding box, length of edges, etc)
        DISPATCH(use_gpu, thrust::fill,
            node_counters.begin(), node_counters.begin() + cs_edge_ids.size(), 0);
        compute_bvh(shapes,
                    edges,
                    cs_edge_ids,
                    edge_bounds.view(0, edge_bounds.size()),
                    node_counters.view(0, cs_edge_ids.size()),
                    cs_bvh_nodes.view(0, cs_bvh_nodes.size()),
                    cs_bvh_leaves.view(0, cs_bvh_leaves.size()),
                    use_gpu);
    }

    // Do the same thing for non camera silhouette edges
    if (ncs_edge_ids.size() > 0) {
        // Compute scene bounding box for BVH
        AABB6 ncs_scene_bounds = DISPATCH(use_gpu,
            thrust::transform_reduce, ncs_edge_ids.begin(), ncs_edge_ids.end(),
            id_to_aabb6{edge_bounds.begin()}, AABB6(), union_bounding_box{});
        assert(ncs_scene_bounds.p_max.x - ncs_scene_bounds.p_min.x >= 0.f &&
               ncs_scene_bounds.p_max.y - ncs_scene_bounds.p_min.y >= 0.f &&
               ncs_scene_bounds.p_max.z - ncs_scene_bounds.p_min.z >= 0.f);
        assert(ncs_scene_bounds.d_max.x - ncs_scene_bounds.d_min.x >= 0.f &&
               ncs_scene_bounds.d_max.y - ncs_scene_bounds.d_min.y >= 0.f &&
               ncs_scene_bounds.d_max.z - ncs_scene_bounds.d_min.z >= 0.f);
        // Compute Morton code for LBVH
        Buffer<uint64_t> ncs_morton_codes(use_gpu, ncs_edge_ids.size());
        compute_morton_codes(ncs_scene_bounds,
                             edge_bounds.view(0, edge_bounds.size()),
                             ncs_morton_codes.view(0, ncs_edge_ids.size()),
                             use_gpu);
        // Sort by Morton code
        DISPATCH(use_gpu, thrust::stable_sort_by_key,
            ncs_morton_codes.begin(), ncs_morton_codes.end(), ncs_edge_ids.begin());
        ncs_bvh_nodes = Buffer<BVHNode6>(use_gpu, max(ncs_morton_codes.size() - 1, 1));
        ncs_bvh_leaves = Buffer<BVHNode6>(use_gpu, ncs_morton_codes.size());
        // Initialize nodes
        BVHNode6 init_node{AABB6(), Real(0), nullptr, {nullptr, nullptr}, -1};
        DISPATCH(use_gpu, thrust::fill, ncs_bvh_nodes.begin(), ncs_bvh_nodes.end(), init_node);
        DISPATCH(use_gpu, thrust::fill, ncs_bvh_leaves.begin(), ncs_bvh_leaves.end(), init_node);
        // Build tree (see
        // "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees")
        build_radix_tree(ncs_morton_codes.view(0, ncs_morton_codes.size()),
                         ncs_edge_ids,
                         ncs_bvh_nodes.view(0, ncs_bvh_nodes.size()),
                         ncs_bvh_leaves.view(0, ncs_bvh_leaves.size()),
                         use_gpu);
        // Compute BVH node information (bounding box, length of edges, etc)
        DISPATCH(use_gpu, thrust::fill,
            node_counters.begin(), node_counters.begin() + ncs_edge_ids.size(), 0);
        compute_bvh(shapes,
                    edges,
                    ncs_edge_ids,
                    edge_bounds.view(0, edge_bounds.size()),
                    node_counters.view(0, ncs_edge_ids.size()),
                    ncs_bvh_nodes.view(0, ncs_bvh_nodes.size()),
                    ncs_bvh_leaves.view(0, ncs_bvh_leaves.size()),
                    use_gpu);
    }
}
