#include "rebuild_topology.h"
#include "vector.h"

#include <map>
#include <cstring>
#include <vector>
#include <cassert>

// https://github.com/mitsuba-renderer/mitsuba/blob/master/src/librender/trimesh.cpp#L468
// TODO: this can be parallelize by sorting & segmented reduction

struct Vertex {
    Vector3f p;
    Vector2f uv;
    inline Vertex() : p(Vector3f{0, 0, 0}), uv(Vector2f{0, 0}) { }
};

struct TopoData {
    size_t idx;   /// Triangle index
    bool clustered; /// Has the tri-vert. pair been assigned to a cluster?
    inline TopoData() { }
    inline TopoData(size_t idx, bool clustered)
        : idx(idx), clustered(clustered) { }
};

/// For using vertices as keys in an associative structure
struct vertex_key_order : public
    std::binary_function<Vertex, Vertex, bool> {
    static int compare(const Vertex &v1, const Vertex &v2) {
        if (v1.p.x < v2.p.x) return -1;
        else if (v1.p.x > v2.p.x) return 1;
        if (v1.p.y < v2.p.y) return -1;
        else if (v1.p.y > v2.p.y) return 1;
        if (v1.p.z < v2.p.z) return -1;
        else if (v1.p.z > v2.p.z) return 1;
        if (v1.uv.x < v2.uv.x) return -1;
        else if (v1.uv.x > v2.uv.x) return 1;
        if (v1.uv.y < v2.uv.y) return -1;
        else if (v1.uv.y > v2.uv.y) return 1;
        return 0;
    }

    bool operator()(const Vertex &v1, const Vertex &v2) const {
        return compare(v1, v2) < 0;
    }
};

// https://github.com/mitsuba-renderer/mitsuba/blob/1fd0f671dfcb77f813c0d6a36f2aa4e480b5ca8e/include/mitsuba/core/util.h#L299
inline float unit_angle(const Vector3f &u, const Vector3f &v) {
    if (dot(u, v) < 0) {
        return float(M_PI) - 2 * asin(0.5f * length(v + u));
    } else {
        return 2 * asin(0.5f * length(v - u));
    }
}

int rebuild_topology(ptr<float> vertices,
                     ptr<int> indices,
                     ptr<float> uvs,
                     ptr<float> normals,
                     int num_vertices,
                     int num_triangles,
                     float max_smooth_angle) {
    using MMap = std::multimap<Vertex, TopoData, vertex_key_order>;

    auto dot_threshold = cos(max_smooth_angle * Real(M_PI) / 180.f);
    auto vertex_to_face = MMap();
    auto new_vertices = std::vector<Vector3f>();
    auto new_uvs = std::vector<Vector2f>();
    auto face_normals = std::vector<Vector3f>(num_triangles);
    auto new_indices = std::vector<Vector3i>(num_triangles);

    new_vertices.reserve(num_vertices);
    if (uvs.get() != nullptr) {
        new_uvs.reserve(num_vertices);
    }

    // Create an associative list and precompute a few things
    for (size_t i = 0; i < (size_t)num_triangles; ++i) {
        auto index = Vector3i{indices[3 * i + 0],
                              indices[3 * i + 1],
                              indices[3 * i + 2]};
        auto v = Vertex();
        for (int j = 0; j < 3; j++) {
            v.p = Vector3f{vertices[3 * index[j] + 0],
                           vertices[3 * index[j] + 1],
                           vertices[3 * index[j] + 2]};
            if (uvs.get() != nullptr) {
                v.uv = Vector2f{uvs[2 * index[j] + 0],
                                uvs[2 * index[j] + 1]};
            }
            vertex_to_face.insert({v, TopoData(i, false)});
        }
        auto v0 = Vector3f{vertices[3 * index[0] + 0],
                           vertices[3 * index[0] + 1],
                           vertices[3 * index[0] + 2]};
        auto v1 = Vector3f{vertices[3 * index[1] + 0],
                           vertices[3 * index[1] + 1],
                           vertices[3 * index[1] + 2]};
        auto v2 = Vector3f{vertices[3 * index[2] + 0],
                           vertices[3 * index[2] + 1],
                           vertices[3 * index[2] + 2]};

        auto n = cross(v1 - v0, v2 - v0);
        auto l = length(n);
        if (l > 1e-20f) {
            n /= l;
        } else {
            // Degenerate triangle
            n = Vector3f{0, 0, 0};
        }

        face_normals[i] = n;
        new_indices[i] = Vector3i{-1, -1, -1};
    }

    for (auto it = vertex_to_face.begin(); it != vertex_to_face.end();) {
        auto start = vertex_to_face.lower_bound(it->first);
        auto end = vertex_to_face.upper_bound(it->first);

        // Greedy clustering of normals
        for (auto it2 = start; it2 != end; it2++) {
            const auto &v = it2->first;
            const auto &t1 = it2->second;
            auto n1 = face_normals[t1.idx];
            if (t1.clustered) {
                continue;
            }

            auto vertex_idx = (int) new_vertices.size();
            new_vertices.push_back(v.p);
            if (uvs.get() != nullptr) {
                new_uvs.push_back(v.uv);
            }

            for (auto it3 = it2; it3 != end; it3++) {
                auto &t2 = it3->second;
                if (t2.clustered) {
                    continue;
                }
                auto n2 = face_normals[t2.idx];

                if (n1 == n2 || dot(n1, n2) > dot_threshold) {
                    auto index = Vector3i{
                        indices[3 * t2.idx + 0],
                        indices[3 * t2.idx + 1],
                        indices[3 * t2.idx + 2]};
                    for (int i = 0; i < 3; i++) {
                        auto vv = Vector3f{
                            vertices[3 * index[i] + 0],
                            vertices[3 * index[i] + 1],
                            vertices[3 * index[i] + 2]};
                        if (vv == v.p) {
                            new_indices[t2.idx][i] = vertex_idx;
                        }
                    }
                    t2.clustered = true;
                }
            }
        }
        it = end;
    }

    for (int i = 0; i < num_triangles; i++) {
        for (int j = 0; j < 3; j++) {
            if (new_indices[i][j] < 0) {
                throw std::runtime_error("Error occurs during rebuilding topology");
            }
        }
    }

    memcpy(vertices.get(), &new_vertices[0], sizeof(Vector3f) * new_vertices.size());
    if (uvs.get() != nullptr) {
        assert(new_uvs.size() == new_vertices.size());
        memcpy(uvs.get(), &new_uvs[0], sizeof(Vector2f) * new_uvs.size());
    }
    memcpy(indices.get(), &new_indices[0], sizeof(Vector3i) * new_indices.size());

    // Compute normals
    // Nelson Max, "Weights for Computing Vertex Normals from Facet Vectors", 1999
    std::vector<Vector3f> new_normals(new_vertices.size(), Vector3f{0, 0, 0});
    for (int i = 0; i < num_triangles; i++) {
        auto n = Vector3f{0, 0, 0};
        auto index = Vector3{indices[3 * i + 0],
                             indices[3 * i + 1],
                             indices[3 * i + 2]};
        for (int j = 0; j < 3; j++) {
            auto v0 = new_vertices[index[j + 0]];
            auto v1 = new_vertices[index[(j + 1) % 3]];
            auto v2 = new_vertices[index[(j + 2) % 3]];
            auto e1 = v1 - v0;
            auto e2 = v2 - v0;
            if (j == 0) {
                n = cross(e1, e2);
                auto l = length(n);
                if (l <= 1e-20f) {
                    // Degenerate triangle
                    break;
                }
                n /= l;
            }
            auto angle = unit_angle(normalize(e1), normalize(e2));
            new_normals[index[j]] += n * sin(angle) / (length(e1) * length(e2));
        }
    }

    for (int i = 0; i < (int)new_normals.size(); i++) {
        auto &n = new_normals[i];
        auto l = length(n);
        if (l > 1e-20f) {
            n /= l;
        } else {
            // Choose some arbitrary value
            // https://github.com/mitsuba-renderer/mitsuba/blob/master/src/librender/trimesh.cpp#L668
            n = Vector3f{0, 0, 1};
        }
    }

    return new_vertices.size();
}
