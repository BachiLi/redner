#include "shape.h"
#include "parallel.h"
#include "test_utils.h"

struct vertex_accumulator {
    DEVICE
    inline void operator()(int idx) {
        auto sid = d_vertices[idx].shape_id;
        auto vid = d_vertices[idx].vertex_id;
        const auto &d_v = d_vertices[idx].d_v;
        d_shapes[sid].vertices[3 * vid + 0] += d_v[0];
        d_shapes[sid].vertices[3 * vid + 1] += d_v[1];
        d_shapes[sid].vertices[3 * vid + 2] += d_v[2];
        if (d_shapes[sid].uvs != nullptr) {
            const auto &d_uv = d_vertices[idx].d_uv;
            d_shapes[sid].uvs[2 * vid + 0] += d_uv[0];
            d_shapes[sid].uvs[2 * vid + 1] += d_uv[1];
        }
        if (d_shapes[sid].normals != nullptr) {
            const auto &d_n = d_vertices[idx].d_n;
            d_shapes[sid].normals[3 * vid + 0] += d_n[0];
            d_shapes[sid].normals[3 * vid + 1] += d_n[1];
            d_shapes[sid].normals[3 * vid + 2] += d_n[2];
        }
    }

    const DVertex *d_vertices = nullptr;
    DShape *d_shapes = nullptr;
};

void accumulate_vertex(const BufferView<DVertex> &d_vertices,
                       BufferView<DShape> shapes,
                       bool use_gpu) {
    parallel_for(vertex_accumulator{
        d_vertices.begin(), shapes.begin()
    }, d_vertices.size(), use_gpu);
}

void test_d_intersect() {
    std::vector<Vector3f> vertices(3);
    vertices[0] = Vector3f{-1.f, 0.f, 1.f};
    vertices[1] = Vector3f{ 1.f, 0.f, 1.f};
    vertices[2] = Vector3f{ 0.f, 1.f, 1.f};
    std::vector<Vector3i> indices(1);
    indices[0] = Vector3i{0, 1, 2};
    Ray ray{Vector3{0, 0, 0}, Vector3{0, 0, 1}};
    Shape shape{ptr<float>(&vertices[0][0]),
                ptr<int>(&indices[0][0]),
                nullptr,
                nullptr,
                3,
                1,
                0,
                -1};
    SurfacePoint d_point;
    d_point.position = Vector3{1, 1, 1};
    d_point.geom_normal = Vector3{1, 1, 1};
    d_point.shading_frame =
        Frame{Vector3{1, 1, 1},
              Vector3{1, 1, 1},
              Vector3{1, 1, 1}};
    d_point.uv = Vector2{1, 1};

    DRay d_ray{};
    DVertex d_v[3] = {DVertex{}, DVertex{}, DVertex{}};
    d_intersect_shape(shape, 0, ray,
        d_point, d_ray, d_v);

    // Check ray derivatives
    auto finite_delta = Real(1e-5);
    for (int i = 0; i < 3; i++) {
        auto delta_ray = ray;
        delta_ray.org[i] += finite_delta;
        auto positive = intersect_shape(shape, 0, delta_ray);
        delta_ray.org[i] -= 2 * finite_delta;
        auto negative = intersect_shape(shape, 0, delta_ray);
        auto diff = (sum(positive.position - negative.position) +
                     sum(positive.geom_normal - negative.geom_normal) +
                     sum(positive.shading_frame.x - negative.shading_frame.x) +
                     sum(positive.shading_frame.y - negative.shading_frame.y) +
                     sum(positive.shading_frame.n - negative.shading_frame.n) +
                     sum(positive.uv - negative.uv)) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_ray.org[i]);
    }
    for (int i = 0; i < 3; i++) {
        auto delta_ray = ray;
        delta_ray.dir[i] += finite_delta;
        auto positive = intersect_shape(shape, 0, delta_ray);
        delta_ray.dir[i] -= 2 * finite_delta;
        auto negative = intersect_shape(shape, 0, delta_ray);
        auto diff = (sum(positive.position - negative.position) +
                     sum(positive.geom_normal - negative.geom_normal) +
                     sum(positive.shading_frame.x - negative.shading_frame.x) +
                     sum(positive.shading_frame.y - negative.shading_frame.y) +
                     sum(positive.shading_frame.n - negative.shading_frame.n) +
                     sum(positive.uv - negative.uv)) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_ray.dir[i]);
    }

    // Check vertex derivatives
    for (int vi = 0; vi < 3; vi++) {
        // Position
        for (int i = 0; i < 3; i++) {
            float tmp = vertices[vi][i];
            vertices[vi][i] += finite_delta;
            auto positive = intersect_shape(shape, 0, ray);
            vertices[vi][i] -= 2 * finite_delta;
            auto negative = intersect_shape(shape, 0, ray);
            vertices[vi][i] = tmp;
            auto diff = (sum(positive.position - negative.position) +
                         sum(positive.geom_normal - negative.geom_normal) +
                         sum(positive.shading_frame.x - negative.shading_frame.x) +
                         sum(positive.shading_frame.y - negative.shading_frame.y) +
                         sum(positive.shading_frame.n - negative.shading_frame.n) +
                         sum(positive.uv - negative.uv)) / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, diff, d_v[vi].d_v[i], Real(5e-3));
        }
    }
}

void test_d_sample_shape() {
    std::vector<Vector3f> vertices(3);
    vertices[0] = Vector3f{-1.f, 0.f, 1.f};
    vertices[1] = Vector3f{ 1.f, 0.f, 1.f};
    vertices[2] = Vector3f{ 0.f, 1.f, 1.f};
    std::vector<Vector3i> indices(1);
    indices[0] = Vector3i{0, 1, 2};
    Shape shape{ptr<float>(&vertices[0][0]),
                ptr<int>(&indices[0][0]),
                nullptr,
                nullptr,
                3,
                1,
                0,
                -1};
    auto sample = Vector2{0.5, 0.5};
    SurfacePoint d_point;
    d_point.position = Vector3{1, 1, 1};
    d_point.geom_normal = Vector3{1, 1, 1};
    d_point.shading_frame =
        Frame{Vector3{1, 1, 1},
              Vector3{1, 1, 1},
              Vector3{1, 1, 1}};
    d_point.uv = Vector2{1, 1};

    DVertex d_v[3] = {DVertex{}, DVertex{}, DVertex{}};
    d_sample_shape(shape, 0, sample, d_point, d_v);

    // Check vertex derivatives
    auto finite_delta = Real(1e-5);
    for (int vi = 0; vi < 3; vi++) {
        // Position
        for (int i = 0; i < 3; i++) {
            float tmp = vertices[vi][i];
            vertices[vi][i] += finite_delta;
            auto positive = sample_shape(shape, 0, sample);
            vertices[vi][i] -= 2 * finite_delta;
            auto negative = sample_shape(shape, 0, sample);
            vertices[vi][i] = tmp;
            auto diff = (sum(positive.position - negative.position) +
                         sum(positive.geom_normal - negative.geom_normal) +
                         sum(positive.shading_frame.x - negative.shading_frame.x) +
                         sum(positive.shading_frame.y - negative.shading_frame.y) +
                         sum(positive.shading_frame.n - negative.shading_frame.n) +
                         sum(positive.uv - negative.uv)) / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__,
                diff, d_v[vi].d_v[i], Real(5e-3));
        }
    }
}
