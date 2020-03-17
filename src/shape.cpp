#include "shape.h"
#include "parallel.h"
#include "test_utils.h"

void test_d_intersect() {
    std::vector<Vector3f> vertices(3);
    vertices[0] = Vector3f{-1.f, 0.f, 1.f};
    vertices[1] = Vector3f{ 1.f, 0.f, 1.f};
    vertices[2] = Vector3f{ 0.f, 1.f, 1.f};
    std::vector<Vector3i> indices(1);
    indices[0] = Vector3i{0, 1, 2};
    Ray ray{Vector3{0, 0, 0}, Vector3{0, 0, 1}};
    RayDifferential ray_diff{Vector3{1, 1, 1}, Vector3{1, 1, 1},
                             Vector3{1, 1, 1}, Vector3{1, 1, 1}};
    Shape shape{ptr<float>(&vertices[0][0]),
                ptr<int>(&indices[0][0]),
                nullptr, // uvs
                nullptr, // normals
                nullptr, // uv_indices
                nullptr, // normal_indices
                nullptr, // color
                3, // num_vertices
                0, // num_uv_vertices
                0, // num_normal_vertices
                1, // num_triangles
                0,
                -1};
    SurfacePoint d_point;
    d_point.position = Vector3{1, 1, 1};
    d_point.geom_normal = Vector3{1, 1, 1};
    d_point.shading_frame = Frame{Vector3{1, 1, 1},
                                  Vector3{1, 1, 1},
                                  Vector3{1, 1, 1}};
    d_point.uv = Vector2{1, 1};
    d_point.du_dxy = Vector2{1, 1};
    d_point.dv_dxy = Vector2{1, 1};
    d_point.dn_dx = Vector3{1, 1, 1};
    d_point.dn_dy = Vector3{1, 1, 1};
    d_point.color = Vector3{1, 1, 1};
    RayDifferential d_new_ray_diff{
        Vector3{1, 1, 1}, Vector3{1, 1, 1},
        Vector3{1, 1, 1}, Vector3{1, 1, 1}};

    DRay d_ray{};
    RayDifferential d_ray_diff{
        Vector3{0, 0, 0}, Vector3{0, 0, 0},
        Vector3{0, 0, 0}, Vector3{0, 0, 0}};
    Vector3 d_v_p[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
    Vector3 d_v_n[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
    Vector2 d_v_uv[3] = {Vector2{0, 0}, Vector2{0, 0}, Vector2{0, 0}};
    Vector3 d_v_c[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
    d_intersect_shape(shape,
                      0,
                      ray,
                      ray_diff,
                      d_point,
                      d_new_ray_diff,
                      d_ray,
                      d_ray_diff,
                      d_v_p,
                      d_v_n,
                      d_v_uv,
                      d_v_c);
    // Check ray derivatives
    auto finite_delta = Real(1e-4);
    for (int i = 0; i < 3; i++) {
        auto ray_diff_pos = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}};
        auto ray_diff_neg = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}};
        auto delta_ray = ray;
        delta_ray.org[i] += finite_delta;
        auto positive = intersect_shape(shape, 0, delta_ray, ray_diff, ray_diff_pos);
        delta_ray.org[i] -= 2 * finite_delta;
        auto negative = intersect_shape(shape, 0, delta_ray, ray_diff, ray_diff_neg);
        auto diff = (sum(positive.position - negative.position) +
                     sum(positive.geom_normal - negative.geom_normal) +
                     sum(positive.shading_frame.x - negative.shading_frame.x) +
                     sum(positive.shading_frame.y - negative.shading_frame.y) +
                     sum(positive.shading_frame.n - negative.shading_frame.n) +
                     sum(positive.uv - negative.uv) +
                     sum(positive.du_dxy - negative.du_dxy) +
                     sum(positive.dv_dxy - negative.dv_dxy) +
                     sum(positive.dn_dx - negative.dn_dx) +
                     sum(positive.dn_dy - negative.dn_dy) +
                     sum(ray_diff_pos.org_dx - ray_diff_neg.org_dx) +
                     sum(ray_diff_pos.org_dy - ray_diff_neg.org_dy) +
                     sum(ray_diff_pos.dir_dx - ray_diff_neg.dir_dx) +
                     sum(ray_diff_pos.dir_dy - ray_diff_neg.dir_dy)) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_ray.org[i]);
    }
    for (int i = 0; i < 3; i++) {
        auto ray_diff_pos = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}};
        auto ray_diff_neg = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}};
        auto delta_ray = ray;
        delta_ray.dir[i] += finite_delta;
        auto positive = intersect_shape(shape, 0, delta_ray, ray_diff, ray_diff_pos);
        delta_ray.dir[i] -= 2 * finite_delta;
        auto negative = intersect_shape(shape, 0, delta_ray, ray_diff, ray_diff_neg);
        auto diff = (sum(positive.position - negative.position) +
                     sum(positive.geom_normal - negative.geom_normal) +
                     sum(positive.shading_frame.x - negative.shading_frame.x) +
                     sum(positive.shading_frame.y - negative.shading_frame.y) +
                     sum(positive.shading_frame.n - negative.shading_frame.n) +
                     sum(positive.uv - negative.uv) +
                     sum(positive.du_dxy - negative.du_dxy) +
                     sum(positive.dv_dxy - negative.dv_dxy) +
                     sum(positive.dn_dx - negative.dn_dx) +
                     sum(positive.dn_dy - negative.dn_dy) +
                     sum(ray_diff_pos.org_dx - ray_diff_neg.org_dx) +
                     sum(ray_diff_pos.org_dy - ray_diff_neg.org_dy) +
                     sum(ray_diff_pos.dir_dx - ray_diff_neg.dir_dx) +
                     sum(ray_diff_pos.dir_dy - ray_diff_neg.dir_dy)) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_ray.dir[i]);
    }

    // Check ray differential derivatives
    for (int i = 0; i < 3; i++) {
        auto ray_diff_pos = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}};
        auto ray_diff_neg = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}};
        auto delta_ray_diff = ray_diff;
        delta_ray_diff.org_dx[i] += finite_delta;
        auto positive = intersect_shape(shape, 0, ray, delta_ray_diff, ray_diff_pos);
        delta_ray_diff.org_dx[i] -= 2 * finite_delta;
        auto negative = intersect_shape(shape, 0, ray, delta_ray_diff, ray_diff_neg);
        auto diff = (sum(positive.position - negative.position) +
                     sum(positive.geom_normal - negative.geom_normal) +
                     sum(positive.shading_frame.x - negative.shading_frame.x) +
                     sum(positive.shading_frame.y - negative.shading_frame.y) +
                     sum(positive.shading_frame.n - negative.shading_frame.n) +
                     sum(positive.uv - negative.uv) +
                     sum(positive.du_dxy - negative.du_dxy) +
                     sum(positive.dv_dxy - negative.dv_dxy) +
                     sum(positive.dn_dx - negative.dn_dx) +
                     sum(positive.dn_dy - negative.dn_dy) +
                     sum(ray_diff_pos.org_dx - ray_diff_neg.org_dx) +
                     sum(ray_diff_pos.org_dy - ray_diff_neg.org_dy) +
                     sum(ray_diff_pos.dir_dx - ray_diff_neg.dir_dx) +
                     sum(ray_diff_pos.dir_dy - ray_diff_neg.dir_dy)) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_ray_diff.org_dx[i]);
    }
    for (int i = 0; i < 3; i++) {
        auto ray_diff_pos = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}};
        auto ray_diff_neg = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}};
        auto delta_ray_diff = ray_diff;
        delta_ray_diff.org_dy[i] += finite_delta;
        auto positive = intersect_shape(shape, 0, ray, delta_ray_diff, ray_diff_pos);
        delta_ray_diff.org_dy[i] -= 2 * finite_delta;
        auto negative = intersect_shape(shape, 0, ray, delta_ray_diff, ray_diff_neg);
        auto diff = (sum(positive.position - negative.position) +
                     sum(positive.geom_normal - negative.geom_normal) +
                     sum(positive.shading_frame.x - negative.shading_frame.x) +
                     sum(positive.shading_frame.y - negative.shading_frame.y) +
                     sum(positive.shading_frame.n - negative.shading_frame.n) +
                     sum(positive.uv - negative.uv) +
                     sum(positive.du_dxy - negative.du_dxy) +
                     sum(positive.dv_dxy - negative.dv_dxy) +
                     sum(positive.dn_dx - negative.dn_dx) +
                     sum(positive.dn_dy - negative.dn_dy) +
                     sum(ray_diff_pos.org_dx - ray_diff_neg.org_dx) +
                     sum(ray_diff_pos.org_dy - ray_diff_neg.org_dy) +
                     sum(ray_diff_pos.dir_dx - ray_diff_neg.dir_dx) +
                     sum(ray_diff_pos.dir_dy - ray_diff_neg.dir_dy)) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_ray_diff.org_dy[i]);
    }
    for (int i = 0; i < 3; i++) {
        auto ray_diff_pos = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}};
        auto ray_diff_neg = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}};
        auto delta_ray_diff = ray_diff;
        delta_ray_diff.dir_dx[i] += finite_delta;
        auto positive = intersect_shape(shape, 0, ray, delta_ray_diff, ray_diff_pos);
        delta_ray_diff.dir_dx[i] -= 2 * finite_delta;
        auto negative = intersect_shape(shape, 0, ray, delta_ray_diff, ray_diff_neg);
        auto diff = (sum(positive.position - negative.position) +
                     sum(positive.geom_normal - negative.geom_normal) +
                     sum(positive.shading_frame.x - negative.shading_frame.x) +
                     sum(positive.shading_frame.y - negative.shading_frame.y) +
                     sum(positive.shading_frame.n - negative.shading_frame.n) +
                     sum(positive.uv - negative.uv) +
                     sum(positive.du_dxy - negative.du_dxy) +
                     sum(positive.dv_dxy - negative.dv_dxy) +
                     sum(positive.dn_dx - negative.dn_dx) +
                     sum(positive.dn_dy - negative.dn_dy) +
                     sum(ray_diff_pos.org_dx - ray_diff_neg.org_dx) +
                     sum(ray_diff_pos.org_dy - ray_diff_neg.org_dy) +
                     sum(ray_diff_pos.dir_dx - ray_diff_neg.dir_dx) +
                     sum(ray_diff_pos.dir_dy - ray_diff_neg.dir_dy)) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_ray_diff.dir_dx[i]);
    }
    for (int i = 0; i < 3; i++) {
        auto ray_diff_pos = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}};
        auto ray_diff_neg = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}};
        auto delta_ray_diff = ray_diff;
        delta_ray_diff.dir_dy[i] += finite_delta;
        auto positive = intersect_shape(shape, 0, ray, delta_ray_diff, ray_diff_pos);
        delta_ray_diff.dir_dy[i] -= 2 * finite_delta;
        auto negative = intersect_shape(shape, 0, ray, delta_ray_diff, ray_diff_neg);
        auto diff = (sum(positive.position - negative.position) +
                     sum(positive.geom_normal - negative.geom_normal) +
                     sum(positive.shading_frame.x - negative.shading_frame.x) +
                     sum(positive.shading_frame.y - negative.shading_frame.y) +
                     sum(positive.shading_frame.n - negative.shading_frame.n) +
                     sum(positive.uv - negative.uv) +
                     sum(positive.du_dxy - negative.du_dxy) +
                     sum(positive.dv_dxy - negative.dv_dxy) +
                     sum(positive.dn_dx - negative.dn_dx) +
                     sum(positive.dn_dy - negative.dn_dy) +
                     sum(ray_diff_pos.org_dx - ray_diff_neg.org_dx) +
                     sum(ray_diff_pos.org_dy - ray_diff_neg.org_dy) +
                     sum(ray_diff_pos.dir_dx - ray_diff_neg.dir_dx) +
                     sum(ray_diff_pos.dir_dy - ray_diff_neg.dir_dy)) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_ray_diff.dir_dy[i]);
    }

    // Check vertex derivatives
    for (int vi = 0; vi < 3; vi++) {
        // Position
        for (int i = 0; i < 3; i++) {
            auto ray_diff_pos = RayDifferential{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
                Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            auto ray_diff_neg = RayDifferential{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
                Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            auto tmp = vertices[vi][i];
            vertices[vi][i] += (float)finite_delta;
            auto positive = intersect_shape(shape, 0, ray, ray_diff, ray_diff_pos);
            vertices[vi][i] -= float(2 * finite_delta);
            auto negative = intersect_shape(shape, 0, ray, ray_diff, ray_diff_neg);
            vertices[vi][i] = tmp;
            auto diff = (sum(positive.position - negative.position) +
                         sum(positive.geom_normal - negative.geom_normal) +
                         sum(positive.shading_frame.x - negative.shading_frame.x) +
                         sum(positive.shading_frame.y - negative.shading_frame.y) +
                         sum(positive.shading_frame.n - negative.shading_frame.n) +
                         sum(positive.uv - negative.uv) +
                         sum(positive.du_dxy - negative.du_dxy) +
                         sum(positive.dv_dxy - negative.dv_dxy) +
                         sum(positive.dn_dx - negative.dn_dx) +
                         sum(positive.dn_dy - negative.dn_dy) +
                         sum(ray_diff_pos.org_dx - ray_diff_neg.org_dx) +
                         sum(ray_diff_pos.org_dy - ray_diff_neg.org_dy) +
                         sum(ray_diff_pos.dir_dx - ray_diff_neg.dir_dx) +
                         sum(ray_diff_pos.dir_dy - ray_diff_neg.dir_dy)) / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, diff, d_v_p[vi][i], Real(5e-3));
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
                nullptr, // uvs
                nullptr, // normals
                nullptr, // uv_indices
                nullptr, // normal_indices
                nullptr, // color
                3, // num_vertices
                0, // num_uv_vertices
                0, // num_normal_vertices
                1, // num_triangles
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
    d_point.du_dxy = Vector2{1, 1};
    d_point.dv_dxy = Vector2{1, 1};
    d_point.dn_dx = Vector3{1, 1, 1};
    d_point.dn_dy = Vector3{1, 1, 1};
    d_point.color = Vector3{1, 1, 1};

    Vector3 d_v[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
    d_sample_shape(shape, 0, sample, d_point, d_v);

    // Check vertex derivatives
    auto finite_delta = Real(1e-5);
    for (int vi = 0; vi < 3; vi++) {
        // Position
        for (int i = 0; i < 3; i++) {
            auto tmp = vertices[vi][i];
            vertices[vi][i] += float(finite_delta);
            auto positive = sample_shape(shape, 0, sample);
            vertices[vi][i] -= float(2 * finite_delta);
            auto negative = sample_shape(shape, 0, sample);
            vertices[vi][i] = tmp;
            auto diff = (sum(positive.position - negative.position) +
                         sum(positive.geom_normal - negative.geom_normal) +
                         sum(positive.shading_frame.x - negative.shading_frame.x) +
                         sum(positive.shading_frame.y - negative.shading_frame.y) +
                         sum(positive.shading_frame.n - negative.shading_frame.n) +
                         sum(positive.uv - negative.uv)) / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__,
                diff, d_v[vi][i], Real(5e-3));
        }
    }
}
