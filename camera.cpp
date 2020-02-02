#include "camera.h"
#include "parallel.h"
#include "test_utils.h"
#include "buffer.h"

#include <cmath>

struct primary_ray_sampler {
    DEVICE void operator()(int idx) {
        auto pixel_x = idx % camera.width;
        auto pixel_y = idx / camera.width;
        auto sample = samples[idx].xy;
        auto screen_pos = Vector2{
            (pixel_x + sample[0]) / Real(camera.width),
            (pixel_y + sample[1]) / Real(camera.height)
        };

        auto ray = sample_primary(camera, screen_pos);
        rays[idx] = ray;
        // Ray differential computation
        auto delta = Real(1e-3);
        auto screen_pos_dx = screen_pos + Vector2{delta, Real(0)};
        auto ray_dx = sample_primary(camera, screen_pos_dx);
        auto screen_pos_dy = screen_pos + Vector2{Real(0), delta};
        auto ray_dy = sample_primary(camera, screen_pos_dy);
        auto pixel_size_x = Real(0.5) / camera.width;
        auto pixel_size_y = Real(0.5) / camera.height;
        auto org_dx = pixel_size_x * (ray_dx.org - ray.org) / delta;
        auto org_dy = pixel_size_y * (ray_dy.org - ray.org) / delta;
        auto dir_dx = pixel_size_x * (ray_dx.dir - ray.dir) / delta;
        auto dir_dy = pixel_size_y * (ray_dy.dir - ray.dir) / delta;
        ray_differentials[idx] = RayDifferential{org_dx, org_dy, dir_dx, dir_dy};
    }

    const Camera camera;
    const CameraSample *samples;
    Ray *rays;
    RayDifferential *ray_differentials;
};

void sample_primary_rays(const Camera &camera,
                         const BufferView<CameraSample> &samples,
                         BufferView<Ray> rays,
                         BufferView<RayDifferential> ray_differentials,
                         bool use_gpu) {
    parallel_for(primary_ray_sampler{
        camera, samples.begin(), rays.begin(), ray_differentials.begin()},
        samples.size(), use_gpu);
}

void test_sample_primary_rays(bool use_gpu) {
    // Let's have a perspective camera with 1x1 pixel, 
    // with identity to world matrix,
    // fov 45 degree
    auto pos = Vector3f{0, 0, 0};
    auto look = Vector3f{0, 0, 1};
    auto up = Vector3f{0, 1, 0};
    Matrix3x3f n2c = Matrix3x3f::identity();
    Matrix3x3f c2n = Matrix3x3f::identity();
    Camera camera{1, 1,
        &pos[0],
        &look[0],
        &up[0],
        nullptr, // cam_to_world
        nullptr, // world_to_cam
        &n2c.data[0][0],
        &c2n.data[0][0],
        1e-2f,
        CameraType::Perspective};
    parallel_init();

    // Sample from the center of pixel
    Buffer<CameraSample> samples(use_gpu, 1);
    samples[0].xy = Vector2{0.5f, 0.5f};
    Buffer<Ray> rays(use_gpu, 1);
    Buffer<RayDifferential> ray_differentials(use_gpu, 1);
    sample_primary_rays(camera,
                        samples.view(0, 1),
                        rays.view(0, 1),
                        ray_differentials.view(0, 1),
                        use_gpu);
    cuda_synchronize();

    equal_or_error<Real>(__FILE__, __LINE__, rays[0].org, Vector3{0, 0, 0});
    equal_or_error<Real>(__FILE__, __LINE__, rays[0].dir, Vector3{0, 0, 1});
    // TODO: test ray differentials

    parallel_cleanup();
}

void test_d_sample_primary_rays() {
    auto pos = Vector3f{0, 0, 0};
    auto look = Vector3f{0, 0, 1};
    auto up = Vector3f{0, 1, 0};
    Matrix3x3f n2c = Matrix3x3f::identity();
    Matrix3x3f c2n = Matrix3x3f::identity();
    Camera camera{1, 1,
        &pos[0],
        &look[0],
        &up[0],
        nullptr, // cam_to_world
        nullptr, // world_to_cam
        &n2c.data[0][0],
        &c2n.data[0][0],
        1e-2f,
        CameraType::Perspective};
    auto d_pos = Vector3f{0, 0, 0};
    auto d_look = Vector3f{0, 0, 0};
    auto d_up = Vector3f{0, 0, 0};
    Matrix3x3f d_n2c = Matrix3x3f{};
    Matrix3x3f d_c2n = Matrix3x3f{};
    DCamera d_camera{&d_pos[0],
                     &d_look[0],
                     &d_up[0],
                     nullptr, // cam_to_world
                     nullptr, // world_to_cam
                     &d_n2c(0, 0),
                     &d_c2n(0, 0)};
    DRay d_ray{Vector3{1, 1, 1}, Vector3{1, 1, 1}};
    d_sample_primary_ray(camera,
                         Vector2{0.5, 0.5},
                         d_ray,
                         d_camera,
                         nullptr); // d_screen_pos
    // Compare with central difference
    auto finite_delta = Real(1e-6);
    for (int i = 0; i < 3; i++) {
        auto delta_pos = pos;
        delta_pos[i] += finite_delta;
        auto delta_camera = Camera{
            1, 1,
            &delta_pos[0],
            &look[0],
            &up[0],
            nullptr, // cam_to_world
            nullptr, // world_to_cam
            &n2c.data[0][0],
            &c2n.data[0][0],
            1e-2f,
            CameraType::Perspective};
        auto positive_ray =
            sample_primary(delta_camera, Vector2{0.5, 0.5});
        delta_pos[i] -= 2 * finite_delta;
        delta_camera = Camera{
            1, 1,
            &delta_pos[0],
            &look[0],
            &up[0],
            nullptr, // cam_to_world
            nullptr, // world_to_cam
            &n2c.data[0][0],
            &c2n.data[0][0],
            1e-2f,
            CameraType::Perspective};
        auto negative_ray =
            sample_primary(delta_camera, Vector2{0.5, 0.5});
        auto diff = (sum(positive_ray.org - negative_ray.org) +
                     sum(positive_ray.dir - negative_ray.dir)) /
                    (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, (Real)diff, (Real)d_camera.position[i]);
    }
    for (int i = 0; i < 3; i++) {
        auto delta_look = look;
        delta_look[i] += finite_delta;
        auto delta_camera = Camera{
            1, 1,
            &pos[0],
            &delta_look[0],
            &up[0],
            nullptr, // cam_to_world
            nullptr, // world_to_cam
            &n2c.data[0][0],
            &c2n.data[0][0],
            1e-2f,
            CameraType::Perspective};
        auto positive_ray =
            sample_primary(delta_camera, Vector2{0.5, 0.5});
        delta_look[i] -= 2 * finite_delta;
        delta_camera = Camera{
            1, 1,
            &pos[0],
            &delta_look[0],
            &up[0],
            nullptr, // cam_to_world
            nullptr, // world_to_cam
            &n2c.data[0][0],
            &c2n.data[0][0],
            1e-2f,
            CameraType::Perspective};
        auto negative_ray =
            sample_primary(delta_camera, Vector2{0.5, 0.5});
        auto diff = (sum(positive_ray.org - negative_ray.org) +
                     sum(positive_ray.dir - negative_ray.dir)) /
                    (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, (Real)diff, (Real)d_camera.look[i]);
    }
    for (int i = 0; i < 3; i++) {
        auto delta_up = up;
        delta_up[i] += finite_delta;
        auto delta_camera = Camera{
            1, 1,
            &pos[0],
            &look[0],
            nullptr, // cam_to_world
            nullptr, // world_to_cam
            &delta_up[0],
            &n2c.data[0][0],
            &c2n.data[0][0],
            1e-2f,
            CameraType::Perspective};
        auto positive_ray =
            sample_primary(delta_camera, Vector2{0.5, 0.5});
        delta_up[i] -= 2 * finite_delta;
        delta_camera = Camera{
            1, 1,
            &pos[0],
            &look[0],
            nullptr, // cam_to_world
            nullptr, // world_to_cam
            &delta_up[0],
            &n2c.data[0][0],
            &c2n.data[0][0],
            1e-2f,
            CameraType::Perspective};
        auto negative_ray =
            sample_primary(delta_camera, Vector2{0.5, 0.5});
        auto diff = (sum(positive_ray.org - negative_ray.org) +
                     sum(positive_ray.dir - negative_ray.dir)) /
                    (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, (Real)diff, (Real)d_camera.up[i]);
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            auto delta_camera = camera;
            delta_camera.intrinsic_mat_inv(i, j) += finite_delta;
            auto positive_ray =
                sample_primary(delta_camera, Vector2{0.5, 0.5});
            delta_camera.intrinsic_mat_inv(i, j) -= 2 * finite_delta;
            auto negative_ray =
                sample_primary(delta_camera, Vector2{0.5, 0.5});
            auto diff = (sum(positive_ray.org - negative_ray.org) +
                         sum(positive_ray.dir - negative_ray.dir)) /
                        (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, (Real)diff, (Real)d_camera.intrinsic_mat_inv[3 * i + j]);
        }
    }
}

void test_d_camera_to_screen() {
    auto pos = Vector3f{0, 0, 0};
    auto look = Vector3f{0, 0, 1};
    auto up = Vector3f{0, 1, 0};
    Matrix3x3f n2c = Matrix3x3f::identity();
    Matrix3x3f c2n = Matrix3x3f::identity();
    Camera camera{1, 1,
        &pos[0],
        &look[0],
        &up[0],
        nullptr, // cam_to_world
        nullptr, // world_to_cam
        &n2c.data[0][0],
        &c2n.data[0][0],
        1e-2f,
        CameraType::Perspective};
    auto pt = Vector3{0.5, 0.5, 1.0};
    auto dx = Real(1);
    auto dy = Real(1);
    auto d_pos = Vector3f{0, 0, 0};
    auto d_look = Vector3f{0, 0, 0};
    auto d_up = Vector3f{0, 0, 0};
    Matrix3x3f d_n2c = Matrix3x3f{};
    Matrix3x3f d_c2n = Matrix3x3f{};
    DCamera d_camera{&d_pos[0],
                     &d_look[0],
                     &d_up[0],
                     nullptr, // cam_to_world
                     nullptr, // world_to_cam
                     &d_n2c(0, 0),
                     &d_c2n(0, 0)};
    auto d_pt = Vector3{0, 0, 0};
    d_camera_to_screen(camera, pt, dx, dy, d_camera, d_pt);
    // Compare with central difference
    auto finite_delta = Real(1e-6);
    for (int i = 0; i < 3; i++) {
        auto delta_pos = pos;
        delta_pos[i] += finite_delta;
        auto delta_camera = Camera{
            1, 1,
            &delta_pos[0],
            &look[0],
            &up[0],
            nullptr, // cam_to_world
            nullptr, // world_to_cam
            &n2c.data[0][0],
            &c2n.data[0][0],
            1e-2f,
            CameraType::Perspective};
        auto pxy = camera_to_screen(delta_camera, pt);
        delta_pos[i] -= 2 * finite_delta;
        delta_camera = Camera{
            1, 1,
            &delta_pos[0],
            &look[0],
            &up[0],
            nullptr, // cam_to_world
            nullptr, // world_to_cam
            &n2c.data[0][0],
            &c2n.data[0][0],
            1e-2f,
            CameraType::Perspective};
        auto nxy = camera_to_screen(delta_camera, pt);
        auto diff = (sum(pxy - nxy)) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, (Real)diff, (Real)d_camera.position[i]);
    }
    for (int i = 0; i < 3; i++) {
        auto delta_look = look;
        delta_look[i] += finite_delta;
        auto delta_camera = Camera{
            1, 1,
            &pos[0],
            &delta_look[0],
            &up[0],
            nullptr, // cam_to_world
            nullptr, // world_to_cam
            &n2c.data[0][0],
            &c2n.data[0][0],
            1e-2f,
            CameraType::Perspective};
        auto pxy = camera_to_screen(delta_camera, pt);
        delta_look[i] -= 2 * finite_delta;
        delta_camera = Camera{
            1, 1,
            &pos[0],
            &delta_look[0],
            &up[0],
            nullptr, // cam_to_world
            nullptr, // world_to_cam
            &n2c.data[0][0],
            &c2n.data[0][0],
            1e-2f,
            CameraType::Perspective};
        auto nxy = camera_to_screen(delta_camera, pt);
        auto diff = (sum(pxy - nxy)) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, (Real)diff, (Real)d_camera.look[i]);
    }
    for (int i = 0; i < 3; i++) {
        auto delta_up = up;
        delta_up[i] += finite_delta;
        auto delta_camera = Camera{
            1, 1,
            &pos[0],
            &look[0],
            &delta_up[0],
            nullptr, // cam_to_world
            nullptr, // world_to_cam
            &n2c.data[0][0],
            &c2n.data[0][0],
            1e-2f,
            CameraType::Perspective};
        auto pxy = camera_to_screen(delta_camera, pt);
        delta_up[i] -= 2 * finite_delta;
        delta_camera = Camera{
            1, 1,
            &pos[0],
            &look[0],
            &delta_up[0],
            nullptr, // cam_to_world
            nullptr, // world_to_cam
            &n2c.data[0][0],
            &c2n.data[0][0],
            1e-2f,
            CameraType::Perspective};
        auto nxy = camera_to_screen(delta_camera, pt);
        auto diff = (sum(pxy - nxy)) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, (Real)diff, (Real)d_camera.up[i]);
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            auto delta_camera = camera;
            delta_camera.intrinsic_mat(i, j) += finite_delta;
            auto pxy = camera_to_screen(delta_camera, pt);
            delta_camera.intrinsic_mat(i, j) -= 2 * finite_delta;
            auto nxy = camera_to_screen(delta_camera, pt);
            auto diff = sum(pxy - nxy) / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, (Real)diff, (Real)d_camera.intrinsic_mat[3 * i + j]);
        }
    }
}

void test_camera_derivatives() {
    test_d_sample_primary_rays();
    test_d_camera_to_screen();
}
