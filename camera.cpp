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

        rays[idx] = sample_primary(camera, screen_pos);
    }

    const Camera camera = Camera{};
    const CameraSample *samples = nullptr;
    Ray *rays = nullptr;
};

void sample_primary_rays(const Camera &camera,
                         const BufferView<CameraSample> &samples,
                         BufferView<Ray> rays,
                         bool use_gpu) {
    parallel_for(primary_ray_sampler{camera, samples.begin(), rays.begin()},
        samples.size(), use_gpu);
}

void accumulate_camera(const DCameraInst &d_camera_inst,
                       DCamera &d_camera,
                       bool use_gpu) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            d_camera.cam_to_world[4 * i + j] +=
                d_camera_inst.cam_to_world(i, j);
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            d_camera.world_to_cam[4 * i + j] +=
                d_camera_inst.world_to_cam(i, j);
        }
    }
    *(d_camera.fov_factor) += d_camera_inst.fov_factor;
}

void test_sample_primary_rays(bool use_gpu) {
    // Let's have a perspective camera with 1x1 pixel, 
    // with identity to world matrix,
    // fov 45 degree
    Matrix4x4f c2w = Matrix4x4f::identity();
    Matrix4x4f w2c = Matrix4x4f::identity();
    Camera camera{1, 1,
        &c2w.data[0][0],
        &w2c.data[0][0],
        1,
        1e-2f,
        false};
    parallel_init();

    // Sample from the center of pixel
    Buffer<CameraSample> samples(use_gpu, 1);
    samples[0].xy = Vector2{0.5f, 0.5f};
    Buffer<Ray> rays(use_gpu, 1);
    sample_primary_rays(camera, samples.view(0, 1), rays.view(0, 1), use_gpu);
    cuda_synchronize();

    equal_or_error(__FILE__, __LINE__, rays[0].org, Vector3{0, 0, 0});
    equal_or_error(__FILE__, __LINE__, rays[0].dir, Vector3{0, 0, 1});

    parallel_cleanup();
}

void test_d_sample_primary_rays() {
    Matrix4x4f c2w = Matrix4x4f::identity();
    Matrix4x4f w2c = Matrix4x4f::identity();
    Camera camera{1, 1,
        &c2w.data[0][0],
        &w2c.data[0][0],
        1,
        1e-2f,
        false};
    DCameraInst d_camera;
    DRay d_ray{Vector3{1, 1, 1}, Vector3{1, 1, 1}};
    d_sample_primary_ray(camera,
                         Vector2{0.5, 0.5},
                         d_ray,
                         d_camera);
    // Compare with central difference
    auto finite_delta = Real(1e-6);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            auto delta_camera = camera;
            delta_camera.cam_to_world(i, j) += finite_delta;
            auto positive_ray =
                sample_primary(delta_camera, Vector2{0.5, 0.5});
            delta_camera.cam_to_world(i, j) -= 2 * finite_delta;
            auto negative_ray =
                sample_primary(delta_camera, Vector2{0.5, 0.5});
            auto diff = (sum(positive_ray.org - negative_ray.org) +
                         sum(positive_ray.dir - negative_ray.dir)) /
                        (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, diff,
                d_camera.cam_to_world(i, j));
        }
    }
    auto delta_camera = camera;
    delta_camera.fov_factor += finite_delta;
    auto positive_ray = sample_primary(delta_camera, Vector2{0.5, 0.5});
    delta_camera.fov_factor -= 2 * finite_delta;
    auto negative_ray = sample_primary(delta_camera, Vector2{0.5, 0.5});
    auto diff = (sum(positive_ray.org - negative_ray.org) +
                 sum(positive_ray.dir - negative_ray.dir)) /
                (2 * finite_delta);
    equal_or_error(__FILE__, __LINE__, diff, d_camera.fov_factor);
}

void test_d_camera_to_screen() {
    Matrix4x4f c2w = Matrix4x4f::identity();
    Matrix4x4f w2c = Matrix4x4f::identity();
    Camera camera{1, 1,
        &c2w.data[0][0],
        &w2c.data[0][0],
        1,
        1e-2f,
        false};
    auto pt = Vector3{0.5, 0.5, 1.0};
    auto dx = Real(1);
    auto dy = Real(1);
    auto d_camera = DCameraInst{};
    auto d_pt = Vector3{0, 0, 0};
    d_camera_to_screen(camera, pt, dx, dy,
        d_camera, d_pt);
    // Compare with central difference
    auto finite_delta = Real(1e-6);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            auto delta_camera = camera;
            delta_camera.cam_to_world(i, j) += finite_delta;
            auto pxy = camera_to_screen(delta_camera, pt);
            delta_camera.cam_to_world(i, j) -= 2 * finite_delta;
            auto nxy = camera_to_screen(delta_camera, pt);
            auto diff = sum(pxy - nxy) / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, diff,
                d_camera.cam_to_world(i, j));
        }
    }
    auto delta_camera = camera;
    delta_camera.fov_factor += finite_delta;
    auto pxy = camera_to_screen(delta_camera, pt);
    delta_camera.fov_factor -= 2 * finite_delta;
    auto nxy = camera_to_screen(delta_camera, pt);
    auto diff = sum(pxy - nxy) / (2 * finite_delta);
    equal_or_error(__FILE__, __LINE__, diff, d_camera.fov_factor);
}

void test_camera_derivatives() {
    test_d_sample_primary_rays();
    test_d_camera_to_screen();
}
