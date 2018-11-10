#include "active_pixels.h"
#include "test_utils.h"
#include "thrust_utils.h"

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/remove.h>

struct is_invalid_ray {
    DEVICE bool operator()(int idx) {
        return is_zero(rays[idx].dir);
    }

    const Ray *rays = nullptr;
};

void init_active_pixels(const BufferView<Ray> &rays,
                        BufferView<int> active_pixels,
                        bool use_gpu) {
    assert(rays.size() == active_pixels.size());
    DISPATCH(use_gpu, thrust::sequence, active_pixels.begin(), active_pixels.end());
    auto op = is_invalid_ray{rays.begin()};
    DISPATCH(use_gpu, thrust::remove_if,
        active_pixels.begin(), active_pixels.end(),
        active_pixels.begin(), op);
}

struct is_valid_intersection {
    DEVICE bool operator()(int pixel_id) {
        return isects[pixel_id].valid();
    }

    const Intersection *isects = nullptr;
};

void update_active_pixels(const BufferView<int> &active_pixels,
                          const BufferView<Intersection> &isects,
                          BufferView<int> &new_active_pixels,
                          bool use_gpu) {
    auto op = is_valid_intersection{isects.begin()};
    auto new_end = DISPATCH(use_gpu, thrust::copy_if,
        active_pixels.begin(), active_pixels.end(),
        new_active_pixels.begin(), op);
    new_active_pixels.count = new_end - new_active_pixels.begin();
}

void test_active_pixels(bool use_gpu) {
    auto num_pixels = 1024;
    auto rays_buffer = Buffer<Ray>(use_gpu, num_pixels);
    auto rays = rays_buffer.view(0, num_pixels);
    for (int i = 0; i < num_pixels; i++) {
        rays[i] = Ray{Vector3{0, 0, 0}, Vector3{0, 0, 1}};   
    }
    auto active_pixels_buffer = Buffer<int>(use_gpu, num_pixels);
    auto active_pixels = active_pixels_buffer.view(0, num_pixels);
    init_active_pixels(rays, active_pixels, use_gpu);
    equal_or_error(__FILE__, __LINE__, num_pixels, active_pixels.size());
    auto isects_buffer = Buffer<Intersection>(use_gpu, num_pixels);
    auto isects = isects_buffer.view(0, num_pixels);
    for (int i = 0; i < num_pixels; i++) {
        if (i % 2 == 0) {
            isects[i] = Intersection{0, 0};
        } else {
            isects[i] = Intersection{-1, -1};
        }
    }
    update_active_pixels(active_pixels,
                         isects,
                         active_pixels,
                         use_gpu);
    equal_or_error(__FILE__, __LINE__, num_pixels / 2, active_pixels.size());
}
