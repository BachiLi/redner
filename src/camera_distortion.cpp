#include "camera_distortion.h"
#include "test_utils.h"

void test_inverse_distort() {
    // Check if inverse_distort(distort(x)) = x
    DistortionParameters param;
    param.defined = true;
    param.k[0] = 0.5f;
    param.k[1] = -0.2f;
    param.k[2] = 0.3f;
    param.k[3] = 0.1f;
    param.k[4] = -0.1f;
    param.k[5] = 0.01f;
    param.p[0] = 0.3f;
    param.p[1] = -0.2f;
    auto pos = Vector2{0.8f, 0.1f};
    auto distorted = distort(param, pos);
    auto undistorted = inverse_distort(param, distorted);
    equal_or_error(__FILE__, __LINE__, pos, undistorted);
}

void test_d_distort() {
    DistortionParameters param;
    param.defined = true;
    param.k[0] = 0.5f;
    param.k[1] = -0.2f;
    param.k[2] = 0.3f;
    param.k[3] = 0.1f;
    param.k[4] = -0.1f;
    param.k[5] = 0.01f;
    param.p[0] = 0.3f;
    param.p[1] = -0.2f;

    auto pos = Vector2{0.8f, 0.1f};
    float d_params_buf[8];
    for (int i = 0; i < 8; i++) {
        d_params_buf[i] = 0;
    }
    DDistortionParameters d_params{d_params_buf};
    auto d_pos = Vector2{0, 0};
    d_distort(param, pos, Vector2{1, 1}, d_params, d_pos);
    // Compare with finite difference
    auto finite_delta = Real(1e-6);
    for (int i = 0; i < 6; i++) {
        DistortionParameters delta_p = param;
        delta_p.k[i] += finite_delta;
        auto positive_pos = distort(delta_p, pos);
        delta_p.k[i] -= 2 * finite_delta;
        auto negative_pos = distort(delta_p, pos);
        auto diff = (sum(positive_pos - negative_pos)) /
                    (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, (Real)diff, (Real)d_params_buf[i]);
    }
    for (int i = 0; i < 2; i++) {
        DistortionParameters delta_p = param;
        delta_p.p[i] += finite_delta;
        auto positive_pos = distort(delta_p, pos);
        delta_p.p[i] -= 2 * finite_delta;
        auto negative_pos = distort(delta_p, pos);
        auto diff = (sum(positive_pos - negative_pos)) /
                    (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, (Real)diff, (Real)d_params_buf[i + 6]);
    }
    for (int i = 0; i < 2; i++) {
        auto delta_pos = pos;
        delta_pos[i] += finite_delta;
        auto positive_pos = distort(param, delta_pos);
        delta_pos[i] -= 2 * finite_delta;
        auto negative_pos = distort(param, delta_pos);
        auto diff = (sum(positive_pos - negative_pos)) /
                    (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, (Real)diff, (Real)d_pos[i]);
    }
}

void test_d_inverse_distort() {
    DistortionParameters param;
    param.defined = true;
    param.k[0] = 0.5f;
    param.k[1] = -0.2f;
    param.k[2] = 0.3f;
    param.k[3] = 0.1f;
    param.k[4] = -0.1f;
    param.k[5] = 0.01f;
    param.p[0] = 0.3f;
    param.p[1] = -0.2f;

    auto pos = Vector2{0.8f, 0.1f};
    float d_params_buf[8];
    for (int i = 0; i < 8; i++) {
        d_params_buf[i] = 0;
    }
    DDistortionParameters d_params{d_params_buf};
    auto d_pos = Vector2{0, 0};
    d_inverse_distort(param, pos, Vector2{1, 1}, d_params, d_pos);
    // Compare with finite difference
    auto finite_delta = Real(1e-6);
    for (int i = 0; i < 6; i++) {
        DistortionParameters delta_p = param;
        delta_p.k[i] += finite_delta;
        auto positive_pos = inverse_distort(delta_p, pos);
        delta_p.k[i] -= 2 * finite_delta;
        auto negative_pos = inverse_distort(delta_p, pos);
        auto diff = (sum(positive_pos - negative_pos)) /
                    (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, (Real)diff, (Real)d_params_buf[i]);
    }
    for (int i = 0; i < 2; i++) {
        DistortionParameters delta_p = param;
        delta_p.p[i] += finite_delta;
        auto positive_pos = inverse_distort(delta_p, pos);
        delta_p.p[i] -= 2 * finite_delta;
        auto negative_pos = inverse_distort(delta_p, pos);
        auto diff = (sum(positive_pos - negative_pos)) /
                    (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, (Real)diff, (Real)d_params_buf[i + 6]);
    }
    for (int i = 0; i < 2; i++) {
        auto delta_pos = pos;
        delta_pos[i] += finite_delta;
        auto positive_pos = inverse_distort(param, delta_pos);
        delta_pos[i] -= 2 * finite_delta;
        auto negative_pos = inverse_distort(param, delta_pos);
        auto diff = (sum(positive_pos - negative_pos)) /
                    (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, (Real)diff, (Real)d_pos[i]);
    }
}

void test_camera_distortion() {
    test_inverse_distort();
    test_d_distort();
    test_d_inverse_distort();
}
