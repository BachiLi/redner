#pragma once

#include "redner.h"
#include "atomic.h"
#include "vector.h"

struct DistortionParameters {
    bool defined;
    Real k[6]; // Radial distortion
    Real p[2]; // Tangential distortion
};

struct DDistortionParameters {
    float *params;
};

inline
DEVICE
Vector2 distort(const DistortionParameters &param, const Vector2 &pos,
                Vector2 *dx_dpos = nullptr, Vector2 *dy_dpos = nullptr) {
    if (!param.defined) {
        return pos;
    }
    // Brown–Conrady model
    // https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    
    // Scale to [-1, 1]
    auto x = 2.f * (pos.x - 0.5f);
    auto y = 2.f * (pos.y - 0.5f);
    auto r = sqrt(x*x + y*y);
    auto r2 = r * r;
    auto r4 = r2 * r2;
    auto r6 = r4 * r2;
    auto rr_numerator = 1 + param.k[0] * r2 + param.k[1] * r4 + param.k[2] * r6;
    auto rr_denominator = 1 + param.k[3] * r2 + param.k[4] * r4 + param.k[5] * r6;
    auto rr = rr_numerator / rr_denominator;
    auto xx = x * rr + 2 * param.p[0] * x * y + param.p[1] * (r2 + 2 * x * x);
    auto yy = y * rr + param.p[0] * (r2 + 2 * y * y) + 2 * param.p[1] * x * y;
    // Scale back to [0, 1]
    auto xx_ = (xx + 1) / 2;
    auto yy_ = (yy + 1) / 2;

    if (dx_dpos != nullptr && dy_dpos != nullptr) {
        // Forward mode
        auto dpos_x = Vector2{1, 0};
        auto dpos_y = Vector2{0, 1};
        // x = 2.f * (pos.x - 0.5f)
        auto dx = 2 * dpos_x;
        // y = 2.f * (pos.y - 0.5f)
        auto dy = 2 * dpos_y;
        // r = sqrt(x*x + y*y)
        auto dr = (dx * x + dy * y) / r;
        // r2 = r * r
        auto dr2 = 2 * r * dr;
        // r4 = r2 * r2;
        auto dr4 = 2 * r2 * dr2;
        // r6 = r4 * r2;
        auto dr6 = r4 * dr2 + dr4 * r2;
        // rr = (1 + param.k[0] * r2 + param.k[1] * r4 + param.k[2] * r6) /
        //      (1 + param.k[3] * r2 + param.k[4] * r4 + param.k[5] * r6)
        auto drr_numerator = (param.k[0] * dr2 + param.k[1] * dr4 + param.k[2] * dr6);
        auto drr_denominator = (param.k[3] * dr2 + param.k[4] * dr4 + param.k[5] * dr6);
        auto drr = (drr_numerator * rr_denominator - rr_numerator * drr_denominator) /
                   (rr_denominator * rr_denominator);
        // xx = x * rr + 2 * param.p[0] * x * y + param.p[1] * (r2 + 2 * x * x)
        auto dxx = dx * rr + x * drr +
                   2 * param.p[0] * (dx * y + x * dy) +
                   param.p[1] * (dr2 + 4 * dx * x);
        // yy = y * rr + param.p[0] * (r2 + 2 * y * y) + 2 * param.p[1] * x * y
        auto dyy = dy * rr + y * drr +
                   param.p[0] * (dr2 + 4 * dy * y) +
                   2 * param.p[1] * (dx * y + x * dy);
        // xx_ = (xx + 1) / 2
        // yy_ = (yy + 1) / 2
        auto dxx_ = dxx / 2;
        auto dyy_ = dyy / 2;
        *dx_dpos = dxx_;
        *dy_dpos = dyy_;
    }

    return Vector2{xx_, yy_};
}

inline
DEVICE
void d_distort(const DistortionParameters &param, const Vector2 &pos,
               const Vector2 &d_output,
               DDistortionParameters &d_params, Vector2 &d_pos) {
    if (!param.defined) {
        d_pos = d_output;
        return;
    }
    auto x = 2.f * (pos.x - 0.5f);
    auto y = 2.f * (pos.y - 0.5f);
    auto r = sqrt(x*x + y*y);
    auto r2 = r * r;
    auto r4 = r2 * r2;
    auto r6 = r4 * r2;
    auto rr_numerator = 1 + param.k[0] * r2 + param.k[1] * r4 + param.k[2] * r6;
    auto rr_denominator = 1 + param.k[3] * r2 + param.k[4] * r4 + param.k[5] * r6;
    auto rr = rr_numerator / rr_denominator;
    // auto xx = x * rr + 2 * param.p[0] * x * y + param.p[1] * (r2 + 2 * x * x);
    // auto yy = y * rr + param.p[0] * (r2 + 2 * y * y) + 2 * param.p[1] * x * y;
    // auto xx_ = (xx + 1) / 2;
    // auto yy_ = (yy + 1) / 2;

    Real d_k[6] = {0, 0, 0, 0, 0, 0};
    Real d_p[2] = {0, 0};
    auto d_xx_ = d_output.x;
    auto d_yy_ = d_output.y;
    // xx_ = (xx + 1) / 2
    auto d_xx = d_xx_ / 2;
    // yy_ = (yy + 1) / 2
    auto d_yy = d_yy_ / 2;
    // xx = x * rr + 2 * param.p[0] * x * y + param.p[1] * (r2 + 2 * x * x)
    auto d_x = d_xx * (rr + 2 * param.p[0] * y + 4 * param.p[1] * x);
    auto d_rr = d_xx * x;
    auto d_y = d_xx * 2 * param.p[0] * x;
    d_p[0] += d_xx * 2 * x * y;
    d_p[1] += d_xx * (r2 + 2 * x * x);
    auto d_r2 = d_xx * param.p[1];
    // yy = y * rr + param.p[0] * (r2 + 2 * y * y) + 2 * param.p[1] * x * y
    d_y += d_yy * (rr + 4 * param.p[0] * y + 2 * param.p[1] * x);
    d_rr += d_yy * y;
    d_p[0] += d_yy * (r2 + 2 * y * y);
    d_r2 += d_yy * param.p[0];
    d_p[1] += d_yy * 2 * x * y;
    d_x += d_yy * 2 * param.p[1] * y;
    // rr = rr_numerator / rr_denominator
    auto d_rr_numerator = d_rr / rr_denominator;
    auto d_rr_denominator = - d_rr * rr / rr_denominator;
    // rr_numerator = 1 + param.k[0] * r2 + param.k[1] * r4 + param.k[2] * r6
    d_k[0] += d_rr_numerator * r2;
    d_r2 += d_rr_numerator * param.k[0];
    d_k[1] += d_rr_numerator * r4;
    auto d_r4 = d_rr_numerator * param.k[1];
    d_k[2] += d_rr_numerator * r6;
    auto d_r6 = d_rr_numerator * param.k[2];
    // rr_denominator = 1 + param.k[3] * r2 + param.k[4] * r4 + param.k[5] * r6
    d_k[3] += d_rr_denominator * r2;
    d_r2 += d_rr_denominator * param.k[3];
    d_k[4] += d_rr_denominator * r4;
    d_r4 += d_rr_denominator * param.k[4];
    d_k[5] += d_rr_denominator * r6;
    d_r6 += d_rr_denominator * param.k[5];
    // r6 = r4 * r2
    d_r4 += d_r6 * r2;
    d_r2 += d_r6 * r2;
    // r4 = r2 * r2
    d_r2 += 2 * d_r4 * r2;
    // r2 = r * r
    auto d_r = 2 * d_r2 * r;
    // r = sqrt(x*x + y*y)
    d_x += d_r * x / r;
    d_y += d_r * y / r;
    // x = 2.f * (pos.x - 0.5f)
    d_pos.x += d_x * 2;
    // y = 2.f * (pos.y - 0.5f)
    d_pos.y += d_y * 2;

    if (d_params.params != nullptr) {
        // Deposit to d_camera
        for (int i = 0; i < 6; i++) {
            atomic_add(d_params.params[i], d_k[i]);
        }
        atomic_add(d_params.params[6], d_p[0]);
        atomic_add(d_params.params[7], d_p[1]);
    }
}

inline
DEVICE
Vector2 inverse_distort(const DistortionParameters &param, const Vector2 &pos) {
    if (!param.defined) {
        return pos;
    }
    auto result = pos;
    auto err = Real(0);
    // Gauss-Newton iteration
    auto iter = 0;
    do {
        auto drxdp = Vector2{0, 0};
        auto drydp = Vector2{0, 0};
        auto next = distort(param, result, &drxdp, &drydp);
        auto residual = next - pos;
        err = fabs(residual[0]) + fabs(residual[1]);
        // J = {drx/dpx, drx/dpy,
        //      dry/dpx, dry/dpy}
        // invJ = 1/det * {dry/dpy, -drx/dpy,
        //                 -dry/dpx, drx/dpx}
        auto det = drxdp.x * drydp.y - drxdp.y * drydp.x;
        auto inv_det = 1 / det;
        auto invJ0 = inv_det * Vector2{drydp.y, -drxdp.y};
        auto invJ1 = inv_det * Vector2{-drydp.x, drxdp.x};
        result = result - Vector2{dot(invJ0, residual), dot(invJ1, residual)};
    } while (err > Real(1e-3) && iter++ < 1000);
    return result;
}

inline
DEVICE
void d_inverse_distort(const DistortionParameters &param, const Vector2 &pos,
                       const Vector2 &d_output,
                       DDistortionParameters &d_params, Vector2 &d_pos) {
    if (!param.defined) {
        d_pos = d_output;
        return;
    }
    auto result = pos;
    // Gauss-Newton iteration
    auto iter = 0;
    auto err = Real(0);
    do {
        auto drxdp = Vector2{0, 0};
        auto drydp = Vector2{0, 0};
        auto next = distort(param, result, &drxdp, &drydp);
        auto residual = next - pos;
        err = fabs(residual[0]) + fabs(residual[1]);
        // J = {drx/dpx, drx/dpy,
        //      dry/dpx, dry/dpy}
        // invJ = 1/det * {dry/dpy, -drx/dpy,
        //                 -dry/dpx, drx/dpx}
        auto det = drxdp.x * drydp.y - drxdp.y * drydp.x;
        auto inv_det = 1 / det;
        auto invJ0 = inv_det * Vector2{drydp.y, -drxdp.y};
        auto invJ1 = inv_det * Vector2{-drydp.x, drxdp.x};
        result = result - Vector2{dot(invJ0, residual), dot(invJ1, residual)};
    } while (err > Real(1e-3) && iter++ < 1000);

    // Use implicit function theorem for backprop.
    // Let f = distort(param, x) - pos,
    // we have f(param, result, pos) = Vector2(0, 0)
    // Implicit function theorem tells us that
    // d result / d param = -J^{-1} df/d param = - (df/d param)^T J^{-1}^T,
    // where J_{ij} = df_i/dx_j
    // (we do the transpose for backprop, so that we can premultiply 
    //  the adjoint with J^{-1}^T)
    auto dfxdp = Vector2{0, 0};
    auto dfydp = Vector2{0, 0};
    distort(param, result, &dfxdp, &dfydp);
    // J = {dfx/dpx, dfx/dpy,
    //      dfy/dpx, dfy/dpy}
    // invJ = 1/det * {dfy/dpy, -dfx/dpy,
    //                 -dfy/dpx, dfx/dpx}
    // invJT = 1/det * {dfy/dpy, -dfy/dpx,
    //                  -dfx/dpy, dfx/dpx}
    auto det = dfxdp.x * dfydp.y - dfxdp.y * dfydp.x;
    auto inv_det = 1 / det;
    auto invJT0 = inv_det * Vector2{dfydp.y, -dfydp.x};
    auto invJT1 = inv_det * Vector2{-dfxdp.y, dfxdp.x};
    // premultiply the adjoint d_output with -invJT
    auto d_result = -Vector2{dot(invJT0, d_output), dot(invJT1, d_output)};
    auto d_x = Vector2{0, 0}; // unused
    if (d_params.params != nullptr) {
        d_distort(param, result, d_result, d_params, d_x);
    }
    // df/dpos = -1
    d_pos -= d_result;
}

void test_camera_distortion();
