#pragma once

#include "redner.h"
#include "vector.h"
#include "matrix.h"
#include "texture.h"
#include "ray.h"
#include "transform.h"
#include "buffer.h"

#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
struct Scene;

struct EnvironmentMap {
    EnvironmentMap() {}
    EnvironmentMap(const Texture3 &values,
                   ptr<float> env_to_world,
                   ptr<float> world_to_env,
                   ptr<float> sample_cdf_ys,
                   ptr<float> sample_cdf_xs,
                   float pdf_norm,
                   bool directly_visible)
        : values(values),
          env_to_world(env_to_world.get()),
          world_to_env(world_to_env.get()),
          sample_cdf_ys(sample_cdf_ys.get()),
          sample_cdf_xs(sample_cdf_xs.get()),
          pdf_norm((Real)pdf_norm),
          directly_visible(directly_visible) {}

    inline int get_levels() const {
        return values.num_levels;
    }

    inline std::tuple<int, int> get_size(int i) const {
        return std::make_tuple(values.width[i], values.height[i]);
    }

    Texture3 values;
    Matrix4x4 env_to_world;
    Matrix4x4 world_to_env;
    float *sample_cdf_ys;
    float *sample_cdf_xs;
    Real pdf_norm;
    bool directly_visible;
};

struct DEnvironmentMap {
    DEnvironmentMap() {}
    DEnvironmentMap(const Texture3 &values,
                    ptr<float> world_to_env)
        : values(values),
          world_to_env(world_to_env.get()) {}
    Texture3 values;
    float *world_to_env;
};

DEVICE
inline Vector3 envmap_eval(const EnvironmentMap &envmap,
                           const Vector3 &dir,
                           const RayDifferential &ray_diff) {
    auto local_dir = normalize(xfm_vector(envmap.world_to_env, dir));
    // Project to spherical coordinate, y is up vector
    auto uv = Vector2{
        atan2(local_dir.x, -local_dir.z) / Real(2 * M_PI),
        safe_acos(local_dir.y) / Real(M_PI)
    };

    // Compute ray differentials
    // There is a singularity at (0, 1, 0)
    if (local_dir.y < 1) {
        // TODO: handle scaling in world_to_env
        auto local_dir_dx = xfm_vector(envmap.world_to_env, ray_diff.dir_dx);
        auto local_dir_dy = xfm_vector(envmap.world_to_env, ray_diff.dir_dy);
        auto du_dlocal_dir_x = local_dir.x /
            (Real(2 * M_PI) * (square(local_dir.x) + square(local_dir.z)));
        auto du_dlocal_dir_z = local_dir.z /
            (Real(2 * M_PI) * (square(local_dir.x) + square(local_dir.z)));
        auto du_dxy = Vector2{du_dlocal_dir_x * local_dir_dx.x + du_dlocal_dir_z * local_dir_dx.z,
                              du_dlocal_dir_x * local_dir_dy.x + du_dlocal_dir_z * local_dir_dy.z};
        auto dv_dlocal_dir_y = -1 / (Real(M_PI) * sqrt(1 - square(local_dir.y)));
        auto dv_dxy = Vector2{dv_dlocal_dir_y * local_dir_dx.y,
                              dv_dlocal_dir_y * local_dir_dy.y};
        Vector3 val;
        get_texture_value(envmap.values, uv, du_dxy, dv_dxy, &val.x);
        return val;
    } else {
        // TODO: this is too conservative
        auto du_dxy = Vector2{0, 0};
        auto dv_dxy = Vector2{0, 0};
        Vector3 val;
        get_texture_value(envmap.values, uv, du_dxy, dv_dxy, &val.x);
        return val;
    }
}

DEVICE
inline void d_envmap_eval(const EnvironmentMap &envmap,
                          const Vector3 &dir,
                          const RayDifferential &ray_diff,
                          const Vector3 &d_output,
                          DEnvironmentMap &d_envmap,
                          Vector3 &d_dir,
                          RayDifferential &d_ray_diff) {
    auto n_local_dir = xfm_vector(envmap.world_to_env, dir);
    auto local_dir = normalize(n_local_dir);
    // Project to spherical coordinate, y is up vector
    auto uv = Vector2{
        atan2(local_dir.x, -local_dir.z) / Real(2 * M_PI),
        safe_acos(local_dir.y) / Real(M_PI)
    };

    // Compute ray differentials
    // TODO: handle scaling in world_to_env
    auto local_dir_dx = xfm_vector(envmap.world_to_env, ray_diff.dir_dx);
    auto local_dir_dy = xfm_vector(envmap.world_to_env, ray_diff.dir_dy);
    auto du_dlocal_dir_x = local_dir.x /
        (Real(2 * M_PI) * (square(local_dir.x) + square(local_dir.z)));
    auto du_dlocal_dir_z = local_dir.z /
        (Real(2 * M_PI) * (square(local_dir.x) + square(local_dir.z)));
    auto du_dxy = Vector2{du_dlocal_dir_x * local_dir_dx.x + du_dlocal_dir_z * local_dir_dx.z,
                          du_dlocal_dir_x * local_dir_dy.x + du_dlocal_dir_z * local_dir_dy.z};
    auto dv_dlocal_dir_y = -1 / (Real(M_PI) * sqrt(1 - square(local_dir.y)));
    auto dv_dxy = Vector2{dv_dlocal_dir_y * local_dir_dx.y,
                          dv_dlocal_dir_y * local_dir_dy.y};

    // val = get_texture_value(envmap.values, uv, du_dxy, dv_dxy)
    auto d_uv = Vector2{0, 0};
    auto d_du_dxy = Vector2{0, 0};
    auto d_dv_dxy = Vector2{0, 0};
    d_get_texture_value(envmap.values,
                        uv,
                        du_dxy,
                        dv_dxy,
                        &d_output.x,
                        d_envmap.values,
                        d_uv,
                        d_du_dxy,
                        d_dv_dxy);
    // dv_dxy = Vector2{dv_dlocal_dir_y * local_dir_dx.y,
    //                  dv_dlocal_dir_y * local_dir_dy.y}
    auto d_dv_dlocal_dir_y = d_dv_dxy.x * local_dir_dx.y + d_dv_dxy.y * local_dir_dy.y;
    auto d_local_dir_dx = Vector3{Real(0), d_dv_dxy.x * dv_dlocal_dir_y, Real(0)};
    auto d_local_dir_dy = Vector3{Real(0), d_dv_dxy.y * dv_dlocal_dir_y, Real(0)};
    // dv_dlocal_dir_y = -1 / (Real(M_PI) * sqrt(1 - square(local_dir.y)))
    auto d_local_dir = Vector3{Real(0), -d_dv_dlocal_dir_y * local_dir.y /
        (Real(M_PI) * sqrt(1 - square(local_dir.y)) * (1 - square(local_dir.y))), Real(0)};
    // du_dxy = Vector2{du_dlocal_dir_x * local_dir_dx.x + du_dlocal_dir_z * local_dir_dx.z,
    //                  du_dlocal_dir_x * local_dir_dy.x + du_dlocal_dir_z * local_dir_dy.z}
    auto d_du_dlocal_dir_x = d_du_dxy.x * local_dir_dx.x + d_du_dxy.y * local_dir_dy.x;
    auto d_du_dlocal_dir_z = d_du_dxy.x * local_dir_dx.z + d_du_dxy.y * local_dir_dy.z;
    d_local_dir_dx.x += d_du_dxy.x * du_dlocal_dir_x;
    d_local_dir_dx.z += d_du_dxy.x * du_dlocal_dir_z;
    d_local_dir_dy.x += d_du_dxy.y * du_dlocal_dir_x;
    d_local_dir_dy.z += d_du_dxy.y * du_dlocal_dir_z;
    // du_dlocal_dir_z = local_dir.z /
    //     (Real(2 * M_PI) * (square(local_dir.x) + square(local_dir.z)))
    d_local_dir.z += d_du_dlocal_dir_z * (square(local_dir.x) - square(local_dir.z)) /
        (Real(2 * M_PI) * square(square(local_dir.x) + square(local_dir.z)));
    d_local_dir.x -= d_du_dlocal_dir_z * local_dir.x * local_dir.z /
        (Real(2 * M_PI) * square(square(local_dir.x) + square(local_dir.z)));
    // du_dlocal_dir_x = local_dir.x /
    //     (Real(2 * M_PI) * (square(local_dir.x) + square(local_dir.z)))
    d_local_dir.x += d_du_dlocal_dir_x * (square(local_dir.z) - square(local_dir.x)) /
        (Real(2 * M_PI) * square(square(local_dir.x) + square(local_dir.z)));
    d_local_dir.z -= d_du_dlocal_dir_x * local_dir.x * local_dir.z /
        (Real(2 * M_PI) * square(square(local_dir.x) + square(local_dir.z)));
    // local_dir_dx = xfm_vector(envmap.world_to_env, ray_diff.dir_dx)
    auto d_world_to_env = Matrix4x4{};
    d_xfm_vector(envmap.world_to_env, ray_diff.dir_dx, d_local_dir_dx,
        d_world_to_env, d_ray_diff.dir_dx);
    // local_dir_dy = xfm_vector(envmap.world_to_env, ray_diff.dir_dy)
    d_xfm_vector(envmap.world_to_env, ray_diff.dir_dy, d_local_dir_dy,
        d_world_to_env, d_ray_diff.dir_dy);
    // uv = Vector2{
    //     atan2(local_dir.x, -local_dir.z) / Real(2 * M_PI),
    //     acos(local_dir.y) / Real(M_PI)
    // }
    auto x2_z2 = square(local_dir.x) + square(local_dir.z);
    if (x2_z2 > 0.f) {
        d_local_dir.x += (- d_uv.x * local_dir.z / (x2_z2 * Real(2 * M_PI)));
        d_local_dir.z += (- d_uv.x * local_dir.x / (x2_z2 * Real(2 * M_PI)));
    }
    if (local_dir.y < 1.f) {
        d_local_dir.y += (- d_uv.y / (sqrt(1 - square(local_dir.y)) * (Real(2 * M_PI))));
    }
    // local_dir = normalize(n_local_dir)
    auto d_n_local_dir = d_normalize(n_local_dir, d_local_dir);
    // n_local_dir = xfm_vector(envmap.world_to_env, dir)
    d_xfm_vector(envmap.world_to_env, dir, d_n_local_dir, d_world_to_env, d_dir);
    atomic_add(d_envmap.world_to_env, d_world_to_env);
}

DEVICE
inline Real tent_inv_cdf(Real x) {
    if (x < Real(0.5)) {
        return 1 - sqrt(2 * x);
    } else {
        return sqrt(2 * x - 0.5f) - 1;
    }
}

DEVICE
inline Vector3 envmap_sample(const EnvironmentMap &envmap, Vector2 sample) {
    const float *y_ptr =
        thrust::upper_bound(thrust::seq,
            envmap.sample_cdf_ys,
            envmap.sample_cdf_ys + envmap.values.height[0],
            sample.y);
    auto y_pos = clamp((int)(y_ptr - envmap.sample_cdf_ys - 1),
                       0, envmap.values.height[0] - 1);
    if (y_pos < envmap.values.height[0] - 1) {
        sample.y = (sample.y - envmap.sample_cdf_ys[y_pos]) /
            (envmap.sample_cdf_ys[y_pos + 1] - envmap.sample_cdf_ys[y_pos]);
    } else {
        sample.y = (sample.y - envmap.sample_cdf_ys[y_pos]) /
            (1 - envmap.sample_cdf_ys[y_pos]);
    }
    auto sample_cdf_xs = envmap.sample_cdf_xs + y_pos * envmap.values.width[0];
    const float *x_ptr =
        thrust::upper_bound(thrust::seq,
            sample_cdf_xs,
            sample_cdf_xs + envmap.values.width[0],
            sample.x);
    auto x_pos = clamp((int)(x_ptr - sample_cdf_xs - 1),
                       0, envmap.values.width[0] - 1);
    if (x_pos < envmap.values.width[0] - 1) {
        sample.x = (sample.x - sample_cdf_xs[x_pos]) /
            (sample_cdf_xs[x_pos + 1] - sample_cdf_xs[x_pos]);
    } else {
        sample.x = (sample.x - sample_cdf_xs[x_pos]) /
            (1 - sample_cdf_xs[x_pos]);
    }

    // Importance sample bilinear sampling
    auto uv = Vector2{x_pos + tent_inv_cdf(sample.x), y_pos + tent_inv_cdf(sample.y)};
    auto phi = (2 * Real(M_PI) / envmap.values.width[0]) * (uv.x + 0.5f);
    auto theta = (Real(M_PI) / envmap.values.height[0]) * (uv.y + 0.5f);
    auto sin_phi = sin(phi);
    auto cos_phi = cos(phi);
    auto sin_theta = sin(theta);
    auto cos_theta = cos(theta);
    auto local_dir = Vector3{sin_phi * sin_theta, cos_theta, -cos_phi * sin_theta};
    return xfm_vector(envmap.env_to_world, local_dir);
}

DEVICE
inline Real envmap_pdf(const EnvironmentMap &envmap, const Vector3 &dir) {
    auto local_dir = xfm_vector(envmap.world_to_env, dir);
    auto uv = Vector2{
        atan2(local_dir.x, -local_dir.z) / Real(2 * M_PI),
        safe_acos(local_dir.y) / Real(M_PI)
    };
    auto w = envmap.values.width[0];
    auto h = envmap.values.height[0];
    auto x = uv.x * w - 0.5f;
    auto y = uv.y * h - 0.5f;
    auto xfi = modulo((int)floor(x), w);
    auto yfi = modulo((int)floor(y), h);
    auto xci = modulo(xfi + 1, w);
    auto yci = modulo(yfi + 1, h);
    auto dx = x - xfi;
    auto dy = y - yfi;
    auto texels = envmap.values.texels[0];
    auto lum_ff = luminance(
        Vector3f{texels[3 * (yfi * w + xfi) + 0],
                 texels[3 * (yfi * w + xfi) + 1],
                 texels[3 * (yfi * w + xfi) + 2]});
    auto lum_cf = luminance(
        Vector3f{texels[3 * (yfi * w + xci) + 0],
                 texels[3 * (yfi * w + xci) + 1],
                 texels[3 * (yfi * w + xci) + 2]});
    auto lum_fc = luminance(
        Vector3f{texels[3 * (yci * w + xfi) + 0],
                 texels[3 * (yci * w + xfi) + 1],
                 texels[3 * (yci * w + xfi) + 2]});
    auto lum_cc = luminance(
        Vector3f{texels[3 * (yci * w + xci) + 0],
                 texels[3 * (yci * w + xci) + 1],
                 texels[3 * (yci * w + xci) + 2]});
    auto lum_fy = lum_ff * (1.f - dx) * (1.f - dy) +
                  lum_cf *        dx  * (1.f - dy);
    auto lum_cy = lum_fc * (1.f - dx) *        dy  +
                  lum_cc *        dx  *        dy;
    auto sin_theta = sqrt(max(1 - square(local_dir.y), Real(0)));
    if (sin_theta == 0.f) {
        return 0.f;
    }
    auto sin_theta_fy = fabs(sin(Real(M_PI) * (yfi + 0.5f) / h));
    auto sin_theta_cy = fabs(sin(Real(M_PI) * (yci + 0.5f) / h));
    return envmap.pdf_norm * fabs(lum_fy * sin_theta_fy + lum_cy * sin_theta_cy) / sin_theta;
}
