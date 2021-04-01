/*
    This file implementats von-Mises Fisher lights, which are essentially
    soft directional lights. Instead of a single specific direction, vMF lights spread the intensity
    in a Gaussian-like region around the primary direction to provide more realistic
    directional lighting, that is also smooth and differentiable.

    This vMF lighting implementation is inspired by, and heavily dependent on, this whitepaper by Wenzel Jakob: 
    https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
 */

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

struct VonMisesFisherLight {
    VonMisesFisherLight() {}
    VonMisesFisherLight(Real kappa,
                   ptr<float> intensity_data,
                   ptr<float> env_to_world,
                   ptr<float> world_to_env,
                   float pdf_norm)
        : kappa(kappa),
          env_to_world(env_to_world.get()),
          world_to_env(world_to_env.get()),
          pdf_norm((Real)pdf_norm) {
              intensity[0] = intensity_data[0];
              intensity[1] = intensity_data[1];
              intensity[2] = intensity_data[2];
          }

    Vector3 intensity;
    Real kappa;
    Matrix4x4 env_to_world;
    Matrix4x4 world_to_env;
    Real pdf_norm;
};

struct DVonMisesFisherLight {
    DVonMisesFisherLight() {}
    DVonMisesFisherLight(
                    ptr<float> kappa,
                    ptr<float> intensity,
                    ptr<float> world_to_env)
        : kappa(kappa.get()),
          intensity(intensity.get()),
          world_to_env(world_to_env.get()) {}

    float *kappa;
    float *intensity;
    float *world_to_env;
};

DEVICE
inline Vector3 get_vmf_value(const VonMisesFisherLight &vmf, 
                             Vector2 uv) {
    const auto k = vmf.kappa;
    const auto norm_const = 2 * M_PI * (1 - exp(-2 * k)) / k;

    return vmf.intensity * exp(k * (uv.y - 1)) / norm_const;
}

DEVICE
inline void d_get_vmf_value(const VonMisesFisherLight &vmf, 
                             Vector2 uv,
                             const Vector3& d_output,
                             Vector2& d_uv,
                             Real& d_kappa,
                             Vector3& d_intensity) {

    const auto k = vmf.kappa;
    const auto norm_const = 2 * M_PI * (1 - exp(-2 * k)) / k;

    d_uv.x = Real(0.f);
    // exp(k * (uv.v - 1)) / norm_const;
    d_uv.y = (k * exp(k * (uv.y - 1)) / norm_const) * dot(vmf.intensity, d_output);

    // TODO: Kappa derivative is not implemented right now!
    d_kappa = Real(0.f);

    d_intensity = d_output * (exp(k * (uv.y - 1)) / norm_const);
}

DEVICE
inline Vector3 vmf_eval(const VonMisesFisherLight &vmf,
                           const Vector3 &dir,
                           const RayDifferential &ray_diff) {
    auto local_dir = normalize(xfm_vector(vmf.world_to_env, dir));
    // Project to spherical coordinate, y is up vector
    auto uv = Vector2{
        atan2(local_dir.x, -local_dir.z) / Real(2 * M_PI),
        local_dir.y
    };

    // Compute ray differentials
    // There is a singularity at (0, 1, 0)
    if (local_dir.y < 1) {
        // TODO: handle scaling in world_to_env
        auto local_dir_dx = xfm_vector(vmf.world_to_env, ray_diff.dir_dx);
        auto local_dir_dy = xfm_vector(vmf.world_to_env, ray_diff.dir_dy);
        auto du_dlocal_dir_x = local_dir.x /
            (Real(2 * M_PI) * (square(local_dir.x) + square(local_dir.z)));
        auto du_dlocal_dir_z = local_dir.z /
            (Real(2 * M_PI) * (square(local_dir.x) + square(local_dir.z)));
        auto du_dxy = Vector2{du_dlocal_dir_x * local_dir_dx.x + du_dlocal_dir_z * local_dir_dx.z,
                              du_dlocal_dir_x * local_dir_dy.x + du_dlocal_dir_z * local_dir_dy.z};
        auto dv_dlocal_dir_y = -1 / (Real(M_PI) * sqrt(1 - square(local_dir.y)));
        auto dv_dxy = Vector2{dv_dlocal_dir_y * local_dir_dx.y,
                              dv_dlocal_dir_y * local_dir_dy.y};
        auto val = get_vmf_value(vmf, uv);
        return val;
    } else {
        // TODO: this is too conservative
        auto du_dxy = Vector2{0, 0};
        auto dv_dxy = Vector2{0, 0};
        auto val = get_vmf_value(vmf, uv);
        return val;
    }
}

DEVICE
inline void d_vmf_eval(const VonMisesFisherLight &vmf_light,
                          const Vector3 &dir,
                          const RayDifferential &ray_diff,
                          const Vector3 &d_output,
                          DVonMisesFisherLight &d_vmf_light,
                          Vector3 &d_dir,
                          RayDifferential &d_ray_diff) {
    auto n_local_dir = xfm_vector(vmf_light.world_to_env, dir);
    auto local_dir = normalize(n_local_dir);
    // Project to spherical coordinate, y is up vector
    auto uv = Vector2{
        atan2(local_dir.x, -local_dir.z) / Real(2 * M_PI),
        //safe_acos(local_dir.y) / Real(M_PI)
        local_dir.y
    };

    // Compute ray differentials
    // TODO: handle scaling in world_to_env
    auto local_dir_dx = xfm_vector(vmf_light.world_to_env, ray_diff.dir_dx);
    auto local_dir_dy = xfm_vector(vmf_light.world_to_env, ray_diff.dir_dy);
    auto du_dlocal_dir_x = local_dir.x /
        (Real(2 * M_PI) * (square(local_dir.x) + square(local_dir.z)));
    auto du_dlocal_dir_z = local_dir.z /
        (Real(2 * M_PI) * (square(local_dir.x) + square(local_dir.z)));
    auto du_dxy = Vector2{du_dlocal_dir_x * local_dir_dx.x + du_dlocal_dir_z * local_dir_dx.z,
                          du_dlocal_dir_x * local_dir_dy.x + du_dlocal_dir_z * local_dir_dy.z};
    auto dv_dlocal_dir_y = -1 / (Real(M_PI) * sqrt(1 - square(local_dir.y)));
    auto dv_dxy = Vector2{dv_dlocal_dir_y * local_dir_dx.y,
                          dv_dlocal_dir_y * local_dir_dy.y};

    // val = get_texture_value(vmf_light.values, uv, du_dxy, dv_dxy)
    auto d_uv = Vector2{0, 0};
    auto d_du_dxy = Vector2{0, 0};
    auto d_dv_dxy = Vector2{0, 0};
    /*d_get_texture_value(vmf_light.values, uv, du_dxy, dv_dxy, d_output,
        d_vmf_light.values, d_uv, d_du_dxy, d_dv_dxy);*/
    Real d_kappa = Real(0.f);
    Vector3 d_intensity = Vector3(0.f, 0.f, 0.f);

    d_get_vmf_value(vmf_light, uv, d_output, d_uv, d_kappa, d_intensity);

    atomic_add(d_vmf_light.kappa, d_kappa);
    atomic_add(d_vmf_light.intensity, d_intensity);

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
    // local_dir_dx = xfm_vector(vmf_light.world_to_env, ray_diff.dir_dx)
    auto d_world_to_env = Matrix4x4{};
    d_xfm_vector(vmf_light.world_to_env, ray_diff.dir_dx, d_local_dir_dx,
        d_world_to_env, d_ray_diff.dir_dx);
    // local_dir_dy = xfm_vector(vmf_light.world_to_env, ray_diff.dir_dy)
    d_xfm_vector(vmf_light.world_to_env, ray_diff.dir_dy, d_local_dir_dy,
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
        d_local_dir.y += d_uv.y;//(- d_uv.y / (sqrt(1 - square(local_dir.y)) * (Real(2 * M_PI))));
    }
    // local_dir = normalize(n_local_dir)
    auto d_n_local_dir = d_normalize(n_local_dir, d_local_dir);
    // n_local_dir = xfm_vector(vmf_light.world_to_env, dir)
    d_xfm_vector(vmf_light.world_to_env, dir, d_n_local_dir, d_world_to_env, d_dir);
    atomic_add(d_vmf_light.world_to_env, d_world_to_env);
}

DEVICE
inline Vector3 vmf_sample(const VonMisesFisherLight &vmf_light, Vector2 sample) {
    auto k = vmf_light.kappa;

    auto phi = sample.y;

    auto sin_phi = sin(phi * 2 * M_PI);
    auto cos_phi = cos(phi * 2 * M_PI);

    auto cos_theta = 1 + log(sample.x + exp(-2*k) * (1 - sample.x)) / k;
    auto sin_theta = sqrt(1 - cos_theta * cos_theta);

    auto local_dir = Vector3{sin_phi * sin_theta, cos_theta, cos_phi * sin_theta};
    return xfm_vector(vmf_light.env_to_world, local_dir);
}

DEVICE
inline Real vmf_pdf(const VonMisesFisherLight &vmf_light, const Vector3& dir) {
    auto k = vmf_light.kappa;

    const auto norm_const = 2 * M_PI * (1 - exp(-2 * k)) / k;

    // Local frame is y-axis oriented.
    auto local_dir = xfm_vector(vmf_light.world_to_env, dir);

    return exp(k * (local_dir.y - 1)) / norm_const;
}
