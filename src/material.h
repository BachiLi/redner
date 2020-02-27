#pragma once

#include "redner.h"
#include "vector.h"
#include "intersection.h"
#include "buffer.h"
#include "ptr.h"
#include "texture.h"

#include <tuple>

struct Material {
    Material() {}

    Material(Texture3 diffuse_reflectance,
             Texture3 specular_reflectance,
             Texture1 roughness,
             TextureN generic_texture,
             Texture3 normal_map,
             bool compute_specular_lighting,
             bool two_sided,
             bool use_vertex_color)
        : diffuse_reflectance(diffuse_reflectance),
          specular_reflectance(specular_reflectance),
          roughness(roughness),
          generic_texture(generic_texture),
          normal_map(normal_map),
          compute_specular_lighting(compute_specular_lighting),
          two_sided(two_sided),
          use_vertex_color(use_vertex_color) {}

    inline int get_diffuse_levels() const {
        return diffuse_reflectance.num_levels;
    }

    inline std::tuple<int, int> get_diffuse_size(int i) const {
        return std::make_tuple(
            diffuse_reflectance.width[i],
            diffuse_reflectance.height[i]);
    }

    inline int get_specular_levels() const {
        return specular_reflectance.num_levels;
    }

    inline std::tuple<int, int> get_specular_size(int i) const {
        return std::make_tuple(
            specular_reflectance.width[i],
            specular_reflectance.height[i]);
    }

    inline int get_roughness_levels() const {
        return roughness.num_levels;
    }

    inline std::tuple<int, int> get_roughness_size(int i) const {
        return std::make_tuple(
            roughness.width[i],
            roughness.height[i]);
    }

    inline int get_generic_levels() const {
        return generic_texture.num_levels;
    }

    inline std::tuple<int, int, int> get_generic_size(int i) const {
        return std::make_tuple(
            generic_texture.channels,
            generic_texture.width[i],
            generic_texture.height[i]);
    }

    inline int get_normal_map_levels() const {
        return normal_map.num_levels;
    }

    inline std::tuple<int, int> get_normal_map_size(int i) const {
        return std::make_tuple(
            normal_map.width[i],
            normal_map.height[i]);
    }

    Texture3 diffuse_reflectance;
    Texture3 specular_reflectance;
    Texture1 roughness;
    TextureN generic_texture;
    Texture3 normal_map;
    bool compute_specular_lighting;
    bool two_sided;
    bool use_vertex_color;
};

struct DMaterial {
    Texture3 diffuse_reflectance;
    Texture3 specular_reflectance;
    Texture1 roughness;
    TextureN generic_texture;
    Texture3 normal_map;
};

template <typename T>
struct TBSDFSample {
    TVector2<T> uv;
    T w;
};

using BSDFSample = TBSDFSample<Real>;

DEVICE
inline Vector3 get_diffuse_reflectance(const Material &material,
                                       const SurfacePoint &shading_point) {
    Vector3 ret;
    get_texture_value(material.diffuse_reflectance,
                      shading_point.uv,
                      shading_point.du_dxy,
                      shading_point.dv_dxy,
                      &ret.x);
    return ret;
}

DEVICE
inline void d_get_diffuse_reflectance(const Material &material,
                                      const SurfacePoint &shading_point,
                                      const Vector3 &d_output,
                                      Texture3 &d_texture,
                                      SurfacePoint &d_shading_point) {
    d_get_texture_value(material.diffuse_reflectance,
                        shading_point.uv,
                        shading_point.du_dxy,
                        shading_point.dv_dxy,
                        &d_output.x,
                        d_texture,
                        d_shading_point.uv,
                        d_shading_point.du_dxy,
                        d_shading_point.dv_dxy);
}

DEVICE
inline Vector3 get_specular_reflectance(const Material &material,
                                        const SurfacePoint &shading_point) {
    Vector3 ret;
    get_texture_value(material.specular_reflectance,
                      shading_point.uv,
                      shading_point.du_dxy,
                      shading_point.dv_dxy,
                      &ret.x);
    return ret;
}

DEVICE
inline void d_get_specular_reflectance(const Material &material,
                                       const SurfacePoint &shading_point,
                                       const Vector3 &d_output,
                                       Texture3 &d_texture,
                                       SurfacePoint &d_shading_point) {
    d_get_texture_value(material.specular_reflectance,
                        shading_point.uv,
                        shading_point.du_dxy,
                        shading_point.dv_dxy,
                        &d_output.x,
                        d_texture,
                        d_shading_point.uv,
                        d_shading_point.du_dxy,
                        d_shading_point.dv_dxy);
}

DEVICE
inline Real get_roughness(const Material &material,
                          const SurfacePoint &shading_point) {
    Real ret;
    get_texture_value(material.roughness,
                      shading_point.uv,
                      shading_point.du_dxy,
                      shading_point.dv_dxy,
                      &ret);
    return ret;
}

DEVICE
inline void d_get_roughness(const Material &material,
                            const SurfacePoint &shading_point,
                            const Real d_output,
                            Texture1 &d_texture,
                            SurfacePoint &d_shading_point) {
    d_get_texture_value(material.roughness,
                        shading_point.uv,
                        shading_point.du_dxy,
                        shading_point.dv_dxy,
                        &d_output,
                        d_texture,
                        d_shading_point.uv,
                        d_shading_point.du_dxy,
                        d_shading_point.dv_dxy);
}

DEVICE
inline void get_generic_texture(const Material &material,
                                const SurfacePoint &shading_point,
                                Real *output) {
    return get_texture_value(material.generic_texture,
                             shading_point.uv,
                             shading_point.du_dxy,
                             shading_point.dv_dxy,
                             output);
}

DEVICE
inline void d_get_generic_texture(const Material &material,
                                  const SurfacePoint &shading_point,
                                  const Real *d_output,
                                  TextureN &d_texture,
                                  SurfacePoint &d_shading_point) {
    d_get_texture_value(material.generic_texture,
                        shading_point.uv,
                        shading_point.du_dxy,
                        shading_point.dv_dxy,
                        d_output,
                        d_texture,
                        d_shading_point.uv,
                        d_shading_point.du_dxy,
                        d_shading_point.dv_dxy);
}


DEVICE
inline bool has_normal_map(const Material &material) {
    return material.normal_map.num_levels > 0;
}

DEVICE
inline Vector3 get_normal(const Material &material,
                          const SurfacePoint &shading_point) {
    Vector3 ret;
    get_texture_value(material.normal_map,
                      shading_point.uv,
                      shading_point.du_dxy,
                      shading_point.dv_dxy,
                      &ret.x);
    return ret;
}

DEVICE
inline void d_get_normal(const Material &material,
                         const SurfacePoint &shading_point,
                         const Vector3 &d_output,
                         Texture3 &d_texture,
                         SurfacePoint &d_shading_point) {
    d_get_texture_value(material.normal_map,
                        shading_point.uv,
                        shading_point.du_dxy,
                        shading_point.dv_dxy,
                        &d_output.x,
                        d_texture,
                        d_shading_point.uv,
                        d_shading_point.du_dxy,
                        d_shading_point.dv_dxy);
}

// y = 2 / x - 2
// y + 2 = 2 / x
// x = 2 / (y + 2)
DEVICE
inline Real roughness_to_phong(Real roughness) {
    return max(2.f / roughness - 2.f, Real(0));
}

DEVICE
inline Real d_roughness_to_phong(Real roughness, Real d_exponent) {
    return (roughness > 0 && roughness <= 1.f) ?
        -2.f * d_exponent / square(roughness) : 0.f;
}

DEVICE
inline Frame perturb_shading_frame(const Material &material,
                                   const SurfacePoint &shading_point) {
    auto n_local = 2 * get_normal(material, shading_point) - 1;
    auto n_world = to_world(shading_point.shading_frame, n_local);
    auto perturb_n = normalize(n_world);
    auto dot_pn_dpdu = dot(perturb_n, shading_point.dpdu);
    auto perturb_x = normalize(
        shading_point.dpdu - perturb_n * dot_pn_dpdu);
    auto perturb_y = cross(perturb_n, perturb_x);
    return Frame(perturb_x, perturb_y, perturb_n);
}

DEVICE
inline void d_perturb_shading_frame(const Material &material,
                                    const SurfacePoint &shading_point,
                                    const Frame &d_frame,
                                    DMaterial &d_material,
                                    SurfacePoint &d_shading_point) {
    // Perturb shading frame
    auto n_local = 2 * get_normal(material, shading_point) - 1;
    auto n_world = to_world(shading_point.shading_frame, n_local);
    auto perturb_n = normalize(n_world);
    auto dot_pn_dpdu = dot(perturb_n, shading_point.dpdu);
    auto npx = shading_point.dpdu - perturb_n * dot_pn_dpdu;
    auto perturb_x = normalize(npx);
    // perturb_y = cross(perturb_n, perturb_x)
    // return Frame(perturb_x, perturb_y, perturb_n)
    auto d_perturb_n = d_frame.n;
    auto d_perturb_x = d_frame.x;
    auto d_perturb_y = d_frame.y;
    // perturb_y = cross(perturb_n, perturb_x)
    d_cross(perturb_n, perturb_x, d_perturb_y, d_perturb_n, d_perturb_x);
    // perturb_x = normalize(npx)
    auto d_npx = d_normalize(npx, d_perturb_x);
    // npx = shading_point.dpdu - perturb_n * dot(perturb_n, shading_point.dpdu)
    d_shading_point.dpdu += d_npx;
    d_perturb_n -= d_npx * dot_pn_dpdu;
    auto d_dot_pn_dpdu = -sum(d_npx * perturb_n);
    // dot_pn_dpdu = dot(perturb_n, shading_point.dpdu)
    d_perturb_n += d_dot_pn_dpdu * shading_point.dpdu;
    d_shading_point.dpdu += d_dot_pn_dpdu * perturb_n;

    // perturb_n = normalize(n_world)
    auto d_n_world = d_normalize(n_world, d_perturb_n);
    // n_world = to_world(shading_point.shading_frame, n_local)
    auto d_local_n = Vector3{0, 0, 0};
    d_to_world(shading_point.shading_frame, n_local, d_n_world,
               d_shading_point.shading_frame, d_local_n);
    // n_local = 2 * get_normal(material, shading_point) - 1
    d_get_normal(material,
                 shading_point,
                 2 * d_local_n,
                 d_material.normal_map,
                 d_shading_point);
}

// Specialized version
DEVICE
inline void d_perturb_shading_frame(const Material &material,
                                    const SurfacePoint &shading_point,
                                    const Vector3 &d_n,
                                    DMaterial &d_material,
                                    SurfacePoint &d_shading_point) {
    // Perturb shading frame
    auto n_local = 2 * get_normal(material, shading_point) - 1;
    auto n_world = to_world(shading_point.shading_frame, n_local);
    // perturb_n = normalize(n_world)
    auto d_n_world = d_normalize(n_world, d_n);
    auto d_local_n = Vector3{0, 0, 0};
    d_to_world(shading_point.shading_frame, n_local, d_n_world,
        d_shading_point.shading_frame, d_local_n);
    // local_n = 2 * get_normal(material, shading_point) - 1
    d_get_normal(material,
                 shading_point,
                 2 * d_local_n,
                 d_material.normal_map,
                 d_shading_point);
}

DEVICE
inline
Vector3 bsdf(const Material &material,
             const SurfacePoint &shading_point,
             const Vector3 &wi,
             const Vector3 &wo,
             const Real min_roughness) {
    // To address the discrepancy between shading normal and geometry normal,
    // we use the strategy recommended by Veach: we define the BSDFs
    // over the whole spherical domain, instead of just the hemisphere domain.
    // See Chapter 5.3.4.1 in Veach's thesis.
    // It is also important to only use geometry normal to reject samples,
    // since our edge sampling only detect geometry discontinuities.
    auto shading_frame = shading_point.shading_frame;
    auto geom_n = shading_point.geom_normal;
    if (has_normal_map(material)) {
        // Perturb shading frame
        shading_frame = perturb_shading_frame(material, shading_point);
    }
    // Flip geometry normal to the same side of the shading frame
    if (dot(geom_n, shading_frame.n) < 0) {
        geom_n = -geom_n;
    }
    auto geom_wi = dot(geom_n, wi);
    auto geom_wo = dot(geom_n, wo);
    auto shading_wi = fabs(dot(shading_frame.n, wi));
    auto shading_wo = fabs(dot(shading_frame.n, wo));
    if (geom_wi * geom_wo < 0) {
        // wi & wo are at different sides of the geometry
        // TODO: implement BTDF
        return Vector3{0, 0, 0};
    }
    if (!material.two_sided) {
        // The surface doesn't reflect light on the other side of
        // the geometry normal.
        // Otherwise two sided means both sides are the same BRDF.
        if (geom_wi < 0 && geom_wo < 0) {
            return Vector3{0, 0, 0};
        }
    }
    if (shading_wi == 0 || shading_wo <= 1e-3f || fabs(geom_wo) <= 1e-3f) {
        // XXX: kind of hacky. We ignore extreme grazing angles
        // for numerical robustness
        return Vector3{0, 0, 0};
    }

    auto diffuse_reflectance_ = material.use_vertex_color ?
        shading_point.color : get_diffuse_reflectance(material, shading_point);
    auto specular_reflectance_ = material.use_vertex_color ?
        Vector3{0, 0, 0} : get_specular_reflectance(material, shading_point);
    auto diffuse_reflectance = max(diffuse_reflectance_, Vector3{0, 0, 0});
    auto specular_reflectance = max(specular_reflectance_, Vector3{0, 0, 0});
    auto roughness = max(get_roughness(material, shading_point), min_roughness);
    auto diffuse_contrib = diffuse_reflectance * shading_wo / Real(M_PI);
    auto specular_contrib = Vector3{0, 0, 0};
    if (material.compute_specular_lighting && !material.use_vertex_color) {
        // blinn-phong BRDF
        // half-vector
        auto m = normalize(wi + wo);
        auto m_local = to_local(shading_frame, m);
        if (material.two_sided) {
            if (m_local[2] < 0) {
                m_local = -m_local;
            }
        }
        if (m_local[2] > 0.f) {
            auto phong_exponent = roughness_to_phong(roughness);
            auto D = pow(max(m_local[2], Real(0)), phong_exponent) *
                (phong_exponent + 2.f) / Real(2 * M_PI);
            auto smithG1 = [&](const Vector3 &v) -> Real {
                auto cos_theta = dot(v, shading_frame.n);
                // tan^2 + 1 = 1/cos^2
                auto tan_theta =
                    sqrt(max(1.f / (cos_theta * cos_theta) - 1.f, Real(0)));
                if (tan_theta == 0.0f) {
                    return 1;
                }
                auto alpha = sqrt(roughness);
                auto a = 1.f / (alpha * tan_theta);
                if (a >= 1.6f) {
                    return 1;
                }
                auto a_sqr = a*a;
                return (3.535f * a + 2.181f * a_sqr)
                     / (1.0f + 2.276f * a + 2.577f * a_sqr);
            };
            auto G = smithG1(wi) * smithG1(wo);
            auto cos_theta_d = fabs(dot(m, wo));
            // Schlick's approximation
            auto F = specular_reflectance +
                (1.f - specular_reflectance) *
                pow(max(Real(1) - cos_theta_d, Real(0)), Real(5));
            specular_contrib = F * D * G / (4.f * shading_wi);
        }
    }
    return diffuse_contrib + specular_contrib;
}

DEVICE
inline
void d_bsdf(const Material &material,
            const SurfacePoint &shading_point,
            const Vector3 &wi,
            const Vector3 &wo,
            const Real min_roughness,
            const Vector3 &d_output,
            DMaterial &d_material,
            SurfacePoint &d_shading_point,
            Vector3 &d_wi,
            Vector3 &d_wo) {
    auto shading_frame = shading_point.shading_frame;
    if (has_normal_map(material)) {
        // Perturb shading frame
        shading_frame = perturb_shading_frame(material, shading_point);
    }

    auto geom_n = shading_point.geom_normal;
    // Flip geometry normal to the same side of the shading frame
    if (dot(geom_n, shading_frame.n) < 0) {
        geom_n = -geom_n;
    }

    auto d_n = Vector3{0, 0, 0};
    auto geom_wi = dot(geom_n, wi);
    auto geom_wo = dot(geom_n, wo);
    auto shading_wi = fabs(dot(shading_frame.n, wi));
    auto shading_wo = fabs(dot(shading_frame.n, wo));
    if (geom_wi * geom_wo < 0) {
        // wi & wo are at different sides of the geometry
        // TODO: implement BTDF
        return;
    }
    if (!material.two_sided) {
        // The surface doesn't reflect light on the other side of
        // the geometry normal.
        // Otherwise two sided means both sides are the same BRDF.
        if (geom_wi < 0 && geom_wo < 0) {
            return;
        }
    }
    if (shading_wi == 0 || shading_wo <= 1e-3f || fabs(geom_wo) <= 1e-3f) {
        // XXX: kind of hacky. We ignore extreme grazing angles
        // for numerical robustness
        return;
    }

    auto diffuse_reflectance_ = material.use_vertex_color ?
        shading_point.color : get_diffuse_reflectance(material, shading_point);
    auto diffuse_reflectance = max(diffuse_reflectance_, Vector3{0, 0, 0});
    // diffuse_contrib = diffuse_reflectance * shading_wo / Real(M_PI)
    auto d_diffuse_reflectance = d_output * (shading_wo / Real(M_PI));
    // diffuse_reflectance = max(diffuse_reflectance_, Vector3{0, 0, 0})
    // HACK: the "correct gradient" is the following, but it makes negative
    //       reflectance never going to come back.
    // auto d_diffuse_reflectance_ = Vector3{
    //     diffuse_reflectance_.x >= 0 ? d_diffuse_reflectance.x : Real(0),
    //     diffuse_reflectance_.y >= 0 ? d_diffuse_reflectance.y : Real(0),
    //     diffuse_reflectance_.z >= 0 ? d_diffuse_reflectance.z : Real(0)
    // };
    // HACK (continued): instead we just use the gradients before the max.
    if (material.use_vertex_color) {
        d_shading_point.color += d_diffuse_reflectance;
    } else {
        d_get_diffuse_reflectance(material, shading_point, d_diffuse_reflectance,
                                  d_material.diffuse_reflectance, d_shading_point);
    }
    auto d_shading_wo = sum(d_output * diffuse_reflectance) / Real(M_PI);
    // shading_wo = fabs(dot(shading_frame.n, wo))
    if (dot(shading_frame.n, wo) < 0)  {
        d_shading_wo = -d_shading_wo;
    }
    d_wo += shading_frame.n * d_shading_wo;
    d_n += wo * d_shading_wo;

    auto specular_reflectance_ = material.use_vertex_color ?
        Vector3{0, 0, 0} : get_specular_reflectance(material, shading_point);
    auto specular_reflectance = max(specular_reflectance_, Vector3{0, 0, 0});
    auto roughness = max(get_roughness(material, shading_point), min_roughness);
    roughness = max(roughness, Real(1e-6));
    if (material.compute_specular_lighting && !material.use_vertex_color) {
        // blinn-phong BRDF
        // half-vector
        auto m = normalize(wi + wo);
        auto m_local = to_local(shading_frame, m);
        auto flipped_m_local = false;
        if (material.two_sided) {
            if (m_local[2] < 0) {
                m_local = -m_local;
                flipped_m_local = true;
            }
        }
        if (m_local[2] > 0.f) {
            auto phong_exponent = roughness_to_phong(roughness);
            auto D = pow(m_local[2], phong_exponent) * (phong_exponent + 2.f) / Real(2 * M_PI);
            auto smithG1 = [&](const Vector3 &v) -> Real {
                auto cos_theta = dot(v, shading_frame.n);
                // tan^2 + 1 = 1/cos^2
                auto tan_theta =
                    sqrt(max(1.f / square(cos_theta) - 1.f, Real(0)));
                if (tan_theta == 0.0f) {
                    return 1;
                }
                auto alpha = sqrt(roughness);
                auto a = 1.f / (alpha * tan_theta);
                if (a >= 1.6f) {
                    return 1;
                }
                auto a_sqr = a * a;
                return (3.535f * a + 2.181f * a_sqr)
                     / (1.0f + 2.276f * a + 2.577f * a_sqr);
            };
            auto d_roughness = Real(0);
            auto d_smithG1 = [&](const Vector3 &v, Real d_G1) -> Vector3 {
                auto cos_theta = dot(v, shading_frame.n);
                if (dot(v, m) * cos_theta <= 0) {
                    return Vector3{0, 0, 0};
                }
                // tan^2 + 1 = 1/cos^2
                auto tan_theta = sqrt(max(1.f / square(cos_theta) - 1.f, Real(0)));
                if (tan_theta <= 1e-10f) {
                    return Vector3{0, 0, 0};
                }
                auto alpha = sqrt(roughness);
                auto a = 1.f / (alpha * tan_theta);
                if (a >= 1.6f) {
                    return Vector3{0, 0, 0};
                }
                auto numerator = 3.535f * a + 2.181f * square(a);
                auto denominator = 1.f + 2.276f * a + 2.557f * square(a);
                // G1 = numerator / denominator
                auto d_numerator = d_G1 / denominator;
                auto d_denominator = - d_G1 * numerator / square(denominator);
                auto d_a = d_numerator * (3.535f + 2.181f * 2 * a) +
                           d_denominator * (2.276f + 2.557f * 2 * a);
                // a = 1.f / (alpha * tan_theta)
                auto d_alpha = - d_a * a / alpha;
                auto d_tan_theta = - d_a * a / tan_theta;
                // alpha = sqrt(roughness)
                d_roughness += 0.5f * d_alpha / alpha;
                // tan_theta = sqrt(max(1.f / (cos_theta * cos_theta) - 1.f, Real(0)))
                auto d_tan_theta_sq = d_tan_theta * 0.5f / tan_theta;
                // tan_theta_sq = 1 / square(cos_theta) - 1
                auto d_cos_theta = -2.f * d_tan_theta_sq / cubic(cos_theta);
                // cos_theta = dot(v, shading_frame.n)
                auto d_v = d_cos_theta * shading_frame.n;
                d_n += d_cos_theta * v;
                return d_v;
            };
            auto Gwi = smithG1(wi);
            auto Gwo = smithG1(wo);
            auto G = Gwi * Gwo;
            auto cos_theta_d = dot(m, wo);
            // Schlick's approximation
            auto cos5 = pow(max(Real(1) - cos_theta_d, Real(0)), Real(5));
            auto F = specular_reflectance + (1.f - specular_reflectance) * cos5;
            auto specular_contrib = F * D * G / (4.f * shading_wi);

            // specular_contrib = F * D * G / (4.f * shading_wi)
            auto d_F = d_output * (D * G / (4.f * shading_wi));
            auto d_D = sum(d_output * F) * (G / (4.f * shading_wi));
            auto d_G = sum(d_output * F) * (D / (4.f * shading_wi));
            auto d_shading_wi = -sum(d_output * specular_contrib) / shading_wi;
            // shading_wi = fabs(dot(wi, shading_frame.n))
            if (dot(wi, shading_frame.n) < 0) {
                shading_wi = -shading_wi;
            }
            d_wi += d_shading_wi * shading_frame.n;
            d_n += d_shading_wi * wi;
            // F = specular_reflectance + (1.f - specular_reflectance) * cos5
            auto d_specular_reflectance = d_F * (1.f - cos5);
            auto d_cos_5 = sum(d_F * (1.f - specular_reflectance));
            // cos5 = pow(max(Real(1) - cos_theta_d, Real(0)), Real(5))
            auto d_cos_theta_d = -5.f * d_cos_5 * pow(max(Real(1) - cos_theta_d, Real(0)), Real(4));
            // cos_theta_d = dot(m, wo)
            auto d_m = d_cos_theta_d * wo;
            d_wo += d_cos_theta_d * m;
            // auto G = Gwi * Gwo;
            auto d_Gwi = d_G * Gwo;
            auto d_Gwo = d_G * Gwi;
            // Gwi = smithG1(wi)
            // Gwo = smithG1(wo)
            d_wi += d_smithG1(wi, d_Gwi);
            d_wo += d_smithG1(wo, d_Gwo);
            // D = pow(max(m_local[2], Real(0)), phong_exponent) *
            // (phong_exponent + 2.f) / Real(2 * M_PI)
            auto d_D_pow = d_D * (phong_exponent + 2.f) / Real(2 * M_PI);
            auto d_D_factor = d_D * pow(m_local[2], phong_exponent);
            auto d_m_local2 = d_D_pow * pow(max(m_local[2], Real(0)), phong_exponent - 1) *
                phong_exponent;
            // D_pow = pow(max(m_local[2], Real(0)), phong_exponent)
            auto d_phong_exponent =
                d_D_pow * pow(max(m_local[2], Real(0)), phong_exponent) * log(m_local[2]);
            // D_factor = (phong_exponent + 2.f) / Real(2 * M_PI)
            d_phong_exponent += d_D_factor / Real(2 * M_PI);
            // phong_exponent = roughness_to_phong(roughness)
            d_roughness += d_roughness_to_phong(roughness, d_phong_exponent);
            if (flipped_m_local) {
                d_m_local2 = -d_m_local2;
            }
            // m_local = to_local(shading_frame, m)
            // This is an isotropic BRDF so only normal is affected
            d_m += d_m_local2 * shading_frame.n;
            d_n += d_m_local2 * m;
            // m = normalize(wi + wo)
            auto d_wi_wo = d_normalize(wi + wo, d_m);
            d_wi += d_wi_wo;
            d_wo += d_wi_wo;
            // HACK: the "correct gradient" is the following, but it makes negative
            //       reflectance never going to come back.
            // auto d_specular_reflectance_ = Vector3{
            //     specular_reflectance_.x >= 0 ? d_specular_reflectance_.x : Real(0),
            //     specular_reflectance_.y >= 0 ? d_specular_reflectance_.y : Real(0),
            //     specular_reflectance_.z >= 0 ? d_specular_reflectance_.z : Real(0)
            // };
            // HACK (continued): instead we just use the gradients before the max.
            // specular_reflectance = get_specular_reflectance(material, shading_point)
            d_get_specular_reflectance(
                material, shading_point, d_specular_reflectance,
                d_material.specular_reflectance, d_shading_point);
            // roughness = get_roughness(material, shading_point.uv)
            if (roughness > min_roughness) {
                d_get_roughness(material,
                                shading_point,
                                d_roughness,
                                d_material.roughness,
                                d_shading_point);
            }
        }
    }

    if (has_normal_map(material)) {
        d_perturb_shading_frame(material,
                                shading_point,
                                d_n,
                                d_material,
                                d_shading_point);
    } else {
        d_shading_point.shading_frame.n += d_n;
    }
}

DEVICE
inline
Vector3 cos_hemisphere(const Vector2 &sample) {
    auto phi = 2.f * float(M_PI) * sample[0];
    auto tmp = sqrt(max(1.f - sample[1], Real(0)));
    return Vector3{cos(phi) * tmp, sin(phi) * tmp, sqrt(sample[1])};
}

DEVICE
inline
Vector3 bsdf_sample(const Material &material,
                    const SurfacePoint &shading_point,
                    const Vector3 &wi,
                    const BSDFSample &bsdf_sample,
                    const Real min_roughness,
                    const RayDifferential &wi_differential,
                    RayDifferential &wo_differential,
                    Real *next_min_roughness = nullptr) {
    if (next_min_roughness != nullptr) {
        *next_min_roughness = min_roughness;
    }
    auto shading_frame = shading_point.shading_frame;
    if (has_normal_map(material)) {
        // Perturb shading frame
        shading_frame = perturb_shading_frame(material, shading_point);
    }
    auto geom_normal = shading_point.geom_normal;
    // Flip geometry normal to the same side of shading normal
    if (dot(geom_normal, shading_frame.n) < 0) {
        geom_normal = -geom_normal;
    }
    auto geom_wi = dot(geom_normal, wi);
    if (!material.two_sided) {
        // The surface doesn't reflect light on the other side of
        // the geometry normal.
        if (geom_wi < 0) {
            return Vector3{0, 0, 0};
        }
    }

    auto diffuse_reflectance_ = material.use_vertex_color ?
        shading_point.color : get_diffuse_reflectance(material, shading_point);
    auto specular_reflectance_ = material.use_vertex_color ?
        Vector3{0, 0, 0} : get_specular_reflectance(material, shading_point);
    auto diffuse_reflectance = max(diffuse_reflectance_, Vector3{0, 0, 0});
    auto specular_reflectance = max(specular_reflectance_, Vector3{0, 0, 0});
    auto diffuse_weight = luminance(diffuse_reflectance);
    auto specular_weight = luminance(specular_reflectance);
    auto weight_sum = diffuse_weight + specular_weight;
    auto diffuse_pmf = Real(0.5);
    if (weight_sum > 0.f) {
        diffuse_pmf = diffuse_weight / weight_sum;
    }
    // auto specular_pmf = specular_weight / weight_sum;
    if (bsdf_sample.w <= diffuse_pmf) {
        // Lambertian
        if (next_min_roughness != nullptr) {
            *next_min_roughness = Real(1);
        }
        auto local_dir = cos_hemisphere(bsdf_sample.uv);
        // Propagate ray differentials
        wo_differential.org_dx = wi_differential.org_dx;
        wo_differential.org_dy = wi_differential.org_dy;
        // HACK: Output direction has no dependencies w.r.t. input
        // However, since the diffuse BRDF serves as a low pass filter,
        // we want to assign a larger prefilter.
        wo_differential.dir_dx = Vector3{0.03f, 0.03f, 0.03f};
        wo_differential.dir_dy = Vector3{0.03f, 0.03f, 0.03f};
        auto dir = to_world(shading_frame, local_dir);
        if (dot(geom_normal, dir) * geom_wi < 0) {
            // Flip the outgoing direction back to the same side of surface
            dir = to_world(shading_frame, -local_dir);
        }
        return dir;
    } else {
        // Blinn-phong
        auto roughness = max(get_roughness(material, shading_point), min_roughness);
        roughness = max(roughness, Real(1e-6));
        if (next_min_roughness != nullptr) {
            *next_min_roughness = max(roughness, min_roughness);
        }
        auto phong_exponent = roughness_to_phong(roughness);
        // Sample phi
        auto phi = 2.f * Real(M_PI) * bsdf_sample.uv[1];
        auto sin_phi = sin(phi);
        auto cos_phi = cos(phi);
        
        // Sample theta
        auto cos_theta = pow(bsdf_sample.uv[0], Real(1) / (phong_exponent + Real(2)));
        auto sin_theta = sqrt(max(1.f - cos_theta * cos_theta, Real(0)));
        // local microfacet normal
        auto m_local = Vector3{sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
        auto m = to_world(shading_frame, m_local);
        auto dir = 2.f * dot(wi, m) * m - wi;
        if (dot(geom_normal, dir) * geom_wi < 0) {
            // Flip m_local
            m_local = -m_local;
            m = to_world(shading_frame, m_local);
            dir = 2.f * dot(wi, m) * m - wi;
        }
        // Propagate ray differentials
        // HACK: we approximate the directional derivative dmdx using dndx * m_local[2]
        // i.e. we ignore the derivatives on the tangent plane
        auto dmdx = shading_point.dn_dx * m_local[2];
        auto dmdy = shading_point.dn_dy * m_local[2];
        auto wi_dx = -wi_differential.dir_dx;
        auto wi_dy = -wi_differential.dir_dy;
        // Igehy 1999, Equation 15
        auto widotm_dx = sum(wi_dx * m) + sum(wi * dmdx);
        auto widotm_dy = sum(wi_dy * m) + sum(wi * dmdy);
        // Igehy 1999, Equation 14
        wo_differential.org_dx = wi_differential.org_dx;
        wo_differential.org_dy = wi_differential.org_dy;
        wo_differential.dir_dx = 2 * (dot(wi, m) * dmdx + widotm_dx * m) - wi_dx;
        wo_differential.dir_dy = 2 * (dot(wi, m) * dmdy + widotm_dy * m) - wi_dy;
        return dir;
    }
}

DEVICE
inline
void d_bsdf_sample(const Material &material,
                   const SurfacePoint &shading_point,
                   const Vector3 &wi,
                   const BSDFSample &bsdf_sample,
                   const Real min_roughness,
                   const RayDifferential &wi_differential,
                   const Vector3 &d_wo,
                   const RayDifferential &d_wo_differential,
                   DMaterial &d_material,
                   SurfacePoint &d_shading_point,
                   Vector3 &d_wi,
                   RayDifferential &d_wi_differential) {
    auto shading_frame = shading_point.shading_frame;
    if (has_normal_map(material)) {
        // Perturb shading frame
        shading_frame = perturb_shading_frame(material, shading_point);
    }
    auto geom_normal = shading_point.geom_normal;
    // Flip geometry normal to the same side of shading normal
    if (dot(geom_normal, shading_frame.n) < 0) {
        geom_normal = -geom_normal;
    }
    auto geom_wi = dot(geom_normal, wi);
    if (!material.two_sided) {
        // The surface doesn't reflect light on the other side of
        // the geometry normal.
        if (geom_wi < 0) {
            return;
        }
    }

    auto diffuse_reflectance_ = material.use_vertex_color ?
        shading_point.color : get_diffuse_reflectance(material, shading_point);
    auto specular_reflectance_ = material.use_vertex_color ?
        Vector3{0, 0, 0} : get_specular_reflectance(material, shading_point);
    auto diffuse_reflectance = max(diffuse_reflectance_, Vector3{0, 0, 0});
    auto specular_reflectance = max(specular_reflectance_, Vector3{0, 0, 0});
    auto diffuse_weight = luminance(diffuse_reflectance);
    auto specular_weight = luminance(specular_reflectance);
    auto weight_sum = diffuse_weight + specular_weight;
    auto diffuse_pmf = Real(0.5);
    if (weight_sum > 0.f) {
        diffuse_pmf = diffuse_weight / weight_sum;
    }
    auto d_shading_frame = Frame{Vector3{0, 0, 0},
                                 Vector3{0, 0, 0},
                                 Vector3{0, 0, 0}};
    // auto specular_pmf = specular_weight / weight_sum;
    if (bsdf_sample.w <= diffuse_pmf) {
        // Lambertian
        if (diffuse_pmf <= 0.f) {
            return;
        }
        auto local_dir = cos_hemisphere(bsdf_sample.uv);
        auto wo = to_world(shading_frame, local_dir);
        // if (dot(geom_normal, wo) * geom_wi < 0) {
        //     // Flip the outgoing direction back to the same side of surface
        //     wo = to_world(shading_frame, -local_dir);
        // }
        auto d_local_dir = Vector3{0, 0, 0};
        if (dot(geom_normal, wo) * geom_wi < 0) {
            d_to_world(shading_frame, local_dir, d_wo, d_shading_frame, d_local_dir);
        } else {
            d_to_world(shading_frame, -local_dir, d_wo, d_shading_frame, d_local_dir);
            d_local_dir = -d_local_dir;
        }
        // No need to propagate to bsdf_sample

        // Propagate ray differentials
        // wo_differential.org_dx = wi_differential.org_dx;
        // wo_differential.org_dy = wi_differential.org_dy;
        // // Output direction has no dependencies w.r.t. input
        // wo_differential.dir_dx = Vector3{0, 0, 0};
        // wo_differential.dir_dy = Vector3{0, 0, 0};
        d_wi_differential.org_dx += d_wo_differential.org_dx;
        d_wi_differential.org_dy += d_wo_differential.org_dy;
    } else {
        if (specular_weight <= 0.f) {
            return;
        }
        // Blinn-phong
        auto roughness = max(get_roughness(material, shading_point), min_roughness);
        roughness = max(roughness, Real(1e-6));
        auto phong_exponent = roughness_to_phong(roughness);
        // Sample phi
        auto phi = 2.f * Real(M_PI) * bsdf_sample.uv[1];
        auto sin_phi = sin(phi);
        auto cos_phi = cos(phi);
        
        // Sample theta
        auto cos_theta = pow(bsdf_sample.uv[0], Real(1) / (phong_exponent + Real(2)));
        auto sin_theta = sqrt(max(1.f - cos_theta*cos_theta, Real(0)));
        // local microfacet normal
        auto m_local = Vector3{sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
        auto m = to_world(shading_frame, m_local);
        auto dir = 2.f * dot(wi, m) * m - wi;
        auto m_flipped = false;
        if (dot(geom_normal, dir) * geom_wi < 0) {
            // Flip m_local
            m_local = -m_local;
            m = to_world(shading_frame, m_local);
            m_flipped = true;
            // dir = 2.f * dot(wi, m) * m - wi;
        }

        // Propagate ray differentials
        // HACK: we approximate the directional derivative dmdx using dndx * m_local[2]
        // i.e. we ignore the derivatives on the tangent plane
        auto dmdx = shading_point.dn_dx * m_local[2];
        auto dmdy = shading_point.dn_dy * m_local[2];
        auto wi_dx = -wi_differential.dir_dx;
        auto wi_dy = -wi_differential.dir_dy;
        // Igehy 1999, Equation 15
        auto widotm_dx = sum(wi_dx * m) + sum(wi * dmdx);
        auto widotm_dy = sum(wi_dy * m) + sum(wi * dmdy);
        // Igehy 1999, Equation 14
        // wo_differential.org_dx = wi_differential.org_dx
        // wo_differential.org_dy = wi_differential.org_dy
        // wo_differential.dir_dx = 2 * (dot(wi, m) * dmdx + widotm_dx * m) - wi_dx
        // wo_differential.dir_dy = 2 * (dot(wi, m) * dmdy + widotm_dy * m) - wi_dy

        d_wi_differential.org_dx += d_wo_differential.org_dx;
        d_wi_differential.org_dy += d_wo_differential.org_dy;
        d_wi_differential.dir_dx += d_wo_differential.dir_dx;
        d_wi_differential.dir_dy += d_wo_differential.dir_dy;
        auto d_dot_wi_m = 2 * sum(d_wo_differential.dir_dx * dmdx) +
                          2 * sum(d_wo_differential.dir_dy * dmdy);
        auto d_dmdx = d_wo_differential.dir_dx * 2 * dot(wi, m);
        auto d_dmdy = d_wo_differential.dir_dy * 2 * dot(wi, m);
        auto d_widotm_dx = 2 * sum(d_wo_differential.dir_dx * m);
        auto d_widotm_dy = 2 * sum(d_wo_differential.dir_dy * m);
        auto d_m = d_wo_differential.dir_dx * 2 * widotm_dx +
                   d_wo_differential.dir_dy * 2 * widotm_dy;
        // widotm_dx = sum(wi_dx * m) + sum(wi * dmdx)
        auto d_wi_dx = d_widotm_dx * m;
        d_m += d_widotm_dx * wi_dx;
        d_wi += d_widotm_dx * dmdx;
        d_dmdx += d_widotm_dx * wi;
        // widotm_dy = sum(wi_dy * m) + sum(wi * dmdy)
        auto d_wi_dy = d_widotm_dy * m;
        d_m += d_widotm_dy * wi_dy;
        d_wi += d_widotm_dy * dmdy;
        d_dmdy += d_widotm_dy * wi;
        // wi_dx = -wi_differential.dir_dx
        // wi_dy = -wi_differential.dir_dy
        d_wi_differential.dir_dx -= d_wi_dx;
        d_wi_differential.dir_dy -= d_wi_dy;
        // dmdx = shading_point.dn_dx * m_local[2]
        // dmdy = shading_point.dn_dy * m_local[2]
        d_shading_point.dn_dx += d_dmdx * m_local[2];
        d_shading_point.dn_dy += d_dmdy * m_local[2];
        auto d_m_local = Vector3{0, 0, 0};
        d_m_local[2] += sum(d_dmdx * shading_point.dn_dx) +
                        sum(d_dmdy * shading_point.dn_dy);

        // wo = 2.f * dot(wi, m) * m - wi
        d_dot_wi_m += 2.f * sum(d_wo * m);
        d_m += d_wo * (2.f * dot(wi, m));
        d_wi += -d_wo;
        // dot_wi_m = dot(wi, m)
        d_wi += d_dot_wi_m * m;
        d_m += d_dot_wi_m * wi;

        // m = to_world(shading_frame, m_local)
        d_to_world(shading_frame, m_local, d_m, d_shading_frame, d_m_local);
        if (m_flipped) {
            d_m_local = -d_m_local;
        }

        // No need to propagate to phi
        // m_local[0] = sin_theta * cos_phi
        auto d_sin_theta = d_m_local[0] * cos_phi;
        // m_local[1] = sin_theta * sin_phi
        d_sin_theta += d_m_local[1] * sin_phi;
        // m_local[2] = cos_theta
        auto d_cos_theta = d_m_local[2];
        // sin_theta = sqrt(max(1.f - cos_theta*cos_theta, Real(0)))
        auto d_one_minus_cos_theta_2 = sin_theta > 0 ? d_sin_theta * 0.5f / sin_theta : Real(0);
        // 1 - cos_theta * cos_theta
        d_cos_theta -= d_one_minus_cos_theta_2 * 2 * cos_theta;
        // cos_theta = pow(bsdf_sample.uv[0], Real(1) / (phong_exponent + Real(2)))
        auto d_one_over_phong_exponent_plus_2 =
            bsdf_sample.uv[0] > 0 ? d_cos_theta * cos_theta * log(bsdf_sample.uv[0]) : Real(0);
        // 1 / (phong_exponent + 2)
        auto d_phong_exponent = -d_one_over_phong_exponent_plus_2 / square(phong_exponent + 2.0f);
        // phong_exponent = roughness_to_phong(roughness)
        auto d_roughness = d_roughness_to_phong(roughness, d_phong_exponent);
        // roughness = get_roughness(material, shading_point)
        if (roughness > min_roughness) {
            d_get_roughness(material,
                            shading_point,
                            d_roughness,
                            d_material.roughness,
                            d_shading_point);
        }
    }

    if (has_normal_map(material)) {
        d_perturb_shading_frame(material,
                                shading_point,
                                d_shading_frame,
                                d_material,
                                d_shading_point);
    } else {
        d_shading_point.shading_frame += d_shading_frame;
    }
}

DEVICE
inline Real bsdf_pdf(const Material &material,
                     const SurfacePoint &shading_point,
                     const Vector3 &wi,
                     const Vector3 &wo,
                     const Real min_roughness) {
    auto shading_frame = shading_point.shading_frame;
    auto geom_n = shading_point.geom_normal;
    if (has_normal_map(material)) {
        // Perturb shading frame
        shading_frame = perturb_shading_frame(material, shading_point);
    }
    // Flip geometry normal to the same side of the shading frame
    if (dot(geom_n, shading_frame.n) < 0) {
        geom_n = -geom_n;
    }
    auto geom_wi = dot(geom_n, wi);
    auto geom_wo = dot(geom_n, wo);
    auto shading_wo = fabs(dot(shading_frame.n, wo));
    if (geom_wi * geom_wo < 0) {
        // wi & wo are at different sides of the geometry
        // TODO: implement BTDF
        return 0;
    }
    if (!material.two_sided) {
        // The surface doesn't reflect light on the other side of
        // the geometry normal.
        // Otherwise two sided means both sides are the same BRDF.
        if (geom_wi < 0 && geom_wo < 0) {
            return 0;
        }
    }

    auto diffuse_reflectance_ = material.use_vertex_color ?
        shading_point.color : get_diffuse_reflectance(material, shading_point);
    auto specular_reflectance_ = material.use_vertex_color ?
        Vector3{0, 0, 0} : get_specular_reflectance(material, shading_point);
    auto diffuse_reflectance = max(diffuse_reflectance_, Vector3{0, 0, 0});
    auto specular_reflectance = max(specular_reflectance_, Vector3{0, 0, 0});
    auto diffuse_weight = luminance(diffuse_reflectance);
    auto specular_weight = luminance(specular_reflectance);
    auto weight_sum = diffuse_weight + specular_weight;
    auto diffuse_pmf = Real(0.5f);
    auto specular_pmf = Real(0.5f);
    if (weight_sum > 0.f) {
        diffuse_pmf = diffuse_weight / weight_sum;
        specular_pmf = specular_weight / weight_sum;
    }
    auto diffuse_pdf = Real(0);
    if (diffuse_pmf > 0.f) {
        diffuse_pdf = diffuse_pmf * shading_wo / Real(M_PI);
    }
    auto specular_pdf = Real(0);
    if (specular_pmf > 0.f) {
        auto m = normalize(wi + wo);
        auto m_local = to_local(shading_point.shading_frame, m);
        if (material.two_sided) {
            if (m_local[2] < 0) {
                m_local = -m_local;
            }
        }
        if (m_local[2] > 0.f && fabs(dot(m, wo)) > 0) {
            auto roughness = max(get_roughness(material, shading_point), min_roughness);
            roughness = max(roughness, Real(1e-6));
            auto phong_exponent = roughness_to_phong(roughness);
            auto D = pow(m_local[2], phong_exponent) * (phong_exponent + 2.f) / Real(2 * M_PI);
            specular_pdf = specular_pmf * D * m_local[2] / (4.f * fabs(dot(m, wo)));
        }
    }
    return diffuse_pdf + specular_pdf;
}

DEVICE
inline void d_bsdf_pdf(const Material &material,
                       const SurfacePoint &shading_point,
                       const Vector3 &wi,
                       const Vector3 &wo,
                       const Real min_roughness,
                       const Real d_pdf,
                       DMaterial &d_material,
                       SurfacePoint &d_shading_point,
                       Vector3 &d_wi,
                       Vector3 &d_wo) {
    auto shading_frame = shading_point.shading_frame;
    auto geom_n = shading_point.geom_normal;
    if (has_normal_map(material)) {
        // Perturb shading frame
        shading_frame = perturb_shading_frame(material, shading_point);
    }
    // Flip geometry normal to the same side of the shading frame
    if (dot(geom_n, shading_frame.n) < 0) {
        geom_n = -geom_n;
    }
    auto geom_wi = dot(geom_n, wi);
    auto geom_wo = dot(geom_n, wo);
    // auto shading_wo = fabs(dot(shading_frame.n, wo));
    if (geom_wi * geom_wo < 0) {
        // wi & wo are at different sides of the geometry
        // TODO: implement BTDF
        return;
    }
    if (!material.two_sided) {
        // The surface doesn't reflect light on the other side of
        // the geometry normal.
        // Otherwise two sided means both sides are the same BRDF.
        if (geom_wi < 0 && geom_wo < 0) {
            return;
        }
    }

    auto diffuse_reflectance_ = material.use_vertex_color ?
        shading_point.color : get_diffuse_reflectance(material, shading_point);
    auto specular_reflectance_ = material.use_vertex_color ?
        Vector3{0, 0, 0} : get_specular_reflectance(material, shading_point);
    auto diffuse_reflectance = max(diffuse_reflectance_, Vector3{0, 0, 0});
    auto specular_reflectance = max(specular_reflectance_, Vector3{0, 0, 0});
    auto diffuse_weight = luminance(diffuse_reflectance);
    auto specular_weight = luminance(specular_reflectance);
    auto weight_sum = diffuse_weight + specular_weight;
    auto diffuse_pmf = Real(0.5f);
    auto specular_pmf = Real(0.5f);
    if (weight_sum > 0.f) {
        diffuse_pmf = diffuse_weight / weight_sum;
        specular_pmf = specular_weight / weight_sum;
    }
    // Diffuse PDF
    // auto diffuse_pdf = diffuse_pmf * shading_wo / Real(M_PI);
    auto d_n = Vector3{0, 0, 0};
    if (diffuse_pmf > 0) {
        // Ignore derivatives to discrete probability
        auto d_shading_wo = d_pdf * diffuse_pmf / Real(M_PI);
        // shading_wo = fabs(dot(shading_frame.n, wo))
        if (dot(shading_frame.n, wo) < 0) {
            d_shading_wo = -d_shading_wo;
        }
        d_wo += shading_frame.n * d_shading_wo;
        d_n += wo * d_shading_wo;
    }

    if (specular_pmf > 0) {
        auto m = normalize(wi + wo);
        auto m_local = to_local(shading_point.shading_frame, m);
        auto m_flipped = false;
        if (material.two_sided) {
            if (m_local[2] < 0) {
                m_local = -m_local;
                m_flipped = true;
            }
        }
        if (m_local[2] > 0.f && fabs(dot(wo, m)) > 0) {
            auto roughness = max(get_roughness(material, shading_point), min_roughness);
            roughness = max(roughness, Real(1e-6));
            auto phong_exponent = roughness_to_phong(roughness);
            auto D = pow(m_local[2], phong_exponent) * (phong_exponent + 2.f) / Real(2 * M_PI);
            // specular_pdf = specular_pmf * D * m_local[2] / (4.f * fabs(dot(m, wo)));
            auto d_D = d_pdf * specular_pmf * m_local[2] / (4.f * fabs(dot(wo, m)));
            auto d_m_local2 = d_pdf * specular_pmf * D / (4.f * fabs(dot(wo, m)));
            auto d_abs_dot_wo_m =
                -d_pdf * specular_pmf * D * m_local[2] / (4.f * square(dot(wo, m)));
            // abs(dot(m, wo))
            auto d_m = Vector3{0, 0, 0};
            if (dot(wo, m) > 0) {
                d_wo += d_abs_dot_wo_m * m;
                d_m += d_abs_dot_wo_m * wo;
            } else {
                d_wo -= d_abs_dot_wo_m * m;
                d_m -= d_abs_dot_wo_m * wo;
            }
            // D = pow(m_local[2], phong_exponent) * (phong_exponent + 2.f) / Real(2 * M_PI)
            auto d_D_pow = d_D * (phong_exponent + 2.f) / Real(2 * M_PI);
            auto d_D_factor = d_D * pow(m_local[2], phong_exponent);
            // pow(m_local[2], phong_exponent)
            d_m_local2 += d_D_pow * pow(m_local[2], phong_exponent - 1) * phong_exponent;
            // pow(m_local[2], phong_exponent)
            auto d_phong_exponent = d_D_pow * pow(max(m_local[2], Real(0)), phong_exponent) *
                                    log(m_local[2]);
            // (phong_exponent + 2.f) / Real(2 * M_PI)
            d_phong_exponent += d_D_factor / Real(2 * M_PI);
            // phong_exponent = roughness_to_phong(roughness)
            auto d_roughness = d_roughness_to_phong(roughness, d_phong_exponent);
            if (m_flipped) {
                d_m_local2 = -d_m_local2;
            }

            // m_local = to_local(shading_point.shading_frame, m)
            // This is an isotropic BRDF so only normal is affected
            d_m += d_m_local2 * shading_frame.n;
            d_n += d_m_local2 * m;
            // m = normalize(wi + wo)
            auto d_wi_wo = d_normalize(wi + wo, d_m);
            d_wi += d_wi_wo;
            d_wo += d_wi_wo;
            // roughness = get_roughness(material, shading_point)
            if (roughness > min_roughness) {
                d_get_roughness(material,
                                shading_point,
                                d_roughness,
                                d_material.roughness,
                                d_shading_point);
            }
        }
    }

    if (has_normal_map(material)) {
        d_perturb_shading_frame(material,
                                shading_point,
                                d_n,
                                d_material,
                                d_shading_point);
    } else {
        d_shading_point.shading_frame.n += d_n;
    }
}

void test_d_bsdf();
void test_d_bsdf_sample();
void test_d_bsdf_pdf();
