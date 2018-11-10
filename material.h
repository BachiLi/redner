#pragma once

#include "redner.h"
#include "vector.h"
#include "intersection.h"
#include "buffer.h"
#include "ptr.h"

struct Texture1 {
    Texture1() {}
    Texture1(ptr<float> texels,
             int width,
             int height)
        : texels(texels.get()),
          width(width), height(height) {}

    float *texels;
    int width;
    int height;
};

struct DTexture1 {
    int material_id = -1, xi = -1, yi = -1;
    Real t00 = 0, t01 = 0, t10 = 0, t11 = 0;

    DEVICE inline bool operator<(const DTexture1 &other) const {
        if (material_id != other.material_id) {
            return material_id < other.material_id;
        }
        if (yi != other.yi) {
            return yi < other.yi;
        }
        return xi < other.xi;
    }

    DEVICE inline bool operator==(const DTexture1 &other) const {
        return material_id == other.material_id && xi == other.xi && yi == other.yi;
    }

    DEVICE inline DTexture1 operator+(const DTexture1 &other) const {
        return DTexture1{material_id, xi, yi,
                         t00 + other.t00,
                         t01 + other.t01,
                         t10 + other.t10,
                         t11 + other.t11};
    }
};

struct Texture3 {
    Texture3() {}
    Texture3(ptr<float> texels,
             int width,
             int height)
        : texels(texels.get()),
          width(width), height(height) {}

    float *texels;
    int width;
    int height;
};

struct DTexture3 {
    int material_id = -1, xi = -1, yi = -1;
    Vector3 t00 = Vector3{0, 0, 0};
    Vector3 t01 = Vector3{0, 0, 0};
    Vector3 t10 = Vector3{0, 0, 0};
    Vector3 t11 = Vector3{0, 0, 0};

    DEVICE inline bool operator<(const DTexture3 &other) const {
        if (material_id != other.material_id) {
            return material_id < other.material_id;
        }
        if (yi != other.yi) {
            return yi < other.yi;
        }
        return xi < other.xi;
    }

    DEVICE inline bool operator==(const DTexture3 &other) const {
        return material_id == other.material_id && xi == other.xi && yi == other.yi;
    }

    DEVICE inline DTexture3 operator+(const DTexture3 &other) const {
        return DTexture3{material_id, xi, yi,
                         t00 + other.t00,
                         t01 + other.t01,
                         t10 + other.t10,
                         t11 + other.t11};
    }
};

struct Material {
    Material() {}

    Material(Texture3 diffuse_reflectance,
             Texture3 specular_reflectance,
             Texture1 roughness,
             ptr<float> diffuse_uv_scale,
             ptr<float> specular_uv_scale,
             ptr<float> roughness_uv_scale,
             bool two_sided)
        : diffuse_reflectance(diffuse_reflectance),
          specular_reflectance(specular_reflectance),
          roughness(roughness),
          diffuse_uv_scale(Vector2f{diffuse_uv_scale[0], diffuse_uv_scale[1]}),
          specular_uv_scale(Vector2f{specular_uv_scale[0], specular_uv_scale[1]}),
          roughness_uv_scale(Vector2f{roughness_uv_scale[0], roughness_uv_scale[1]}),
          two_sided(two_sided) {}

    inline std::pair<int, int> get_diffuse_size() const {
        return {diffuse_reflectance.width, diffuse_reflectance.height};
    }

    inline std::pair<int, int> get_specular_size() const {
        return {specular_reflectance.width, specular_reflectance.height};
    }

    inline std::pair<int, int> get_roughness_size() const {
        return {roughness.width, roughness.height};
    }

    Texture3 diffuse_reflectance;
    Texture3 specular_reflectance;
    Texture1 roughness;
    Vector2f diffuse_uv_scale;
    Vector2f specular_uv_scale;
    Vector2f roughness_uv_scale;
    bool two_sided;
};

struct DMaterial {
    Texture3 diffuse_reflectance;
    Texture3 specular_reflectance;
    Texture1 roughness;
};

template <typename T>
struct TBSDFSample {
    TVector2<T> uv;
    T w;
};

using BSDFSample = TBSDFSample<Real>;

DEVICE
inline Vector3 get_texture_value(const Texture3 &tex, const Vector2 &uv) {
    if (tex.width <= 0 && tex.height <= 0) {
        // Constant texture
        return Vector3{tex.texels[0], tex.texels[1], tex.texels[2]};
    } else {
        // Bilinear interpolation
        // TODO: mipmap
        auto x = uv[0] * tex.width - 0.5f;
        auto y = uv[1] * tex.height - 0.5f;
        auto xf = (int)floor(x);
        auto yf = (int)floor(y);
        auto xc = xf + 1;
        auto yc = yf + 1;
        auto u = x - xf;
        auto v = y - yf;
        auto xfi = modulo(xf, tex.width);
        auto yfi = modulo(yf, tex.height);
        auto xci = modulo(xc, tex.width);
        auto yci = modulo(yc, tex.height);
        auto color_ff = Vector3f{tex.texels[3 * (yfi * tex.width + xfi) + 0],
                                 tex.texels[3 * (yfi * tex.width + xfi) + 1],
                                 tex.texels[3 * (yfi * tex.width + xfi) + 2]};
        auto color_cf = Vector3f{tex.texels[3 * (yfi * tex.width + xci) + 0],
                                 tex.texels[3 * (yfi * tex.width + xci) + 1],
                                 tex.texels[3 * (yfi * tex.width + xci) + 2]};
        auto color_fc = Vector3f{tex.texels[3 * (yci * tex.width + xfi) + 0],
                                 tex.texels[3 * (yci * tex.width + xfi) + 1],
                                 tex.texels[3 * (yci * tex.width + xfi) + 2]};
        auto color_cc = Vector3f{tex.texels[3 * (yci * tex.width + xci) + 0],
                                 tex.texels[3 * (yci * tex.width + xci) + 1],
                                 tex.texels[3 * (yci * tex.width + xci) + 2]};
        auto color = color_ff * (1.f - u) * (1.f - v) +
                     color_fc * (1.f - u) *        v  +
                     color_cf *        u  * (1.f - v) +
                     color_cc *        u  *        v;
        return color;
    }
}

DEVICE
inline void d_get_texture_value(const Texture3 &tex, const Vector2 &uv, const Vector3 &d_output,
                                DTexture3 &d_tex, Vector2 &d_uv) {
    if (tex.width <= 0 && tex.height <= 0) {
        // Constant texture
        // output =  Vector3{tex.texels[0], tex.texels[1], tex.texels[2]};
        d_tex.t00 = d_output;
    } else {
        // Bilinear interpolation
        // TODO: mipmap
        auto x = uv[0] * tex.width - 0.5f;
        auto y = uv[1] * tex.height - 0.5f;
        auto xf = (int)floor(x);
        auto yf = (int)floor(y);
        auto xc = xf + 1;
        auto yc = yf + 1;
        auto u = x - xf;
        auto v = y - yf;
        auto xfi = modulo(xf, tex.width);
        auto yfi = modulo(yf, tex.height);
        auto xci = modulo(xc, tex.width);
        auto yci = modulo(yc, tex.height);
        auto color_ff = Vector3f{tex.texels[3 * (yfi * tex.width + xfi) + 0],
                                 tex.texels[3 * (yfi * tex.width + xfi) + 1],
                                 tex.texels[3 * (yfi * tex.width + xfi) + 2]};
        auto color_cf = Vector3f{tex.texels[3 * (yfi * tex.width + xci) + 0],
                                 tex.texels[3 * (yfi * tex.width + xci) + 1],
                                 tex.texels[3 * (yfi * tex.width + xci) + 2]};
        auto color_fc = Vector3f{tex.texels[3 * (yci * tex.width + xfi) + 0],
                                 tex.texels[3 * (yci * tex.width + xfi) + 1],
                                 tex.texels[3 * (yci * tex.width + xfi) + 2]};
        auto color_cc = Vector3f{tex.texels[3 * (yci * tex.width + xci) + 0],
                                 tex.texels[3 * (yci * tex.width + xci) + 1],
                                 tex.texels[3 * (yci * tex.width + xci) + 2]};
        // color = color_ff * (1.f - u) * (1.f - v) +
        //         color_fc * (1.f - u) *        v  +
        //         color_cf *        u  * (1.f - v) +
        //         color_cc *        u  *        v;
        auto d_color_ff = d_output * (1.f - u) * (1.f - v);
        auto d_color_cf = d_output *        u  * (1.f - v);
        auto d_color_fc = d_output * (1.f - u) *        v ;
        auto d_color_cc = d_output *        u  *        v ;
        auto d_u = sum(d_output * (-color_ff * (1.f - v) +
                                    color_cf * (1.f - v) +
                                   -color_fc *        v  +
                                    color_cc *        v));
        auto d_v = sum(d_output * (-color_ff * (1.f - u) +
                                   -color_cf *        u  +
                                    color_fc * (1.f - u) +
                                    color_cc *        u));
        d_tex.xi = xfi;
        d_tex.yi = yfi;
        d_tex.t00 = d_color_ff;
        d_tex.t01 = d_color_fc;
        d_tex.t10 = d_color_cf;
        d_tex.t11 = d_color_cc;
        // du = dx, dv = dy
        // x = uv[0] * tex.width - 0.5f
        // y = uv[1] * tex.height - 0.5f
        d_uv[0] += d_u * tex.width;
        d_uv[1] += d_v * tex.height;
    }
}

DEVICE
inline Real get_texture_value(const Texture1 &tex, const Vector2 &uv) {
    if (tex.width <= 0 && tex.height <= 0) {
        // Constant texture
        return Real(tex.texels[0]);
    } else {
        // Bilinear interpolation
        // TODO: mipmap
        auto x = uv[0] * tex.width - 0.5f;
        auto y = uv[1] * tex.height - 0.5f;
        auto xf = (int)floor(x);
        auto yf = (int)floor(y);
        auto xc = xf + 1;
        auto yc = yf + 1;
        auto u = x - xf;
        auto v = y - yf;
        auto xfi = modulo(xf, tex.width);
        auto yfi = modulo(yf, tex.height);
        auto xci = modulo(xc, tex.width);
        auto yci = modulo(yc, tex.height);
        auto value_ff = tex.texels[yfi * tex.width + xfi];
        auto value_cf = tex.texels[yfi * tex.width + xci];
        auto value_fc = tex.texels[yci * tex.width + xfi];
        auto value_cc = tex.texels[yci * tex.width + xci];
        auto value = value_ff * (1.f - u) * (1.f - v) +
                     value_fc * (1.f - u) *        v  +
                     value_cf *        u  * (1.f - v) +
                     value_cc *        u  *        v;
        return value;
    }
}

DEVICE
inline void d_get_texture_value(const Texture1 &tex, const Vector2 &uv, Real d_output,
                                DTexture1 &d_tex, Vector2 &d_uv) {
    if (tex.width <= 0 && tex.height <= 0) {
        // Constant texture
        d_tex.t00 += d_output;
    } else {
        // Bilinear interpolation
        // TODO: mipmap
        auto x = uv[0] * tex.width - 0.5f;
        auto y = uv[1] * tex.height - 0.5f;
        auto xf = (int)floor(x);
        auto yf = (int)floor(y);
        auto xc = xf + 1;
        auto yc = yf + 1;
        auto u = x - xf;
        auto v = y - yf;
        auto xfi = modulo(xf, tex.width);
        auto yfi = modulo(yf, tex.height);
        auto xci = modulo(xc, tex.width);
        auto yci = modulo(yc, tex.height);
        auto value_ff = tex.texels[yfi * tex.width + xfi];
        auto value_cf = tex.texels[yfi * tex.width + xci];
        auto value_fc = tex.texels[yci * tex.width + xfi];
        auto value_cc = tex.texels[yci * tex.width + xci];
        // value = value_ff * (1.f - u) * (1.f - v) +
        //         value_fc * (1.f - u) *        v  +
        //         value_cf *        u  * (1.f - v) +
        //         value_cc *        u  *        v;
        auto d_value_ff = d_output * (1.f - u) * (1.f - v);
        auto d_value_cf = d_output *        u  * (1.f - v);
        auto d_value_fc = d_output * (1.f - u) *        v ;
        auto d_value_cc = d_output *        u  *        v ;
        auto d_u = d_output * (-value_ff * (1.f - v) +
                                value_cf * (1.f - v) +
                               -value_fc *        v  +
                                value_cc *        v);
        auto d_v = d_output * (-value_ff * (1.f - u) +
                               -value_cf *        u  +
                                value_fc * (1.f - u) +
                                value_cc *        u);
        d_tex.xi = xfi;
        d_tex.yi = yfi;
        d_tex.t00 = d_value_ff;
        d_tex.t01 = d_value_cf;
        d_tex.t10 = d_value_fc;
        d_tex.t11 = d_value_cc;
        // du = dx, dv = dy
        // x = uv[0] * tex.width - 0.5f
        // y = uv[1] * tex.height - 0.5f
        d_uv[0] += d_u * tex.width;
        d_uv[1] += d_v * tex.height;
    }
}


DEVICE
inline Vector3 get_diffuse_reflectance(const Material &material,
                                       const Vector2 &uv) {
    auto uv_scale = material.diffuse_uv_scale;
    return get_texture_value(material.diffuse_reflectance, uv * uv_scale);
}

DEVICE
inline void d_get_diffuse_reflectance(const Material &material,
                                      const Vector2 &uv, const Vector3 &d_output,
                                      DTexture3 &d_texture, Vector2 &d_uv) {
    auto uv_scale = material.diffuse_uv_scale;
    auto d_scaled_uv = Vector2{0, 0};
    d_get_texture_value(material.diffuse_reflectance, uv * uv_scale, d_output,
                        d_texture, d_scaled_uv);
    // scaled_uv = uv * uv_scale
    d_uv += d_scaled_uv * uv_scale;
    // d_material.d_diffuse_uv_scale += d_scaled_uv * uv;
}

DEVICE
inline Vector3 get_specular_reflectance(const Material &material,
                                        const Vector2 &uv) {
    auto uv_scale = material.specular_uv_scale;
    return get_texture_value(material.specular_reflectance, uv * uv_scale);
}

DEVICE
inline void d_get_specular_reflectance(const Material &material,
                                       const Vector2 &uv, const Vector3 &d_output,
                                       DTexture3 &d_texture, Vector2 &d_uv) {
    auto uv_scale = material.specular_uv_scale;
    auto d_scaled_uv = Vector2{0, 0};
    d_get_texture_value(material.specular_reflectance, uv * uv_scale, d_output,
                        d_texture, d_scaled_uv);
    // scaled_uv = uv * uv_scale
    d_uv += d_scaled_uv * uv_scale;
    // d_material.d_specular_uv_scale += d_scaled_uv * uv;
}

DEVICE
inline Real get_roughness(const Material &material, const Vector2 &uv) {
    auto uv_scale = material.roughness_uv_scale;
    return get_texture_value(material.roughness, uv * uv_scale);
}

DEVICE
inline void d_get_roughness(const Material &material, const Vector2 &uv,
                            Real d_output,
                            DTexture1 &d_texture, Vector2 &d_uv) {
    auto uv_scale = material.roughness_uv_scale;
    auto d_scaled_uv = Vector2{0, 0};
    d_get_texture_value(material.roughness, uv * uv_scale, d_output,
                        d_texture, d_scaled_uv);
    // scaled_uv = uv * uv_scale
    d_uv += d_scaled_uv * uv_scale;
    // d_material.d_roughness_uv_scale += d_scaled_uv * uv;
}


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
inline
Vector3 bsdf(const Material &material,
             const SurfacePoint &surface_point,
             const Vector3 &wi,
             const Vector3 &wo,
             const Real min_roughness) {
    auto shading_frame = surface_point.shading_frame;
    auto geom_n = surface_point.geom_normal;
    if (material.two_sided) {
        if (dot(wi, shading_frame.n) < 0.f) {
            shading_frame = -shading_frame;
            geom_n = -geom_n;
        }
    }
    auto bsdf_cos = dot(shading_frame.n, wo);
    auto geom_cos = dot(geom_n, wo);
    if (bsdf_cos <= 1e-3f || geom_cos <= 1e-3f) {
        // XXX: kind of hacky. we ignore extreme grazing angles
        // for numerical robustness
        return Vector3{0, 0, 0};
    }

    auto diffuse_reflectance = get_diffuse_reflectance(material, surface_point.uv);
    auto specular_reflectance = get_specular_reflectance(material, surface_point.uv);
    auto roughness = max(get_roughness(material, surface_point.uv), min_roughness);
    assert(roughness > 0.f);
    auto diffuse_contrib = diffuse_reflectance * bsdf_cos / Real(M_PI);
    auto specular_contrib = Vector3{0, 0, 0};
    if (sum(specular_reflectance) > 0.f) {
        // blinn-phong
        if (dot(wi, shading_frame.n) > 0.f) {
            // half-vector
            auto m = normalize(wi + wo);
            auto m_local = to_local(shading_frame, m);
            if (m_local[2] > 0.f) {
                auto phong_exponent = roughness_to_phong(roughness);
                auto D = pow(max(m_local[2], Real(0)), phong_exponent) *
                    (phong_exponent + 2.f) / Real(2 * M_PI);
                auto smithG1 = [&](const Vector3 &v) -> Real {
                    auto cos_theta = dot(v, shading_frame.n);
                    if (dot(v, m) * cos_theta <= 0) {
                        return 0;
                    }
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
                auto cos_theta_d = dot(m, wo);
                // Schlick's approximation
                auto F = specular_reflectance +
                    (1.f - specular_reflectance) *
                    pow(max(1.f - cos_theta_d, Real(0)), 5.f);
                specular_contrib = F * D * G / (4.f * dot(wi, shading_frame.n));
            }
        }
    }
    return diffuse_contrib + specular_contrib;
}

DEVICE
inline
void d_bsdf(const Material &material,
            const SurfacePoint &surface_point,
            const Vector3 &wi,
            const Vector3 &wo,
            const Real min_roughness,
            const Vector3 &d_output,
            DTexture3 &d_diffuse_tex,
            DTexture3 &d_specular_tex,
            DTexture1 &d_roughness_tex,
            SurfacePoint &d_surface_point,
            Vector3 &d_wi,
            Vector3 &d_wo) {
    auto shading_frame = surface_point.shading_frame;
    auto geom_n = surface_point.geom_normal;
    auto d_n = Vector3{0, 0, 0};
    bool flipped_normal = false;
    if (material.two_sided) {
        if (dot(wi, shading_frame.n) < 0.f) {
            flipped_normal = true;
            shading_frame = -shading_frame;
            geom_n = -geom_n;
        }
    }
    auto bsdf_cos = dot(shading_frame.n, wo);
    auto geom_cos = dot(geom_n, wo);
    if (bsdf_cos <= 1e-3f || geom_cos <= 1e-3f) {
        // XXX: kind of hacky. we ignore extreme grazing angles
        // for numerical robustness
        return;
    }

    auto diffuse_reflectance = get_diffuse_reflectance(material, surface_point.uv);
    // diffuse_contrib = diffuse_reflectance * bsdf_cos / Real(M_PI)
    auto d_diffuse_reflectance = d_output * (bsdf_cos / Real(M_PI));
    d_get_diffuse_reflectance(material, surface_point.uv, d_diffuse_reflectance,
                              d_diffuse_tex, d_surface_point.uv);
    auto d_bsdf_cos = d_output * sum(diffuse_reflectance) / Real(M_PI);
    // bsdf_cos = dot(shading_frame.n, wo)
    d_wo += shading_frame.n * d_bsdf_cos;
    d_n += wo * d_bsdf_cos;

    auto specular_reflectance = get_specular_reflectance(material, surface_point.uv);
    auto roughness = max(get_roughness(material, surface_point.uv), min_roughness);
    assert(roughness > 0.f);
    if (sum(specular_reflectance) > 0.f) {
        // blinn-phong
        if (dot(wi, shading_frame.n) > 0.f) {
            // half-vector
            auto m = normalize(wi + wo);
            auto m_local = to_local(shading_frame, m);
            if (m_local[2] > 0.f) {
                auto phong_exponent = roughness_to_phong(roughness);
                auto D = pow(m_local[2], phong_exponent) * (phong_exponent + 2.f) / Real(2 * M_PI);
                auto smithG1 = [&](const Vector3 &v) -> Real {
                    auto cos_theta = dot(v, shading_frame.n);
                    if (dot(v, m) * cos_theta <= 0) {
                        return 0;
                    }
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
                auto cos5 = pow(max(1.f - cos_theta_d, Real(0)), 5.f);
                auto F = specular_reflectance + (1.f - specular_reflectance) * cos5;
                auto cos_wi = dot(wi, shading_frame.n);
                auto specular_contrib = F * D * G / (4.f * cos_wi);

                // specular_contrib = F * D * G / (4.f * cos_wi)
                auto d_F = d_output * (D * G / (4.f * cos_wi));
                auto d_D = sum(d_output * F) * (G / (4.f * cos_wi));
                auto d_G = sum(d_output * F) * (D / (4.f * cos_wi));
                auto d_dot_wi_n = -sum(d_output * specular_contrib) / cos_wi;
                // dot_wi_n = dot(wi, shading_frame.n)
                d_wi += d_dot_wi_n * shading_frame.n;
                d_n += d_dot_wi_n * wi;
                // F = specular_reflectance + (1.f - specular_reflectance) * cos5
                auto d_specular_reflectance = d_F * (1.f - cos5);
                auto d_cos_5 = sum(d_F * (1.f - specular_reflectance));
                // cos5 = pow(max(1.f - cos_theta_d, Real(0)), 5.f)
                auto d_cos_theta_d = -5.f * d_cos_5 * pow(max(1.f - cos_theta_d, Real(0)), 4.f);
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
                // m_local = to_local(shading_frame, m)
                // This is an isotropic BRDF so only normal is affected
                d_m += d_m_local2 * shading_frame.n;
                d_n += d_m_local2 * m;
                // m = normalize(wi + wo)
                auto d_wi_wo = d_normalize(wi + wo, d_m);
                d_wi += d_wi_wo;
                d_wo += d_wi_wo;
                // specular_reflectance = get_specular_reflectance(material, surface_point.uv)
                d_get_specular_reflectance(material, surface_point.uv, d_specular_reflectance,
                    d_specular_tex, d_surface_point.uv);
                // roughness = get_roughness(material, surface_point.uv)
                if (roughness >= min_roughness) {
                    d_get_roughness(material, surface_point.uv, d_roughness,
                        d_roughness_tex, d_surface_point.uv);
                }
            }
        }
    }

    if (flipped_normal) {
        d_n = -d_n;
    }
    d_surface_point.shading_frame.n += d_n;
}

DEVICE
inline
Vector3 cos_hemisphere(const Vector2 &sample) {
    auto phi = 2.f * float(M_PI) * sample[0];
    auto tmp = sqrt(max(1.f - sample[1], Real(0)));
    return Vector3{
        cos(phi) * tmp, sin(phi) * tmp, sqrt(sample[1])
    };
}

DEVICE
inline
Vector3 bsdf_sample(const Material &material,
                    const SurfacePoint &surface_point,
                    const Vector3 &wi,
                    const BSDFSample &bsdf_sample,
                    const Real min_roughness,
                    Real *next_min_roughness = nullptr) {
    if (next_min_roughness != nullptr) {
        *next_min_roughness = min_roughness;
    }
    auto shading_frame = surface_point.shading_frame;
    auto cos_wi = dot(shading_frame.n, wi);
    if (material.two_sided && cos_wi < 0.f) {
        shading_frame = -shading_frame;
        cos_wi = -cos_wi;
    }
    if (cos_wi <= 0.f) {
        return Vector3{0, 0, 0};
    }

    auto diffuse_reflectance = get_diffuse_reflectance(material, surface_point.uv);
    auto specular_reflectance = get_specular_reflectance(material, surface_point.uv);
    // TODO: this is wrong for black materials
    auto diffuse_weight = luminance(diffuse_reflectance);
    auto specular_weight = luminance(specular_reflectance);
    auto weight_sum = diffuse_weight + specular_weight;
    if (weight_sum <= 0.f) {
        return Vector3{0, 0, 0};
    }
    auto diffuse_pmf = diffuse_weight / weight_sum;
    // auto specular_pmf = specular_weight / weight_sum;
    if (bsdf_sample.w <= diffuse_pmf) {
        // Lambertian
        if (diffuse_pmf <= 0.f) {
            return Vector3{0, 0, 0};
        }
        if (next_min_roughness != nullptr) {
            *next_min_roughness = Real(1);
        }
        auto local_dir = cos_hemisphere(bsdf_sample.uv);
        return to_world(shading_frame, local_dir);
    } else {
        if (specular_weight <= 0.f) {
            return Vector3{0, 0, 0};
        }
        // Blinn-phong
        auto roughness = max(get_roughness(material, surface_point.uv), min_roughness);
        if (next_min_roughness != nullptr) {
            *next_min_roughness = max(roughness, min_roughness);
        }
        auto phong_exponent = roughness_to_phong(roughness);
        // Sample phi
        auto phi = 2.f * Real(M_PI) * bsdf_sample.uv[1];
        auto sin_phi = sin(phi);
        auto cos_phi = cos(phi);
        
        // Sample theta
        auto cos_theta = pow(bsdf_sample.uv[0], 1.0f / (phong_exponent + 2.0f));
        auto sin_theta = sqrt(max(1.f - cos_theta * cos_theta, Real(0)));
        // local microfacet normal
        auto m_local = Vector3{sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
        auto m = to_world(shading_frame, m_local);
        auto dir = 2.f * dot(wi, m) * m - wi;
        return dir;
    }
}

DEVICE
inline
void d_bsdf_sample(const Material &material,
                   const SurfacePoint &surface_point,
                   const Vector3 &wi,
                   const BSDFSample &bsdf_sample,
                   const Real min_roughness,
                   const Vector3 &d_wo,
                   DTexture1 &d_roughness_tex,
                   SurfacePoint &d_surface_point,
                   Vector3 &d_wi) {
    auto shading_frame = surface_point.shading_frame;
    auto cos_wi = dot(shading_frame.n, wi);
    bool normal_flipped = false;
    if (material.two_sided && cos_wi < 0.f) {
        shading_frame = -shading_frame;
        cos_wi = -cos_wi;
        normal_flipped = true;
    }
    if (cos_wi <= 0.f) {
        return;
    }

    auto diffuse_reflectance = get_diffuse_reflectance(material, surface_point.uv);
    auto specular_reflectance = get_specular_reflectance(material, surface_point.uv);
    // TODO: this is wrong for black materials
    auto diffuse_weight = luminance(diffuse_reflectance);
    auto specular_weight = luminance(specular_reflectance);
    auto weight_sum = diffuse_weight + specular_weight;
    if (weight_sum <= 0.f) {
        return;
    }
    auto diffuse_pmf = diffuse_weight / weight_sum;
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
        // wo = to_world(shading_frame, local_dir)
        auto d_local_dir = Vector3{0, 0, 0};
        d_to_world(shading_frame, local_dir, d_wo, d_shading_frame, d_local_dir);
        // No need to propagate to bsdf_sample
    } else {
        if (specular_weight <= 0.f) {
            return;
        }
        // Blinn-phong
        auto roughness = max(get_roughness(material, surface_point.uv), min_roughness);
        auto phong_exponent = roughness_to_phong(roughness);
        // Sample phi
        auto phi = 2.f * Real(M_PI) * bsdf_sample.uv[1];
        auto sin_phi = sin(phi);
        auto cos_phi = cos(phi);
        
        // Sample theta
        auto cos_theta = pow(bsdf_sample.uv[0], 1.0f / (phong_exponent + 2.0f));
        auto sin_theta = sqrt(max(1.f - cos_theta*cos_theta, Real(0)));
        // local microfacet normal
        auto m_local = Vector3{sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
        auto m = to_world(shading_frame, m_local);

        // wo = 2.f * dot(wi, m) * m - wi
        auto d_dot_wi_m = 2.f * sum(d_wo * m);
        auto d_m = d_wo * (2.f * dot(wi, m));
        d_wi += -d_wo;
        // dot_wi_m = dot(wi, m)
        d_wi += d_dot_wi_m * m;
        d_m += d_dot_wi_m * wi;
        // m = to_world(shading_frame, m_local)
        auto d_m_local = Vector3{0, 0, 0};
        d_to_world(shading_frame, m_local, d_m, d_shading_frame, d_m_local);
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
        // cos_theta = pow(bsdf_sample.uv[0], 1.0f / (phong_exponent + 2.0f))
        auto d_one_over_phong_exponent_plus_2 =
            bsdf_sample.uv[0] > 0 ? d_cos_theta * cos_theta * log(bsdf_sample.uv[0]) : Real(0);
        // 1 / (phong_exponent + 2)
        auto d_phong_exponent = -d_one_over_phong_exponent_plus_2 / square(phong_exponent + 2.0f);
        // phong_exponent = roughness_to_phong(roughness)
        auto d_roughness = d_roughness_to_phong(roughness, d_phong_exponent);
        // roughness = get_roughness(material, surface_point.uv)
        if (roughness >= min_roughness) {
            d_get_roughness(material,
                            surface_point.uv,
                            d_roughness,
                            d_roughness_tex,
                            d_surface_point.uv);
        }
    }

    if (normal_flipped) {
        d_shading_frame = - d_shading_frame;
    }
    d_surface_point.shading_frame += d_shading_frame;
}

DEVICE
inline Real bsdf_pdf(const Material &material,
                     const SurfacePoint &surface_point,
                     const Vector3 &wi,
                     const Vector3 &wo,
                     const Real min_roughness) {
    auto shading_frame = surface_point.shading_frame;
    auto cos_wi = dot(shading_frame.n, wi);
    if (material.two_sided) {
        if (cos_wi < 0) {
            shading_frame = -shading_frame;
            cos_wi = -cos_wi;
        }
    }
    if (cos_wi <= 0) {
        return 0;
    }
    auto diffuse_reflectance = get_diffuse_reflectance(material, surface_point.uv);
    auto specular_reflectance = get_specular_reflectance(material, surface_point.uv);
    // TODO: this is wrong for black materials
    auto diffuse_weight = luminance(diffuse_reflectance);
    auto specular_weight = luminance(specular_reflectance);
    auto weight_sum = diffuse_weight + specular_weight;
    if (weight_sum <= 0.f) {
        return 0;
    }
    auto diffuse_pmf = diffuse_weight / weight_sum;
    auto specular_pmf = specular_weight / weight_sum;
    auto diffuse_pdf = Real(0);
    if (diffuse_pmf > 0.f) {
        auto bsdf_cos = dot(shading_frame.n, wo);
        if (!material.two_sided) {
            bsdf_cos = max(bsdf_cos, Real(0));
        } else {
            bsdf_cos = fabs(bsdf_cos);
        }
        diffuse_pdf = diffuse_pmf * bsdf_cos / Real(M_PI);
    }
    auto specular_pdf = Real(0);
    if (specular_pmf > 0.f) {
        auto m = normalize(wi + wo);
        auto m_local = to_local(surface_point.shading_frame, m);
        if (m_local[2] > 0.f) {
            auto roughness = max(get_roughness(material, surface_point.uv), min_roughness);
            auto phong_exponent = roughness_to_phong(roughness);
            auto D = pow(m_local[2], phong_exponent) * (phong_exponent + 2.f) / Real(2 * M_PI);
            specular_pdf = specular_pmf * D * m_local[2] / (4.f * fabs(dot(m, wo)));
        }
    }
    return diffuse_pdf + specular_pdf;
}

DEVICE
inline void d_bsdf_pdf(const Material &material,
                       const SurfacePoint &surface_point,
                       const Vector3 &wi,
                       const Vector3 &wo,
                       const Real min_roughness,
                       const Real d_pdf,
                       DTexture1 &d_roughness_tex,
                       SurfacePoint &d_surface_point,
                       Vector3 &d_wi,
                       Vector3 &d_wo) {
    auto shading_frame = surface_point.shading_frame;
    auto cos_wi = dot(shading_frame.n, wi);
    bool normal_flipped = false;
    if (material.two_sided) {
        if (cos_wi < 0) {
            shading_frame = -shading_frame;
            cos_wi = -cos_wi;
            normal_flipped = true;
        }
    }
    if (cos_wi <= 0) {
        return;
    }
    auto diffuse_reflectance = get_diffuse_reflectance(material, surface_point.uv);
    auto specular_reflectance = get_specular_reflectance(material, surface_point.uv);
    // TODO: this is wrong for black materials
    auto diffuse_weight = luminance(diffuse_reflectance);
    auto specular_weight = luminance(specular_reflectance);
    auto weight_sum = diffuse_weight + specular_weight;
    if (weight_sum <= 0.f) {
        return;
    }
    auto diffuse_pmf = diffuse_weight / weight_sum;
    auto specular_pmf = specular_weight / weight_sum;
    auto bsdf_cos = dot(surface_point.shading_frame.n, wo);
    // auto unsigned_bsdf_cos = bsdf_cos;
    // if (!material.two_sided) {
    //     unsigned_bsdf_cos = max(bsdf_cos, Real(0));
    // } else {
    //     unsigned_bsdf_cos = fabs(bsdf_cos);
    // }
    // Diffuse PDF
    // auto diffuse_pdf = diffuse_pmf * unsigned_bsdf_cos / Real(M_PI);
    auto d_n = Vector3{0, 0, 0};
    if (diffuse_weight > 0) {
        // Ignore derivatives to discrete probability
        auto d_unsigned_bsdf_cos = d_pdf * diffuse_pmf / Real(M_PI);
        auto d_bsdf_cos = 0.f;
        if (!material.two_sided) {
            if (bsdf_cos >= 0) {
                d_bsdf_cos += d_unsigned_bsdf_cos;
            }
        } else {
            if (bsdf_cos >= 0) {
                d_bsdf_cos += d_unsigned_bsdf_cos;
            } else {
                d_bsdf_cos -= d_unsigned_bsdf_cos;
            }
        }
        // bsdf_cos = dot(shading_frame.n, wo)
        d_wo += shading_frame.n * d_bsdf_cos;
        d_n += wo * d_bsdf_cos;
    }

    if (specular_weight > 0) {
        auto m = normalize(wi + wo);
        auto m_local = to_local(surface_point.shading_frame, m);
        if (m_local[2] > 0.f) {
            auto roughness = max(get_roughness(material, surface_point.uv), min_roughness);
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
            // m_local = to_local(surface_point.shading_frame, m)
            // This is an isotropic BRDF so only normal is affected
            d_m += d_m_local2 * shading_frame.n;
            d_n += d_m_local2 * m;
            // m = normalize(wi + wo)
            auto d_wi_wo = d_normalize(wi + wo, d_m);
            d_wi += d_wi_wo;
            d_wo += d_wi_wo;
            // roughness = get_roughness(material, surface_point.uv)
            if (roughness >= min_roughness) {
                d_get_roughness(material,
                                surface_point.uv,
                                d_roughness,
                                d_roughness_tex,
                                d_surface_point.uv);
            }
        }
    }

    if (normal_flipped) {
        d_n = -d_n;
    }
    d_surface_point.shading_frame.n += d_n;
}

struct Scene;

void accumulate_diffuse(const Scene &scene,
                        const BufferView<DTexture3> &d_texs,
                        BufferView<DMaterial> d_materials);
void accumulate_specular(const Scene &scene,
                         const BufferView<DTexture3> &d_texs,
                         BufferView<DMaterial> d_materials);
void accumulate_roughness(const Scene &scene,
                          const BufferView<DTexture1> &d_texs,
                          BufferView<DMaterial> d_materials);

void test_d_bsdf();
void test_d_bsdf_sample();
void test_d_bsdf_pdf();
