#include "material.h"
#include "scene.h"
#include "parallel.h"
#include "test_utils.h"

void test_d_bsdf() {
    Vector3f d{0.5, 0.4, 0.3};
    Vector2f uv_scale{1, 1};
    Texture3 diffuse{{&d[0]}, {-1}, {-1}, -1, &uv_scale[0]};
    Vector3f s{0.2, 0.3, 0.4};
    Texture3 specular{{&s[0]}, {-1}, {-1}, -1, &uv_scale[0]};
    float r = 0.5;
    Texture1 roughness{{&r}, {-1}, {-1}, -1, &uv_scale[0]};
    TextureN generic{{&d[0]}, {-1}, {-1}, 3 /* channels */, &uv_scale[0]};
    Texture3 normal_map{{nullptr}, {0}, {0}, 0, nullptr};
    Material m{diffuse,
               specular,
               roughness,
               generic,
               normal_map,
               true, // compute_specular_lighting
               false, // two_sided
               false}; // use_vertex_color
    Vector3f d_d{0, 0, 0};
    Vector2f d_uv_scale{0, 0};
    Texture3 d_diffuse_tex{{&d_d[0]}, {-1}, {-1}, -1, &d_uv_scale[0]};
    Vector3f d_s{0, 0, 0};
    Texture3 d_specular_tex{{&d_s[0]}, {-1}, {-1}, -1, &d_uv_scale[0]};
    float d_r = 0.f;
    Texture1 d_roughness_tex{{&d_r}, {-1}, {-1}, -1, &d_uv_scale[0]};
    TextureN d_generic_tex{{&d_d[0]}, {-1}, {-1}, 3 /* channels */, &d_uv_scale[0]};
    Texture3 d_normal_map{{nullptr}, {0}, {0}, 0, nullptr};
    DMaterial d_material{d_diffuse_tex, d_specular_tex, d_roughness_tex, d_generic_tex, d_normal_map};
    SurfacePoint p{Vector3{0, 0, 0},
                   Vector3{0, 1, 0},
                   Frame(Vector3{0, 1, 0}),
                   Vector3{1, 0, 0}, // dpdu
                   Vector2{0.5, 0.5}, // uv
                   Vector2{0, 0}, Vector2{0, 0}, // du_dxy & dv_dxy
                   Vector3{0, 0, 0}, Vector3{0, 0, 0}, // dn_dx, dn_dy
                   Vector3{0, 0, 0}}; // color
    auto wi = normalize(Vector3{0.5, 1.0, 0.5});
    auto wo = normalize(Vector3{-0.5, 1.0, -0.5});
    auto min_roughness = Real(0);
    auto d_p = SurfacePoint::zero();
    auto d_wi = Vector3{0, 0, 0};
    auto d_wo = Vector3{0, 0, 0};

    d_bsdf(m, p, wi, wo, min_roughness, Vector3{1, 1, 1},
           d_material, d_p, d_wi, d_wo);

    // Check diffuse derivatives
    auto finite_delta = Real(1e-6);
    for (int i = 0; i < 3; i++) {
        auto delta_m = m;
        delta_m.diffuse_reflectance.texels[0][i] += finite_delta;
        auto positive = bsdf(delta_m, p, wi, wo, min_roughness);
        delta_m.diffuse_reflectance.texels[0][i] -= 2 * finite_delta;
        auto negative = bsdf(delta_m, p, wi, wo, min_roughness);
        auto diff = sum(positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_d[i]);
    }

    // Check specular derivatives
    for (int i = 0; i < 3; i++) {
        auto delta_m = m;
        delta_m.specular_reflectance.texels[0][i] += finite_delta;
        auto positive = bsdf(delta_m, p, wi, wo, min_roughness);
        delta_m.specular_reflectance.texels[0][i] -= 2 * finite_delta;
        auto negative = bsdf(delta_m, p, wi, wo, min_roughness);
        auto diff = sum(positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_s[i]);
    }

    // Check roughness derivatives
    {
        auto delta_m = m;
        delta_m.roughness.texels[0][0] += finite_delta;
        auto positive = bsdf(delta_m, p, wi, wo, min_roughness);
        delta_m.roughness.texels[0][0] -= 2 * finite_delta;
        auto negative = bsdf(delta_m, p, wi, wo, min_roughness);
        auto diff = sum(positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_r);
    }

    // Check surface point derivatives
    equal_or_error<Real>(__FILE__, __LINE__, Vector3{0, 0, 0}, d_p.position);
    equal_or_error<Real>(__FILE__, __LINE__, Vector3{0, 0, 0}, d_p.geom_normal);
    // Shading frame x
    for (int i = 0; i < 3; i++) {
        auto delta_p = p;
        delta_p.shading_frame.x[i] += finite_delta;
        auto positive = bsdf(m, delta_p, wi, wo, min_roughness);
        delta_p.shading_frame.x[i] -= 2 * finite_delta;
        auto negative = bsdf(m, delta_p, wi, wo, min_roughness);
        auto diff = sum(positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_p.shading_frame.x[i]);
    }
    // Shading frame y
    for (int i = 0; i < 3; i++) {
        auto delta_p = p;
        delta_p.shading_frame.y[i] += finite_delta;
        auto positive = bsdf(m, delta_p, wi, wo, min_roughness);
        delta_p.shading_frame.y[i] -= 2 * finite_delta;
        auto negative = bsdf(m, delta_p, wi, wo, min_roughness);
        auto diff = sum(positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_p.shading_frame.y[i]);
    }
    // Shading frame n
    for (int i = 0; i < 3; i++) {
        auto delta_p = p;
        delta_p.shading_frame.n[i] += finite_delta;
        auto positive = bsdf(m, delta_p, wi, wo, min_roughness);
        delta_p.shading_frame.n[i] -= 2 * finite_delta;
        auto negative = bsdf(m, delta_p, wi, wo, min_roughness);
        auto diff = sum(positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_p.shading_frame.n[i]);
    }
    // uv
    for (int i = 0; i < 2; i++) {
        auto delta_p = p;
        delta_p.uv[i] += finite_delta;
        auto positive = bsdf(m, delta_p, wi, wo, min_roughness);
        delta_p.uv[i] -= 2 * finite_delta;
        auto negative = bsdf(m, delta_p, wi, wo, min_roughness);
        auto diff = sum(positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_p.uv[i]);
    }

    // Check wi, wo
    for (int i = 0; i < 3; i++) {
        auto delta_wi = wi;
        delta_wi[i] += finite_delta;
        auto positive = bsdf(m, p, delta_wi, wo, min_roughness);
        delta_wi[i] -= 2 * finite_delta;
        auto negative = bsdf(m, p, delta_wi, wo, min_roughness);
        auto diff = sum(positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_wi[i]);
    }
    for (int i = 0; i < 3; i++) {
        auto delta_wo = wo;
        delta_wo[i] += finite_delta;
        auto positive = bsdf(m, p, wi, delta_wo, min_roughness);
        delta_wo[i] -= 2 * finite_delta;
        auto negative = bsdf(m, p, wi, delta_wo, min_roughness);
        auto diff = sum(positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_wo[i]);
    }
}

void test_d_bsdf_sample() {
    Vector2f uv_scale = Vector2f{1, 1};
    Vector3f d{0.5, 0.4, 0.3};
    Texture3 diffuse{{&d[0]}, {-1}, {-1}, -1, &uv_scale[0]};
    Vector3f s{0.2, 0.3, 0.4};
    Texture3 specular{{&s[0]}, {-1}, {-1}, -1, &uv_scale[0]};
    float r = 0.5;
    Texture1 roughness{{&r}, {-1}, {-1}, -1, &uv_scale[0]};
    TextureN generic{{&d[0]}, {-1}, {-1}, 3 /* channels*/, &uv_scale[0]};
    Texture3 normal_map{{nullptr}, {0}, {0}, 0, nullptr};
    Material m{diffuse,
               specular,
               roughness,
               generic,
               normal_map,
               true, // compute_specular_lighting
               false, // two_sided
               false}; // use_vertex_color
    Vector3f d_d{0, 0, 0};
    Vector2f d_uv_scale{0, 0};
    Texture3 d_diffuse_tex{{&d_d[0]}, {-1}, {-1}, -1, &d_uv_scale[0]};
    Vector3f d_s{0, 0, 0};
    Texture3 d_specular_tex{{&d_s[0]}, {-1}, {-1}, -1, &d_uv_scale[0]};
    float d_r = 0.f;
    Texture1 d_roughness_tex{{&d_r}, {-1}, {-1}, -1, &d_uv_scale[0]};
    TextureN d_generic_tex{{&d_d[0]}, {-1}, {-1}, 3 /* channels */, &d_uv_scale[0]};
    Texture3 d_normal_map{{nullptr}, {0}, {0}, 0, nullptr};
    DMaterial d_material{d_diffuse_tex, d_specular_tex, d_roughness_tex, d_generic_tex, d_normal_map};
    SurfacePoint p{Vector3{0, 0, 0},
                   Vector3{0, 1, 0},
                   Frame(Vector3{0, 1, 0}),
                   Vector3{1, 0, 0}, // dpdu
                   Vector2{0.5, 0.5}, // uv
                   Vector2{1, 1}, Vector2{1, 1}, // du_dxy, dv_dxy
                   Vector3{1, 1, 1}, Vector3{1, 1, 1}, // dn_dx, dn_dy
                   Vector3{0, 0, 0}}; // color
    auto wi = normalize(Vector3{0.5, 1.0, 0.5});
    auto wi_differential = RayDifferential{
        Vector3{1, 1, 1}, Vector3{1, 1, 1},
        Vector3{1, 1, 1}, Vector3{1, 1, 1}};
    auto min_roughness = Real(0.0);
    for (int j = 0; j < 2; j++) {
        // Test for both specular and diffuse path
        auto sample = j == 0 ?
            BSDFSample{Vector2{0.5, 0.5}, 0.0} :
            BSDFSample{Vector2{0.5, 0.5}, 0.99};
        auto d_wo_differential = RayDifferential{
            Vector3{1, 1, 1}, Vector3{1, 1, 1},
            Vector3{1, 1, 1}, Vector3{1, 1, 1}};
        auto d_p = SurfacePoint::zero();
        auto d_wi = Vector3{0, 0, 0};
        auto d_wi_differential = RayDifferential{
            Vector3{0, 0, 0}, Vector3{0, 0, 0},
            Vector3{0, 0, 0}, Vector3{0, 0, 0}};

        d_bsdf_sample(m,
                      p,
                      wi,
                      sample,
                      min_roughness,
                      wi_differential,
                      Vector3{1, 1, 1},
                      d_wo_differential,
                      d_material,
                      d_p,
                      d_wi,
                      d_wi_differential);

        // Check roughness derivatives
        auto finite_delta = Real(1e-4);
        {
            auto ray_diff_pos = RayDifferential{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
                Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            auto ray_diff_neg = RayDifferential{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
                Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            auto delta_m = m;
            delta_m.roughness.texels[0][0] += finite_delta;
            auto positive = bsdf_sample(delta_m, p, wi, sample, min_roughness,
                wi_differential, ray_diff_pos);
            delta_m.roughness.texels[0][0] -= 2 * finite_delta;
            auto negative = bsdf_sample(delta_m, p, wi, sample, min_roughness,
                wi_differential, ray_diff_neg);
            auto diff = (sum(positive - negative) +
                     sum(ray_diff_pos.org_dx - ray_diff_neg.org_dx) +
                     sum(ray_diff_pos.org_dy - ray_diff_neg.org_dy) +
                     sum(ray_diff_pos.dir_dx - ray_diff_neg.dir_dx) +
                     sum(ray_diff_pos.dir_dy - ray_diff_neg.dir_dy))
                    / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, diff, d_r);
        }

        // Check surface point derivatives
        equal_or_error<Real>(__FILE__, __LINE__, Vector3{0, 0, 0}, d_p.position);
        equal_or_error<Real>(__FILE__, __LINE__, Vector3{0, 0, 0}, d_p.geom_normal);
        // Shading frame x
        for (int i = 0; i < 3; i++) {
            auto ray_diff_pos = RayDifferential{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
                Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            auto ray_diff_neg = RayDifferential{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
                Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            auto delta_p = p;
            delta_p.shading_frame.x[i] += finite_delta;
            auto positive = bsdf_sample(m, delta_p, wi, sample, min_roughness,
                wi_differential, ray_diff_pos);
            delta_p.shading_frame.x[i] -= 2 * finite_delta;
            auto negative = bsdf_sample(m, delta_p, wi, sample, min_roughness,
                wi_differential, ray_diff_neg);
            auto diff = (sum(positive - negative) +
                     sum(ray_diff_pos.org_dx - ray_diff_neg.org_dx) +
                     sum(ray_diff_pos.org_dy - ray_diff_neg.org_dy) +
                     sum(ray_diff_pos.dir_dx - ray_diff_neg.dir_dx) +
                     sum(ray_diff_pos.dir_dy - ray_diff_neg.dir_dy))
                    / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, diff, d_p.shading_frame.x[i]);
        }
        // Shading frame y
        for (int i = 0; i < 3; i++) {
            auto ray_diff_pos = RayDifferential{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
                Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            auto ray_diff_neg = RayDifferential{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
                Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            auto delta_p = p;
            delta_p.shading_frame.y[i] += finite_delta;
            auto positive = bsdf_sample(m, delta_p, wi, sample, min_roughness,
                wi_differential, ray_diff_pos);
            delta_p.shading_frame.y[i] -= 2 * finite_delta;
            auto negative = bsdf_sample(m, delta_p, wi, sample, min_roughness,
                wi_differential, ray_diff_neg);
            auto diff = (sum(positive - negative) +
                     sum(ray_diff_pos.org_dx - ray_diff_neg.org_dx) +
                     sum(ray_diff_pos.org_dy - ray_diff_neg.org_dy) +
                     sum(ray_diff_pos.dir_dx - ray_diff_neg.dir_dx) +
                     sum(ray_diff_pos.dir_dy - ray_diff_neg.dir_dy))
                    / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, diff, d_p.shading_frame.y[i]);
        }
        // Shading frame n
        for (int i = 0; i < 3; i++) {
            auto ray_diff_pos = RayDifferential{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
                Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            auto ray_diff_neg = RayDifferential{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
                Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            auto delta_p = p;
            delta_p.shading_frame.n[i] += finite_delta;
            auto positive = bsdf_sample(m, delta_p, wi, sample, min_roughness,
                wi_differential, ray_diff_pos);
            delta_p.shading_frame.n[i] -= 2 * finite_delta;
            auto negative = bsdf_sample(m, delta_p, wi, sample, min_roughness,
                wi_differential, ray_diff_neg);
            auto diff = (sum(positive - negative) +
                     sum(ray_diff_pos.org_dx - ray_diff_neg.org_dx) +
                     sum(ray_diff_pos.org_dy - ray_diff_neg.org_dy) +
                     sum(ray_diff_pos.dir_dx - ray_diff_neg.dir_dx) +
                     sum(ray_diff_pos.dir_dy - ray_diff_neg.dir_dy))
                    / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, diff, d_p.shading_frame.n[i]);
        }
        // uv
        for (int i = 0; i < 2; i++) {
            auto ray_diff_pos = RayDifferential{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
                Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            auto ray_diff_neg = RayDifferential{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
                Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            auto delta_p = p;
            delta_p.uv[i] += finite_delta;
            auto positive = bsdf_sample(m, delta_p, wi, sample, min_roughness,
                wi_differential, ray_diff_pos);
            delta_p.uv[i] -= 2 * finite_delta;
            auto negative = bsdf_sample(m, delta_p, wi, sample, min_roughness,
                wi_differential, ray_diff_neg);
            auto diff = (sum(positive - negative) +
                     sum(ray_diff_pos.org_dx - ray_diff_neg.org_dx) +
                     sum(ray_diff_pos.org_dy - ray_diff_neg.org_dy) +
                     sum(ray_diff_pos.dir_dx - ray_diff_neg.dir_dx) +
                     sum(ray_diff_pos.dir_dy - ray_diff_neg.dir_dy))
                    / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, diff, d_p.uv[i]);
        }

        // wi
        for (int i = 0; i < 3; i++) {
            auto ray_diff_pos = RayDifferential{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
                Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            auto ray_diff_neg = RayDifferential{
                Vector3{0, 0, 0}, Vector3{0, 0, 0},
                Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            auto delta_wi = wi;
            delta_wi[i] += finite_delta;
            auto positive = bsdf_sample(m, p, delta_wi, sample, min_roughness,
                wi_differential, ray_diff_pos);
            delta_wi[i] -= 2 * finite_delta;
            auto negative = bsdf_sample(m, p, delta_wi, sample, min_roughness,
                wi_differential, ray_diff_neg);
            auto diff = (sum(positive - negative) +
                     sum(ray_diff_pos.org_dx - ray_diff_neg.org_dx) +
                     sum(ray_diff_pos.org_dy - ray_diff_neg.org_dy) +
                     sum(ray_diff_pos.dir_dx - ray_diff_neg.dir_dx) +
                     sum(ray_diff_pos.dir_dy - ray_diff_neg.dir_dy))
                    / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, diff, d_wi[i], Real(5e-2f));
        }
    }
}

void test_d_bsdf_pdf() {
    Vector2f uv_scale = Vector2f{1, 1};
    Vector3f d{0.5, 0.4, 0.3};
    Texture3 diffuse{{&d[0]}, {-1}, {-1}, -1, &uv_scale[0]};
    Vector3f s{0.2, 0.3, 0.4};
    Texture3 specular{{&s[0]}, {-1}, {-1}, -1, &uv_scale[0]};
    float r = 0.5;
    Texture1 roughness{{&r}, {-1}, {-1}, -1, &uv_scale[0]};
    TextureN generic{{&d[0]}, {-1}, {-1}, 3 /* channels*/, &uv_scale[0]};
    Texture3 normal_map{{nullptr}, {0}, {0}, 0, nullptr};
    Material m{diffuse,
               specular,
               roughness,
               generic,
               normal_map,
               true, // compute_specular_lighting
               false, // two_sided
               false}; // use_vertex_color
    Vector3f d_d{0, 0, 0};
    Vector2f d_uv_scale{0, 0};
    Texture3 d_diffuse_tex{{&d_d[0]}, {-1}, {-1}, -1, &d_uv_scale[0]};
    Vector3f d_s{0, 0, 0};
    Texture3 d_specular_tex{{&d_s[0]}, {-1}, {-1}, -1, &d_uv_scale[0]};
    float d_r = 0.f;
    Texture1 d_roughness_tex{{&d_r}, {-1}, {-1}, -1, &d_uv_scale[0]};
    TextureN d_generic_tex{{&d_d[0]}, {-1}, {-1}, 3 /* channels*/, &d_uv_scale[0]};
    Texture3 d_normal_map{{nullptr}, {0}, {0}, 0, nullptr};
    DMaterial d_material{d_diffuse_tex, d_specular_tex, d_roughness_tex, d_generic_tex, d_normal_map};
    SurfacePoint p{Vector3{0, 0, 0},
                   Vector3{0, 1, 0},
                   Frame(Vector3{0, 1, 0}),
                   Vector3{1, 0, 0}, // dpdu
                   Vector2{0.5, 0.5}, // uv
                   Vector2{0, 0}, Vector2{0, 0}, // du_dxy, dv_dxy
                   Vector3{0, 0, 0}, Vector3{0, 0, 0}, // dn_dx, dn_dy
                   Vector3{0, 0, 0}}; // color
    auto wi = normalize(Vector3{0.5, 1.0, 0.5});
    auto wo = normalize(Vector3{-0.5, 1.0, -0.5});
    auto d_p = SurfacePoint::zero();
    auto d_wi = Vector3{0, 0, 0};
    auto d_wo = Vector3{0, 0, 0};
    auto min_roughness = Real(0.0);

    d_bsdf_pdf(m, p, wi, wo, min_roughness, 1,
               d_material, d_p, d_wi, d_wo);

    // Check roughness derivatives
    auto finite_delta = Real(1e-5);
    {
        auto delta_m = m;
        delta_m.roughness.texels[0][0] += finite_delta;
        auto positive = bsdf_pdf(delta_m, p, wi, wo, min_roughness);
        delta_m.roughness.texels[0][0] -= 2 * finite_delta;
        auto negative = bsdf_pdf(delta_m, p, wi, wo, min_roughness);
        auto diff = (positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_r);
    }

    // Check surface point derivatives
    equal_or_error<Real>(__FILE__, __LINE__, Vector3{0, 0, 0}, d_p.position);
    equal_or_error<Real>(__FILE__, __LINE__, Vector3{0, 0, 0}, d_p.geom_normal);
    // Shading frame x
    for (int i = 0; i < 3; i++) {
        auto delta_p = p;
        delta_p.shading_frame.x[i] += finite_delta;
        auto positive = bsdf_pdf(m, delta_p, wi, wo, min_roughness);
        delta_p.shading_frame.x[i] -= 2 * finite_delta;
        auto negative = bsdf_pdf(m, delta_p, wi, wo, min_roughness);
        auto diff = (positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_p.shading_frame.x[i]);
    }
    // Shading frame y
    for (int i = 0; i < 3; i++) {
        auto delta_p = p;
        delta_p.shading_frame.y[i] += finite_delta;
        auto positive = bsdf_pdf(m, delta_p, wi, wo, min_roughness);
        delta_p.shading_frame.y[i] -= 2 * finite_delta;
        auto negative = bsdf_pdf(m, delta_p, wi, wo, min_roughness);
        auto diff = (positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_p.shading_frame.y[i]);
    }
    // Shading frame n
    for (int i = 0; i < 3; i++) {
        auto delta_p = p;
        delta_p.shading_frame.n[i] += finite_delta;
        auto positive = bsdf_pdf(m, delta_p, wi, wo, min_roughness);
        delta_p.shading_frame.n[i] -= 2 * finite_delta;
        auto negative = bsdf_pdf(m, delta_p, wi, wo, min_roughness);
        auto diff = (positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_p.shading_frame.n[i]);
    }
    // uv
    for (int i = 0; i < 2; i++) {
        auto delta_p = p;
        delta_p.uv[i] += finite_delta;
        auto positive = bsdf_pdf(m, delta_p, wi, wo, min_roughness);
        delta_p.uv[i] -= 2 * finite_delta;
        auto negative = bsdf_pdf(m, delta_p, wi, wo, min_roughness);
        auto diff = (positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_p.uv[i]);
    }

    // Check wi, wo
    for (int i = 0; i < 3; i++) {
        auto delta_wi = wi;
        delta_wi[i] += finite_delta;
        auto positive = bsdf_pdf(m, p, delta_wi, wo, min_roughness);
        delta_wi[i] -= 2 * finite_delta;
        auto negative = bsdf_pdf(m, p, delta_wi, wo, min_roughness);
        auto diff = (positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_wi[i]);
    }
    for (int i = 0; i < 3; i++) {
        auto delta_wo = wo;
        delta_wo[i] += finite_delta;
        auto positive = bsdf_pdf(m, p, wi, delta_wo, min_roughness);
        delta_wo[i] -= 2 * finite_delta;
        auto negative = bsdf_pdf(m, p, wi, delta_wo, min_roughness);
        auto diff = (positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_wo[i]);
    }
}
