#include "material.h"
#include "scene.h"
#include "parallel.h"
#include "test_utils.h"

struct diffuse_accumulator {
    DEVICE
    inline void operator()(int idx) {
        const auto &d_tex = d_diffuse_texs[idx];
        auto mid = d_tex.material_id;
        auto xi = d_tex.xi;
        auto yi = d_tex.yi;
        auto texels = d_materials[mid].diffuse_reflectance.texels;
        if (xi < 0) {
            texels[0] += d_tex.t00[0];
            texels[1] += d_tex.t00[1];
            texels[2] += d_tex.t00[2];
        } else {
            auto w = d_materials[mid].diffuse_reflectance.width;
            auto h = d_materials[mid].diffuse_reflectance.height;
            auto xi0 = xi;
            auto xi1 = modulo(xi + 1, w);
            auto yi0 = yi;
            auto yi1 = modulo(yi + 1, h);
#ifdef __CUDA_ARCH__ 
            // Different DTexture may overlap, so we need to use atomic updates
            // The probability of collision should be small in SIMD regime though
            atomic_add(texels[3 * (yi0 * w + xi0) + 0], d_tex.t00[0]);
            atomic_add(texels[3 * (yi0 * w + xi0) + 1], d_tex.t00[1]);
            atomic_add(texels[3 * (yi0 * w + xi0) + 2], d_tex.t00[2]);
            atomic_add(texels[3 * (yi0 * w + xi1) + 0], d_tex.t10[0]);
            atomic_add(texels[3 * (yi0 * w + xi1) + 1], d_tex.t10[1]);
            atomic_add(texels[3 * (yi0 * w + xi1) + 2], d_tex.t10[2]);
            atomic_add(texels[3 * (yi1 * w + xi0) + 0], d_tex.t01[0]);
            atomic_add(texels[3 * (yi1 * w + xi0) + 1], d_tex.t01[1]);
            atomic_add(texels[3 * (yi1 * w + xi0) + 2], d_tex.t01[2]);
            atomic_add(texels[3 * (yi1 * w + xi1) + 0], d_tex.t11[0]);
            atomic_add(texels[3 * (yi1 * w + xi1) + 1], d_tex.t11[1]);
            atomic_add(texels[3 * (yi1 * w + xi1) + 2], d_tex.t11[2]);
#else
            // Lock at material level. Slow but probably not bottleneck.
            std::unique_lock<std::mutex> guard(((std::mutex*)mutexes)[mid]);
            texels[3 * (yi0 * w + xi0) + 0] += d_tex.t00[0];
            texels[3 * (yi0 * w + xi0) + 1] += d_tex.t00[1];
            texels[3 * (yi0 * w + xi0) + 2] += d_tex.t00[2];
            texels[3 * (yi0 * w + xi1) + 0] += d_tex.t10[0];
            texels[3 * (yi0 * w + xi1) + 1] += d_tex.t10[1];
            texels[3 * (yi0 * w + xi1) + 2] += d_tex.t10[2];
            texels[3 * (yi1 * w + xi0) + 0] += d_tex.t01[0];
            texels[3 * (yi1 * w + xi0) + 1] += d_tex.t01[1];
            texels[3 * (yi1 * w + xi0) + 2] += d_tex.t01[2];
            texels[3 * (yi1 * w + xi1) + 0] += d_tex.t11[0];
            texels[3 * (yi1 * w + xi1) + 1] += d_tex.t11[1];
            texels[3 * (yi1 * w + xi1) + 2] += d_tex.t11[2];
#endif
        }
    }

    const DTexture3 *d_diffuse_texs;
    DMaterial *d_materials;
    void *mutexes; // CUDA doesn't recognize std::mutex
};

struct specular_accumulator {
    DEVICE
    inline void operator()(int idx) {
        const auto &d_tex = d_specular_texs[idx];
        auto mid = d_tex.material_id;
        auto xi = d_tex.xi;
        auto yi = d_tex.yi;
        auto texels = d_materials[mid].specular_reflectance.texels;
        if (xi < 0) {
            texels[0] += d_tex.t00[0];
            texels[1] += d_tex.t00[1];
            texels[2] += d_tex.t00[2];
        } else {
            auto w = d_materials[mid].specular_reflectance.width;
            auto h = d_materials[mid].specular_reflectance.height;
            auto xi0 = xi;
            auto xi1 = modulo(xi + 1, w);
            auto yi0 = yi;
            auto yi1 = modulo(yi + 1, h);
            // Different DTexture may overlap, so we need to use atomic updates
            // The probability of collision should be small in SIMD regime though
#ifdef __CUDA_ARCH__ 
            atomic_add(texels[3 * (yi0 * w + xi0) + 0], d_tex.t00[0]);
            atomic_add(texels[3 * (yi0 * w + xi0) + 1], d_tex.t00[1]);
            atomic_add(texels[3 * (yi0 * w + xi0) + 2], d_tex.t00[2]);
            atomic_add(texels[3 * (yi0 * w + xi1) + 0], d_tex.t10[0]);
            atomic_add(texels[3 * (yi0 * w + xi1) + 1], d_tex.t10[1]);
            atomic_add(texels[3 * (yi0 * w + xi1) + 2], d_tex.t10[2]);
            atomic_add(texels[3 * (yi1 * w + xi0) + 0], d_tex.t01[0]);
            atomic_add(texels[3 * (yi1 * w + xi0) + 1], d_tex.t01[1]);
            atomic_add(texels[3 * (yi1 * w + xi0) + 2], d_tex.t01[2]);
            atomic_add(texels[3 * (yi1 * w + xi1) + 0], d_tex.t11[0]);
            atomic_add(texels[3 * (yi1 * w + xi1) + 1], d_tex.t11[1]);
            atomic_add(texels[3 * (yi1 * w + xi1) + 2], d_tex.t11[2]);
#else
            // Lock at material level. Slow but probably not bottleneck.
            std::unique_lock<std::mutex> guard(((std::mutex*)mutexes)[mid]);
            texels[3 * (yi0 * w + xi0) + 0] += d_tex.t00[0];
            texels[3 * (yi0 * w + xi0) + 1] += d_tex.t00[1];
            texels[3 * (yi0 * w + xi0) + 2] += d_tex.t00[2];
            texels[3 * (yi0 * w + xi1) + 0] += d_tex.t10[0];
            texels[3 * (yi0 * w + xi1) + 1] += d_tex.t10[1];
            texels[3 * (yi0 * w + xi1) + 2] += d_tex.t10[2];
            texels[3 * (yi1 * w + xi0) + 0] += d_tex.t01[0];
            texels[3 * (yi1 * w + xi0) + 1] += d_tex.t01[1];
            texels[3 * (yi1 * w + xi0) + 2] += d_tex.t01[2];
            texels[3 * (yi1 * w + xi1) + 0] += d_tex.t11[0];
            texels[3 * (yi1 * w + xi1) + 1] += d_tex.t11[1];
            texels[3 * (yi1 * w + xi1) + 2] += d_tex.t11[2];
#endif

        }
    }

    const DTexture3 *d_specular_texs;
    DMaterial *d_materials;
    void *mutexes; // CUDA doesn't recognize std::mutex
};

struct roughness_accumulator {
    DEVICE
    inline void operator()(int idx) {
        const auto &d_tex = d_roughness_texs[idx];
        auto mid = d_tex.material_id;
        auto xi = d_tex.xi;
        auto yi = d_tex.yi;
        auto texels = d_materials[mid].roughness.texels;
        if (xi < 0) {
            texels[0] += d_tex.t00;
        } else {
            auto w = d_materials[mid].roughness.width;
            auto h = d_materials[mid].roughness.height;
            auto xi0 = xi;
            auto xi1 = modulo(xi + 1, w);
            auto yi0 = yi;
            auto yi1 = modulo(yi + 1, h);
            // Different DTexture may overlap, so we need to use atomic updates
            // The probability of collision should be small in SIMD regime though
#ifdef __CUDA_ARCH__ 
            atomic_add(texels[yi0 * w + xi0], d_tex.t00);
            atomic_add(texels[yi0 * w + xi1], d_tex.t10);
            atomic_add(texels[yi1 * w + xi0], d_tex.t01);
            atomic_add(texels[yi1 * w + xi1], d_tex.t11);
#else
            // Lock at material level. Slow but probably not bottleneck.
            std::unique_lock<std::mutex> guard(((std::mutex*)mutexes)[mid]);
            texels[yi0 * w + xi0] += d_tex.t00;
            texels[yi0 * w + xi1] += d_tex.t10;
            texels[yi1 * w + xi0] += d_tex.t01;
            texels[yi1 * w + xi1] += d_tex.t11;

#endif
        }
    }

    const DTexture1 *d_roughness_texs;
    DMaterial *d_materials;
    void *mutexes; // CUDA doesn't recognize std::mutex
};

void accumulate_diffuse(const Scene &scene,
                        const BufferView<DTexture3> &d_diffuse_texs,
                        BufferView<DMaterial> d_materials) {
    parallel_for(
        diffuse_accumulator{d_diffuse_texs.begin(), d_materials.begin(),
            (void*)&scene.material_mutexes[0]},
        d_diffuse_texs.size(), scene.use_gpu);
}

void accumulate_specular(const Scene &scene,
                         const BufferView<DTexture3> &d_specular_texs,
                         BufferView<DMaterial> d_materials) {
    parallel_for(
        specular_accumulator{d_specular_texs.begin(), d_materials.begin(),
            (void*)&scene.material_mutexes[0]},
        d_specular_texs.size(), scene.use_gpu);
}

void accumulate_roughness(const Scene &scene,
                          const BufferView<DTexture1> &d_roughness_texs,
                          BufferView<DMaterial> d_materials) {
    parallel_for(
        roughness_accumulator{d_roughness_texs.begin(), d_materials.begin()},
        d_roughness_texs.size(), scene.use_gpu);
}

void test_d_bsdf() {
    Vector3f d{0.5, 0.4, 0.3};
    Texture3 diffuse{&d[0], -1, -1};
    Vector3f s{0.2, 0.3, 0.4};
    Texture3 specular{&s[0], -1, -1};
    float r = 0.5;
    Texture1 roughness{&r, -1, -1};
    Vector2f uv_scale = Vector2f{1, 1};
    Material m{diffuse, specular, roughness,
               &uv_scale[0],
               &uv_scale[0],
               &uv_scale[0],
               false};
    DTexture3 d_diffuse_tex;
    DTexture3 d_specular_tex;
    DTexture1 d_roughness_tex;
    SurfacePoint p{Vector3{0, 0, 0},
                   Vector3{0, 1, 0},
                   Frame(Vector3{0, 1, 0}),
                   Vector2{0.5, 0.5}};
    auto wi = normalize(Vector3{0.5, 1.0, 0.5});
    auto wo = normalize(Vector3{-0.5, 1.0, -0.5});
    auto min_roughness = Real(0);
    auto d_p = SurfacePoint::zero();
    auto d_wi = Vector3{0, 0, 0};
    auto d_wo = Vector3{0, 0, 0};

    d_bsdf(m, p, wi, wo, min_roughness, Vector3{1, 1, 1},
           d_diffuse_tex, d_specular_tex, d_roughness_tex,
           d_p, d_wi, d_wo);

    // Check diffuse derivatives
    auto finite_delta = Real(1e-6);
    for (int i = 0; i < 3; i++) {
        auto delta_m = m;
        delta_m.diffuse_reflectance.texels[i] += finite_delta;
        auto positive = bsdf(delta_m, p, wi, wo, min_roughness);
        delta_m.diffuse_reflectance.texels[i] -= 2 * finite_delta;
        auto negative = bsdf(delta_m, p, wi, wo, min_roughness);
        auto diff = sum(positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_diffuse_tex.t00[i]);
    }

    // Check specular derivatives
    for (int i = 0; i < 3; i++) {
        auto delta_m = m;
        delta_m.specular_reflectance.texels[i] += finite_delta;
        auto positive = bsdf(delta_m, p, wi, wo, min_roughness);
        delta_m.specular_reflectance.texels[i] -= 2 * finite_delta;
        auto negative = bsdf(delta_m, p, wi, wo, min_roughness);
        auto diff = sum(positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_specular_tex.t00[i]);
    }

    // Check roughness derivatives
    {
        auto delta_m = m;
        delta_m.roughness.texels[0] += finite_delta;
        auto positive = bsdf(delta_m, p, wi, wo, min_roughness);
        delta_m.roughness.texels[0] -= 2 * finite_delta;
        auto negative = bsdf(delta_m, p, wi, wo, min_roughness);
        auto diff = sum(positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_roughness_tex.t00);
    }

    // Check surface point derivatives
    equal_or_error(__FILE__, __LINE__, Vector3{0, 0, 0}, d_p.position);
    equal_or_error(__FILE__, __LINE__, Vector3{0, 0, 0}, d_p.geom_normal);
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
    Vector3f d{0.5, 0.4, 0.3};
    Texture3 diffuse{&d[0], -1, -1};
    Vector3f s{0.2, 0.3, 0.4};
    Texture3 specular{&s[0], -1, -1};
    float r = 0.5;
    Texture1 roughness{&r, -1, -1};
    Vector2f uv_scale = Vector2f{1, 1};
    Material m{diffuse, specular, roughness,
               &uv_scale[0],
               &uv_scale[0],
               &uv_scale[0],
               false};
    DTexture1 d_roughness_tex;
    SurfacePoint p{Vector3{0, 0, 0},
                   Vector3{0, 1, 0},
                   Frame(Vector3{0, 1, 0}),
                   Vector2{0.5, 0.5}};
    auto wi = normalize(Vector3{0.5, 1.0, 0.5});
    auto min_roughness = Real(0.0);
    for (int i = 0; i < 2; i++) {
        // Test for both specular and diffuse path
        auto sample = i == 0 ?
            BSDFSample{Vector2{0.5, 0.5}, 0.0} :
            BSDFSample{Vector2{0.5, 0.5}, 0.99};
        auto d_p = SurfacePoint::zero();
        auto d_wi = Vector3{0, 0, 0};

        d_bsdf_sample(m, p, wi, sample,
               min_roughness, Vector3{1, 1, 1},
               d_roughness_tex,
               d_p, d_wi);

        // Check roughness derivatives
        auto finite_delta = Real(1e-5);
        {
            auto delta_m = m;
            delta_m.roughness.texels[0] += finite_delta;
            auto positive = bsdf_sample(delta_m, p, wi, sample, min_roughness);
            delta_m.roughness.texels[0] -= 2 * finite_delta;
            auto negative = bsdf_sample(delta_m, p, wi, sample, min_roughness);
            auto diff = sum(positive - negative) / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, diff, d_roughness_tex.t00);
        }

        // Check surface point derivatives
        equal_or_error(__FILE__, __LINE__, Vector3{0, 0, 0}, d_p.position);
        equal_or_error(__FILE__, __LINE__, Vector3{0, 0, 0}, d_p.geom_normal);
        // Shading frame x
        for (int i = 0; i < 3; i++) {
            auto delta_p = p;
            delta_p.shading_frame.x[i] += finite_delta;
            auto positive = bsdf_sample(m, delta_p, wi, sample, min_roughness);
            delta_p.shading_frame.x[i] -= 2 * finite_delta;
            auto negative = bsdf_sample(m, delta_p, wi, sample, min_roughness);
            auto diff = sum(positive - negative) / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, diff, d_p.shading_frame.x[i]);
        }
        // Shading frame y
        for (int i = 0; i < 3; i++) {
            auto delta_p = p;
            delta_p.shading_frame.y[i] += finite_delta;
            auto positive = bsdf_sample(m, delta_p, wi, sample, min_roughness);
            delta_p.shading_frame.y[i] -= 2 * finite_delta;
            auto negative = bsdf_sample(m, delta_p, wi, sample, min_roughness);
            auto diff = sum(positive - negative) / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, diff, d_p.shading_frame.y[i]);
        }
        // Shading frame n
        for (int i = 0; i < 3; i++) {
            auto delta_p = p;
            delta_p.shading_frame.n[i] += finite_delta;
            auto positive = bsdf_sample(m, delta_p, wi, sample, min_roughness);
            delta_p.shading_frame.n[i] -= 2 * finite_delta;
            auto negative = bsdf_sample(m, delta_p, wi, sample, min_roughness);
            auto diff = sum(positive - negative) / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, diff, d_p.shading_frame.n[i]);
        }
        // uv
        for (int i = 0; i < 2; i++) {
            auto delta_p = p;
            delta_p.uv[i] += finite_delta;
            auto positive = bsdf_sample(m, delta_p, wi, sample, min_roughness);
            delta_p.uv[i] -= 2 * finite_delta;
            auto negative = bsdf_sample(m, delta_p, wi, sample, min_roughness);
            auto diff = sum(positive - negative) / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, diff, d_p.uv[i]);
        }

        // wi
        for (int i = 0; i < 3; i++) {
            auto delta_wi = wi;
            delta_wi[i] += finite_delta;
            auto positive = bsdf_sample(m, p, delta_wi, sample, min_roughness);
            delta_wi[i] -= 2 * finite_delta;
            auto negative = bsdf_sample(m, p, delta_wi, sample, min_roughness);
            auto diff = sum(positive - negative) / (2 * finite_delta);
            equal_or_error(__FILE__, __LINE__, diff, d_wi[i]);
        }
    }
}

void test_d_bsdf_pdf() {
    Vector3f d{0.5, 0.4, 0.3};
    Texture3 diffuse{&d[0], -1, -1};
    Vector3f s{0.2, 0.3, 0.4};
    Texture3 specular{&s[0], -1, -1};
    float r = 0.5;
    Texture1 roughness{&r, -1, -1};
    Vector2f uv_scale = Vector2f{1, 1};
    Material m{diffuse, specular, roughness,
               &uv_scale[0],
               &uv_scale[0],
               &uv_scale[0],
               false};
    DTexture1 d_roughness_tex;
    SurfacePoint p{Vector3{0, 0, 0},
                   Vector3{0, 1, 0},
                   Frame(Vector3{0, 1, 0}),
                   Vector2{0.5, 0.5}};
    auto wi = normalize(Vector3{0.5, 1.0, 0.5});
    auto wo = normalize(Vector3{-0.5, 1.0, -0.5});
    auto d_p = SurfacePoint::zero();
    auto d_wi = Vector3{0, 0, 0};
    auto d_wo = Vector3{0, 0, 0};
    auto min_roughness = Real(0.0);

    d_bsdf_pdf(m, p, wi, wo, min_roughness, 1,
               d_roughness_tex,
               d_p, d_wi, d_wo);

    // Check roughness derivatives
    auto finite_delta = Real(1e-5);
    {
        auto delta_m = m;
        delta_m.roughness.texels[0] += finite_delta;
        auto positive = bsdf_pdf(delta_m, p, wi, wo, min_roughness);
        delta_m.roughness.texels[0] -= 2 * finite_delta;
        auto negative = bsdf_pdf(delta_m, p, wi, wo, min_roughness);
        auto diff = (positive - negative) / (2 * finite_delta);
        equal_or_error(__FILE__, __LINE__, diff, d_roughness_tex.t00);
    }

    // Check surface point derivatives
    equal_or_error(__FILE__, __LINE__, Vector3{0, 0, 0}, d_p.position);
    equal_or_error(__FILE__, __LINE__, Vector3{0, 0, 0}, d_p.geom_normal);
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
