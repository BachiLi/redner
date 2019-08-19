#pragma once

#include "redner.h"
#include "vector.h"
#include "ptr.h"
#include "atomic.h"

struct Texture1 {
    Texture1() {}
    Texture1(ptr<float> texels,
             int width,
             int height,
             int num_levels,
             ptr<float> uv_scale)
        : texels(texels.get()),
          width(width), height(height),
          num_levels(num_levels),
          uv_scale(uv_scale.get()) {}

    float *texels;
    int width;
    int height;
    int num_levels;
    float *uv_scale;
};

struct Texture3 {
    Texture3() {}
    Texture3(ptr<float> texels,
             int width,
             int height,
             int num_levels,
             ptr<float> uv_scale)
        : texels(texels.get()),
          width(width), height(height),
          num_levels(num_levels),
          uv_scale(uv_scale.get()) {}

    float *texels;
    int width;
    int height;
    int num_levels;
    float *uv_scale;
};

DEVICE
inline Vector3 bilinear_interp(const Texture3 &tex,
                               int xfi, int yfi,
                               int xci, int yci,
                               Real u, Real v,
                               int level) {
    auto texels = tex.texels + level * tex.width * tex.height * 3;
    auto color_ff = Vector3f{texels[3 * (yfi * tex.width + xfi) + 0],
                             texels[3 * (yfi * tex.width + xfi) + 1],
                             texels[3 * (yfi * tex.width + xfi) + 2]};
    auto color_cf = Vector3f{texels[3 * (yfi * tex.width + xci) + 0],
                             texels[3 * (yfi * tex.width + xci) + 1],
                             texels[3 * (yfi * tex.width + xci) + 2]};
    auto color_fc = Vector3f{texels[3 * (yci * tex.width + xfi) + 0],
                             texels[3 * (yci * tex.width + xfi) + 1],
                             texels[3 * (yci * tex.width + xfi) + 2]};
    auto color_cc = Vector3f{texels[3 * (yci * tex.width + xci) + 0],
                             texels[3 * (yci * tex.width + xci) + 1],
                             texels[3 * (yci * tex.width + xci) + 2]};
    auto color = color_ff * (1.f - u) * (1.f - v) +
                 color_fc * (1.f - u) *        v  +
                 color_cf *        u  * (1.f - v) +
                 color_cc *        u  *        v;
    return color;
}

DEVICE
inline void d_bilinear_interp(const Texture3 &tex,
                              int xfi, int yfi,
                              int xci, int yci,
                              Real u, Real v,
                              int level,
                              const Vector3 &d_output,
                              Vector3f &d_color_ff, Vector3f &d_color_cf,
                              Vector3f &d_color_fc, Vector3f &d_color_cc,
                              Real &d_u, Real &d_v) {
    auto texels = tex.texels + level * tex.width * tex.height * 3;
    auto color_ff = Vector3f{texels[3 * (yfi * tex.width + xfi) + 0],
                             texels[3 * (yfi * tex.width + xfi) + 1],
                             texels[3 * (yfi * tex.width + xfi) + 2]};
    auto color_cf = Vector3f{texels[3 * (yfi * tex.width + xci) + 0],
                             texels[3 * (yfi * tex.width + xci) + 1],
                             texels[3 * (yfi * tex.width + xci) + 2]};
    auto color_fc = Vector3f{texels[3 * (yci * tex.width + xfi) + 0],
                             texels[3 * (yci * tex.width + xfi) + 1],
                             texels[3 * (yci * tex.width + xfi) + 2]};
    auto color_cc = Vector3f{texels[3 * (yci * tex.width + xci) + 0],
                             texels[3 * (yci * tex.width + xci) + 1],
                             texels[3 * (yci * tex.width + xci) + 2]};
    // color = color_ff * (1.f - u) * (1.f - v) +
    //         color_fc * (1.f - u) *        v  +
    //         color_cf *        u  * (1.f - v) +
    //         color_cc *        u  *        v;
    d_color_ff = d_output * (1.f - u) * (1.f - v);
    d_color_cf = d_output *        u  * (1.f - v);
    d_color_fc = d_output * (1.f - u) *        v ;
    d_color_cc = d_output *        u  *        v ;
    d_u += sum(d_output * (-color_ff * (1.f - v) +
                            color_cf * (1.f - v) +
                           -color_fc *        v  +
                            color_cc *        v));
    d_v += sum(d_output * (-color_ff * (1.f - u) +
                           -color_cf *        u  +
                            color_fc * (1.f - u) +
                            color_cc *        u));
}

DEVICE
inline void d_bilinear_interp(const Texture3 &tex,
                              int xfi, int yfi,
                              int xci, int yci,
                              Real u, Real v,
                              int level,
                              const Vector3 &d_output,
                              Texture3 &d_tex,
                              Real &d_u, Real &d_v) {
    auto texels = tex.texels + level * tex.width * tex.height * 3;
    auto color_ff = Vector3f{texels[3 * (yfi * tex.width + xfi) + 0],
                             texels[3 * (yfi * tex.width + xfi) + 1],
                             texels[3 * (yfi * tex.width + xfi) + 2]};
    auto color_cf = Vector3f{texels[3 * (yfi * tex.width + xci) + 0],
                             texels[3 * (yfi * tex.width + xci) + 1],
                             texels[3 * (yfi * tex.width + xci) + 2]};
    auto color_fc = Vector3f{texels[3 * (yci * tex.width + xfi) + 0],
                             texels[3 * (yci * tex.width + xfi) + 1],
                             texels[3 * (yci * tex.width + xfi) + 2]};
    auto color_cc = Vector3f{texels[3 * (yci * tex.width + xci) + 0],
                             texels[3 * (yci * tex.width + xci) + 1],
                             texels[3 * (yci * tex.width + xci) + 2]};
    // color = color_ff * (1.f - u) * (1.f - v) +
    //         color_fc * (1.f - u) *        v  +
    //         color_cf *        u  * (1.f - v) +
    //         color_cc *        u  *        v;
    auto d_texels = d_tex.texels + level * tex.width * tex.height * 3;
    // d_color_ff
    atomic_add(&d_texels[3 * (yfi * tex.width + xfi)], d_output * (1.f - u) * (1.f - v));
    // d_color_fc
    atomic_add(&d_texels[3 * (yfi * tex.width + xci)], d_output *        u  * (1.f - v));
    // d_color_cf
    atomic_add(&d_texels[3 * (yci * tex.width + xfi)], d_output * (1.f - u) *        v );
    // d_color_cc
    atomic_add(&d_texels[3 * (yci * tex.width + xci)], d_output *        u  *        v );
    d_u += sum(d_output * (-color_ff * (1.f - v) +
                            color_cf * (1.f - v) +
                           -color_fc *        v  +
                            color_cc *        v));
    d_v += sum(d_output * (-color_ff * (1.f - u) +
                           -color_cf *        u  +
                            color_fc * (1.f - u) +
                            color_cc *        u));
}

DEVICE
inline Real bilinear_interp(const Texture1 &tex,
                            int xfi, int yfi,
                            int xci, int yci,
                            Real u, Real v,
                            int level) {
    auto texels = tex.texels + level * tex.width * tex.height;
    auto value_ff = texels[yfi * tex.width + xfi];
    auto value_cf = texels[yfi * tex.width + xci];
    auto value_fc = texels[yci * tex.width + xfi];
    auto value_cc = texels[yci * tex.width + xci];
    auto value = value_ff * (1.f - u) * (1.f - v) +
                 value_fc * (1.f - u) *        v  +
                 value_cf *        u  * (1.f - v) +
                 value_cc *        u  *        v;
    return value;
}

DEVICE
inline void d_bilinear_interp(const Texture1 &tex,
                              int xfi, int yfi,
                              int xci, int yci,
                              Real u, Real v,
                              int level,
                              const Real d_output,
                              Real &d_value_ff, Real &d_value_cf,
                              Real &d_value_fc, Real &d_value_cc,
                              Real &d_u, Real &d_v) {
    auto texels = tex.texels + level * tex.width * tex.height;
    auto value_ff = texels[yfi * tex.width + xfi];
    auto value_cf = texels[yfi * tex.width + xci];
    auto value_fc = texels[yci * tex.width + xfi];
    auto value_cc = texels[yci * tex.width + xci];

    // value = value_ff * (1.f - u) * (1.f - v) +
    //         value_fc * (1.f - u) *        v  +
    //         value_cf *        u  * (1.f - v) +
    //         value_cc *        u  *        v;
    d_value_ff = d_output * (1.f - u) * (1.f - v);
    d_value_cf = d_output *        u  * (1.f - v);
    d_value_fc = d_output * (1.f - u) *        v ;
    d_value_cc = d_output *        u  *        v ;
    d_u += d_output * (-value_ff * (1.f - v) +
                        value_cf * (1.f - v) +
                       -value_fc *        v  +
                        value_cc *        v);
    d_v += d_output * (-value_ff * (1.f - u) +
                       -value_cf *        u  +
                        value_fc * (1.f - u) +
                        value_cc *        u);
}

DEVICE
inline void d_bilinear_interp(const Texture1 &tex,
                              int xfi, int yfi,
                              int xci, int yci,
                              Real u, Real v,
                              int level,
                              const Real d_output,
                              Texture1 &d_tex,
                              Real &d_u, Real &d_v) {
    auto texels = tex.texels + level * tex.width * tex.height;
    auto value_ff = texels[yfi * tex.width + xfi];
    auto value_cf = texels[yfi * tex.width + xci];
    auto value_fc = texels[yci * tex.width + xfi];
    auto value_cc = texels[yci * tex.width + xci];

    // value = value_ff * (1.f - u) * (1.f - v) +
    //         value_fc * (1.f - u) *        v  +
    //         value_cf *        u  * (1.f - v) +
    //         value_cc *        u  *        v;
    auto d_texels = d_tex.texels + level * tex.width * tex.height;
    // d_value_ff
    atomic_add(&d_texels[yfi * tex.width + xfi], d_output * (1.f - u) * (1.f - v));
    // d_value_cf
    atomic_add(&d_texels[yfi * tex.width + xci], d_output * (1.f - u) *        v );
    // d_value_fc
    atomic_add(&d_texels[yci * tex.width + xfi], d_output *        u  * (1.f - v));
    // d_value_cc
    atomic_add(&d_texels[yci * tex.width + xci], d_output *        u  *        v );
    d_u += d_output * (-value_ff * (1.f - v) +
                        value_cf * (1.f - v) +
                       -value_fc *        v  +
                        value_cc *        v);
    d_v += d_output * (-value_ff * (1.f - u) +
                       -value_cf *        u  +
                        value_fc * (1.f - u) +
                        value_cc *        u);
}

DEVICE
inline Vector3 get_texture_value_constant(const Texture3 &tex) {
    return Vector3{tex.texels[0], tex.texels[1], tex.texels[2]};
}

DEVICE
inline Real get_texture_value_constant(const Texture1 &tex) {
    return tex.texels[0];
}

template <typename TextureType>
DEVICE
inline auto get_texture_value(const TextureType &tex,
                              const Vector2 &uv_,
                              const Vector2 &du_dxy_,
                              const Vector2 &dv_dxy_) {
    if (tex.num_levels <= 0) {
        // Constant texture
        return get_texture_value_constant(tex);
    } else {
        // Trilinear interpolation
        auto uv_scale = Vector2f{tex.uv_scale[0], tex.uv_scale[1]};
        auto uv = uv_ * uv_scale;
        auto du_dxy = du_dxy_ * uv_scale[0];
        auto dv_dxy = dv_dxy_ * uv_scale[1];
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
        auto max_footprint = max(length(du_dxy) * tex.width, length(dv_dxy) * tex.height);
        auto level = log2(max(max_footprint, Real(1e-8f)));
        if (level <= 0) {
            return bilinear_interp(tex, xfi, yfi, xci, yci, u, v, 0);
        } else if (level >= tex.num_levels - 1) {
            return bilinear_interp(tex, xfi, yfi, xci, yci, u, v, tex.num_levels - 1);
        } else {
            auto li = (int)floor(level);
            auto ld = level - li;
            return (1 - ld) * bilinear_interp(tex, xfi, yfi, xci, yci, u, v, li) +
                        ld  * bilinear_interp(tex, xfi, yfi, xci, yci, u, v, li + 1);
        }
    }
}

template <typename TextureType, typename OutputType, typename DTextureType>
DEVICE
inline void d_get_texture_value(const TextureType &tex,
                                const Vector2 &uv_,
                                const Vector2 &du_dxy_,
                                const Vector2 &dv_dxy_,
                                const OutputType &d_output,
                                DTextureType &d_tex,
                                Vector2 &d_uv_,
                                Vector2 &d_du_dxy_,
                                Vector2 &d_dv_dxy_) {
    if (tex.width <= 0 && tex.height <= 0) {
        // Constant texture
        // output =  Vector3{tex.texels[0], tex.texels[1], tex.texels[2]};
        d_tex.t000 = d_output;
    } else {
        // Trilinear interpolation
        auto uv_scale = Vector2f{tex.uv_scale[0], tex.uv_scale[1]};
        auto uv = uv_ * uv_scale;
        auto du_dxy = du_dxy_ * uv_scale[0];
        auto dv_dxy = dv_dxy_ * uv_scale[1];
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
        auto u_footprint = length(du_dxy) * tex.width;
        auto v_footprint = length(dv_dxy) * tex.height;
        bool is_u_max = true;
        auto max_footprint = u_footprint;
        if (v_footprint > u_footprint) {
            is_u_max = false;
            max_footprint = v_footprint;
        }
        auto level = log2(max(max_footprint, Real(1e-8f)));
        d_tex.xi = xfi;
        d_tex.yi = yfi;
        auto d_u = Real(0);
        auto d_v = Real(0);
        auto d_max_footprint = Real(0);
        if (level <= 0) {
            d_tex.li = -1;
            d_bilinear_interp(tex, xfi, yfi, xci, yci, u, v, 0, d_output,
                d_tex.t000, d_tex.t100, d_tex.t010, d_tex.t110, d_u, d_v);
        } else if (level >= tex.num_levels - 1) {
            d_tex.li = tex.num_levels - 1;
            d_bilinear_interp(tex, xfi, yfi, xci, yci, u, v, tex.num_levels - 1, d_output,
                d_tex.t000, d_tex.t100, d_tex.t010, d_tex.t110, d_u, d_v);
        } else {
            auto li = (int)floor(level);
            d_tex.li = li;
            auto ld = level - li;
            // (1 - ld) * bilinear_interp(tex, xfi, yfi, xci, yci, u, v, li) +
            //      ld  * bilinear_interp(tex, xfi, yfi, xci, yci, u, v, li + 1);
            auto l0 = bilinear_interp(tex, xfi, yfi, xci, yci, u, v, li);
            auto l1 = bilinear_interp(tex, xfi, yfi, xci, yci, u, v, li + 1);
            auto d_l0 = (1 - ld) * d_output;
            auto d_l1 = ld * d_output;
            d_bilinear_interp(tex, xfi, yfi, xci, yci, u, v, li, d_l0,
                d_tex.t000, d_tex.t100, d_tex.t010, d_tex.t110, d_u, d_v);
            d_bilinear_interp(tex, xfi, yfi, xci, yci, u, v, li + 1, d_l1,
                d_tex.t001, d_tex.t101, d_tex.t011, d_tex.t111, d_u, d_v);
            auto d_ld = sum(d_output * (l1 - l0));
            // level = log2(max(max_footprint, Real(1e-8f)))
            if (max_footprint > Real(1e-8f)) {
                d_max_footprint += d_ld / (max_footprint * log(Real(2)));
            }
        }
        // max_footprint = max(length(du_dxy) * tex.width, length(dv_dxy) * tex.height)
        auto d_uv = Vector2{0, 0};
        auto d_du_dxy = Vector2{0, 0};
        auto d_dv_dxy = Vector2{0, 0};
        if (max_footprint > Real(1e-8f)) {
            if (is_u_max) {
                d_du_dxy += d_length(du_dxy, d_max_footprint) * tex.width;
            } else {
                d_dv_dxy += d_length(dv_dxy, d_max_footprint) * tex.height;
            }
        }

        // du = dx, dv = dy
        // x = uv[0] * tex.width - 0.5f
        // y = uv[1] * tex.height - 0.5f
        d_uv[0] += d_u * tex.width;
        d_uv[1] += d_v * tex.height;

        // uv = uv_ * uv_scale
        // du_dxy = du_dxy_ * uv_scale[0]
        // dv_dxy = dv_dxy_ * uv_scale[1]
        d_uv_ += d_uv * uv_scale;
        d_du_dxy_ += d_du_dxy * uv_scale[0];
        d_dv_dxy_ += d_dv_dxy * uv_scale[1];
    }
}

template <typename TextureType, typename OutputType>
DEVICE
inline void d_get_texture_value(const TextureType &tex,
                                const Vector2 &uv_,
                                const Vector2 &du_dxy_,
                                const Vector2 &dv_dxy_,
                                const OutputType &d_output,
                                TextureType &d_tex,
                                Vector2 &d_uv_,
                                Vector2 &d_du_dxy_,
                                Vector2 &d_dv_dxy_) {
    if (tex.width <= 0 && tex.height <= 0) {
        // Constant texture
        // output =  Vector3{tex.texels[0], tex.texels[1], tex.texels[2]};
        atomic_add(d_tex.texels, d_output);
    } else {
        // Trilinear interpolation
        auto uv_scale = Vector2f{tex.uv_scale[0], tex.uv_scale[1]};
        auto uv = uv_ * uv_scale;
        auto du_dxy = du_dxy_ * uv_scale[0];
        auto dv_dxy = dv_dxy_ * uv_scale[1];
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
        auto u_footprint = length(du_dxy) * tex.width;
        auto v_footprint = length(dv_dxy) * tex.height;
        bool is_u_max = true;
        auto max_footprint = u_footprint;
        if (v_footprint > u_footprint) {
            is_u_max = false;
            max_footprint = v_footprint;
        }
        auto level = log2(max(max_footprint, Real(1e-8f)));
        auto d_u = Real(0);
        auto d_v = Real(0);
        auto d_max_footprint = Real(0);
        if (level <= 0) {
            d_bilinear_interp(tex, xfi, yfi, xci, yci, u, v, 0, d_output, d_tex, d_u, d_v);
        } else if (level >= tex.num_levels - 1) {
            d_bilinear_interp(tex, xfi, yfi, xci, yci, u, v, tex.num_levels - 1, d_output,
                d_tex, d_u, d_v);
        } else {
            auto li = (int)floor(level);
            auto ld = level - li;
            // (1 - ld) * bilinear_interp(tex, xfi, yfi, xci, yci, u, v, li) +
            //      ld  * bilinear_interp(tex, xfi, yfi, xci, yci, u, v, li + 1);
            auto l0 = bilinear_interp(tex, xfi, yfi, xci, yci, u, v, li);
            auto l1 = bilinear_interp(tex, xfi, yfi, xci, yci, u, v, li + 1);
            auto d_l0 = (1 - ld) * d_output;
            auto d_l1 = ld * d_output;
            d_bilinear_interp(tex, xfi, yfi, xci, yci, u, v, li, d_l0,
                d_tex, d_u, d_v);
            d_bilinear_interp(tex, xfi, yfi, xci, yci, u, v, li + 1, d_l1,
                d_tex, d_u, d_v);
            auto d_ld = sum(d_output * (l1 - l0));
            // level = log2(max(max_footprint, Real(1e-8f)))
            if (max_footprint > Real(1e-8f)) {
                d_max_footprint += d_ld / (max_footprint * log(Real(2)));
            }
        }
        // max_footprint = max(length(du_dxy) * tex.width, length(dv_dxy) * tex.height)
        auto d_uv = Vector2{0, 0};
        auto d_du_dxy = Vector2{0, 0};
        auto d_dv_dxy = Vector2{0, 0};
        if (max_footprint > Real(1e-8f)) {
            if (is_u_max) {
                d_du_dxy += d_length(du_dxy, d_max_footprint) * tex.width;
            } else {
                d_dv_dxy += d_length(dv_dxy, d_max_footprint) * tex.height;
            }
        }

        // du = dx, dv = dy
        // x = uv[0] * tex.width - 0.5f
        // y = uv[1] * tex.height - 0.5f
        d_uv[0] += d_u * tex.width;
        d_uv[1] += d_v * tex.height;

        // uv = uv_ * uv_scale
        // du_dxy = du_dxy_ * uv_scale[0]
        // dv_dxy = dv_dxy_ * uv_scale[1]
        d_uv_ += d_uv * uv_scale;
        d_du_dxy_ += d_du_dxy * uv_scale[0];
        d_dv_dxy_ += d_dv_dxy * uv_scale[1];
        atomic_add(d_tex.uv_scale,
            d_uv * uv_ + Vector2{sum(d_du_dxy * du_dxy_), sum(d_dv_dxy * dv_dxy_)});
    }
}
