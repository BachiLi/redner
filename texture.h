#pragma once

#include "redner.h"
#include "vector.h"
#include "ptr.h"

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
          uv_scale(Vector2f{uv_scale[0], uv_scale[1]}) {}

    float *texels;
    int width;
    int height;
    int num_levels;
    Vector2f uv_scale;
};

struct DTexture1 {
    DTexture1(int mi = -1, int xi = -1, int yi = -1, int li = -1,
        Real t000 = 0, Real t010 = 0, Real t100 = 0, Real t110 = 0,
        Real t001 = 0, Real t011 = 0, Real t101 = 0, Real t111 = 0) :
            material_id(mi),
            xi(xi),
            yi(yi),
            li(li),
            t000(t000), t010(t010), t100(t100), t110(t110),
            t001(t001), t011(t011), t101(t101), t111(t111) {}
    int material_id, xi, yi, li;
    Real t000, t010, t100, t110;
    Real t001, t011, t101, t111;

    DEVICE inline bool operator<(const DTexture1 &other) const {
        if (material_id != other.material_id) {
            return material_id < other.material_id;
        }
        if (li != other.li) {
            return li < other.li;
        }
        if (yi != other.yi) {
            return yi < other.yi;
        }
        return xi < other.xi;
    }

    DEVICE inline bool operator==(const DTexture1 &other) const {
        return material_id == other.material_id &&
               xi == other.xi && yi == other.yi && li == other.li;
    }

    DEVICE inline DTexture1 operator+(const DTexture1 &other) const {
        return DTexture1{material_id, xi, yi, li,
                         t000 + other.t000,
                         t010 + other.t010,
                         t100 + other.t100,
                         t110 + other.t110,
                         t001 + other.t001,
                         t011 + other.t011,
                         t101 + other.t101,
                         t111 + other.t111};
    }
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
          uv_scale(Vector2{uv_scale[0], uv_scale[1]}) {}

    float *texels;
    int width;
    int height;
    int num_levels;
    Vector2f uv_scale;
};

struct DTexture3 {
    DTexture3(int mi = -1, int xi = -1, int yi = -1, int li = -1,
        const Vector3f &t000 = Vector3f{0, 0, 0},
        const Vector3f &t010 = Vector3f{0, 0, 0},
        const Vector3f &t100 = Vector3f{0, 0, 0},
        const Vector3f &t110 = Vector3f{0, 0, 0},
        const Vector3f &t001 = Vector3f{0, 0, 0},
        const Vector3f &t011 = Vector3f{0, 0, 0},
        const Vector3f &t101 = Vector3f{0, 0, 0},
        const Vector3f &t111 = Vector3f{0, 0, 0}) :
        material_id(mi),
        xi(xi),
        yi(yi),
        li(li),
        t000(t000), t010(t010), t100(t100), t110(t110),
        t001(t001), t011(t011), t101(t101), t111(t111) {}
    int material_id, xi, yi, li;
    /**
     * HACK: We use Vector3f instead of Vector3 as a workaround for a bug in thrust
     * It seems that thrust has some memory bugs when a struct is larger than 128 bytes
     * see https://devtalk.nvidia.com/default/topic/1036643/cuda-programming-and-performance/thrust-remove_if-memory-corruption/
     * (and after 5 months the bugs is still not fixed ; ( )
     * maybe we should drop thrust dependencies at some point.
     */
    Vector3f t000;
    Vector3f t010;
    Vector3f t100;
    Vector3f t110;
    Vector3f t001;
    Vector3f t011;
    Vector3f t101;
    Vector3f t111;

    DEVICE inline bool operator<(const DTexture3 &other) const {
        if (material_id != other.material_id) {
            return material_id < other.material_id;
        }
        if (li != other.li) {
            return li < other.li;
        }
        if (yi != other.yi) {
            return yi < other.yi;
        }
        return xi < other.xi;
    }

    DEVICE inline bool operator==(const DTexture3 &other) const {
        return material_id == other.material_id &&
               xi == other.xi && yi == other.yi && li == other.li;
    }

    DEVICE inline DTexture3 operator+(const DTexture3 &other) const {
        return DTexture3{material_id, xi, yi, li,
                         t000 + other.t000,
                         t010 + other.t010,
                         t100 + other.t100,
                         t110 + other.t110,
                         t001 + other.t001,
                         t011 + other.t011,
                         t101 + other.t101,
                         t111 + other.t111};
    }
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
    auto texels = tex.texels + level * tex.width * tex.height * 3;
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
        auto uv = uv_ * tex.uv_scale;
        auto du_dxy = du_dxy_ * tex.uv_scale[0];
        auto dv_dxy = dv_dxy_ * tex.uv_scale[1];
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
        auto uv = uv_ * tex.uv_scale;
        auto du_dxy = du_dxy_ * tex.uv_scale[0];
        auto dv_dxy = dv_dxy_ * tex.uv_scale[1];
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

        // uv = uv_ * tex.uv_scale
        // du_dxy = du_dxy_ * tex.uv_scale[0]
        // dv_dxy = dv_dxy_ * tex.uv_scale[1]
        d_uv_ += d_uv * tex.uv_scale;
        d_du_dxy_ += d_du_dxy * tex.uv_scale[0];
        d_dv_dxy_ += d_dv_dxy * tex.uv_scale[1];
    }
}
