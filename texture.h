#pragma once

#include "redner.h"
#include "vector.h"
#include "ptr.h"

struct Texture1 {
    Texture1() {}
    Texture1(ptr<float> texels,
             int width,
             int height,
             ptr<float> uv_scale)
        : texels(texels.get()),
          width(width), height(height),
          uv_scale(Vector2f{uv_scale[0], uv_scale[1]}) {}

    float *texels;
    int width;
    int height;
    Vector2f uv_scale;
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
             int height,
             ptr<float> uv_scale)
        : texels(texels.get()),
          width(width), height(height),
          uv_scale(Vector2{uv_scale[0], uv_scale[1]}) {}

    float *texels;
    int width;
    int height;
    Vector2f uv_scale;
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
