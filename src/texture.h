#pragma once

#include "redner.h"
#include "vector.h"
#include "ptr.h"
#include "atomic.h"

#include <vector>
#include <cassert>

constexpr auto max_num_texels = 8;

template <int N>
struct Texture {
    Texture() {}
    Texture(const std::vector<ptr<float>> &texels,
            const std::vector<int> &width,
            const std::vector<int> &height,
            int channels, // ignored if N=-1
            ptr<float> uv_scale)
        : channels(channels),
          uv_scale(uv_scale.get()) {
        if (texels.size() > max_num_texels) {
            std::cout << "[redner] Warning: a mipmap has size more than " << max_num_texels << ". " <<
                         "Levels higher than it will be ignored." << std::endl;
        }
        assert(texels.size() == width.size() && width.size() == height.size());
        num_levels = min((int)texels.size(), max_num_texels);
        for (int i = 0; i < max_num_texels; i++) {
            this->texels[i] = nullptr;
            this->width[i] = 0;
            this->height[i] = 0;
        }
        for (int i = 0; i < num_levels; i++) {
            this->texels[i] = texels[i].get();
            this->width[i] = width[i];
            this->height[i] = height[i];
        }
    }

    float *texels[max_num_texels];
    int width[max_num_texels];
    int height[max_num_texels];
    int channels;
    int num_levels; // how many texels are not nullptr
    float *uv_scale;
};

using TextureN = Texture<-1>;
using Texture3 = Texture<3>;
using Texture1 = Texture<1>;

template <int N>
DEVICE
inline void trilinear_interp(const Texture<N> &tex,
                             const Vector2 &uv,
                             Real level,
                             Real *output) {
    // If channels == N, hopefully the constant would propagate and simplify the code.
    auto channels = N == -1 ? tex.channels : N;
    if (level <= 0 || level >= tex.num_levels - 1) {
        auto li = level <= 0 ? 0 : tex.num_levels - 1;
        auto texels = tex.texels[li];
        auto x = uv[0] * tex.width[li] - 0.5f;
        auto y = uv[1] * tex.height[li] - 0.5f;
        auto xf = (int)floor(x);
        auto yf = (int)floor(y);
        auto xc = xf + 1;
        auto yc = yf + 1;
        auto u = x - xf;
        auto v = y - yf;
        auto xfi = modulo(xf, tex.width[li]);
        auto yfi = modulo(yf, tex.height[li]);
        auto xci = modulo(xc, tex.width[li]);
        auto yci = modulo(yc, tex.height[li]);

        for (int i = 0; i < channels; i++) {
            auto value_ff = texels[channels * (yfi * tex.width[li] + xfi) + i];
            auto value_cf = texels[channels * (yfi * tex.width[li] + xci) + i];
            auto value_fc = texels[channels * (yci * tex.width[li] + xfi) + i];
            auto value_cc = texels[channels * (yci * tex.width[li] + xci) + i];
            output[i] = value_ff * (1.f - u) * (1.f - v) +
                        value_fc * (1.f - u) *        v  +
                        value_cf *        u  * (1.f - v) +
                        value_cc *        u  *        v;
        }        
    } else {
        auto li = (int)floor(level);
        assert(li + 1 < tex.num_levels);
        auto ld = level - li;
        auto texels0 = tex.texels[li];
        auto texels1 = tex.texels[li + 1];

        auto x0 = uv[0] * tex.width[li] - 0.5f;
        auto y0 = uv[1] * tex.height[li] - 0.5f;
        auto xf0 = (int)floor(x0);
        auto yf0 = (int)floor(y0);
        auto xc0 = xf0 + 1;
        auto yc0 = yf0 + 1;
        auto u0 = x0 - xf0;
        auto v0 = y0 - yf0;
        auto xfi0 = modulo(xf0, tex.width[li]);
        auto yfi0 = modulo(yf0, tex.height[li]);
        auto xci0 = modulo(xc0, tex.width[li]);
        auto yci0 = modulo(yc0, tex.height[li]);

        auto x1 = uv[0] * tex.width[li + 1] - 0.5f;
        auto y1 = uv[1] * tex.height[li + 1] - 0.5f;
        auto xf1 = (int)floor(x1);
        auto yf1 = (int)floor(y1);
        auto xc1 = xf1 + 1;
        auto yc1 = yf1 + 1;
        auto u1 = x1 - xf1;
        auto v1 = y1 - yf1;
        auto xfi1 = modulo(xf1, tex.width[li + 1]);
        auto yfi1 = modulo(yf1, tex.height[li + 1]);
        auto xci1 = modulo(xc1, tex.width[li + 1]);
        auto yci1 = modulo(yc1, tex.height[li + 1]);

        for (int i = 0; i < channels; i++) {
            auto value_ff0 = texels0[channels * (yfi0 * tex.width[li] + xfi0) + i];
            auto value_cf0 = texels0[channels * (yfi0 * tex.width[li] + xci0) + i];
            auto value_fc0 = texels0[channels * (yci0 * tex.width[li] + xfi0) + i];
            auto value_cc0 = texels0[channels * (yci0 * tex.width[li] + xci0) + i];
            auto value_ff1 = texels1[channels * (yfi1 * tex.width[li + 1] + xfi1) + i];
            auto value_cf1 = texels1[channels * (yfi1 * tex.width[li + 1] + xci1) + i];
            auto value_fc1 = texels1[channels * (yci1 * tex.width[li + 1] + xfi1) + i];
            auto value_cc1 = texels1[channels * (yci1 * tex.width[li + 1] + xci1) + i];
            auto val0 = value_ff0 * (1.f - u0) * (1.f - v0) +
                        value_fc0 * (1.f - u0) *        v0  +
                        value_cf0 *        u0  * (1.f - v0) +
                        value_cc0 *        u0  *        v0;
            auto val1 = value_ff1 * (1.f - u1) * (1.f - v1) +
                        value_fc1 * (1.f - u1) *        v1  +
                        value_cf1 *        u1  * (1.f - v1) +
                        value_cc1 *        u1  *        v1;
            output[i] = val0 * (1 - ld) + val1 * ld;
        }
    }
}

template <int N>
DEVICE
inline void d_trilinear_interp(const Texture<N> &tex,
                               const Vector2 &uv,
                               Real level,
                               const Real *d_output,
                               Texture<N> &d_tex,
                               Vector2 &d_uv,
                               Real &d_level) {
    // If channels == N, hopefully the constant would propagate and simplify the code.
    auto channels = N == -1 ? tex.channels : N;
    if (level <= 0 || level >= tex.num_levels - 1) {
        auto li = level <= 0 ? 0 : tex.num_levels - 1;
        auto texels = tex.texels[li];
        auto d_texels = d_tex.texels[li];

        auto x = uv[0] * tex.width[li] - 0.5f;
        auto y = uv[1] * tex.height[li] - 0.5f;
        auto xf = (int)floor(x);
        auto yf = (int)floor(y);
        auto xc = xf + 1;
        auto yc = yf + 1;
        auto u = x - xf;
        auto v = y - yf;
        auto xfi = modulo(xf, tex.width[li]);
        auto yfi = modulo(yf, tex.height[li]);
        auto xci = modulo(xc, tex.width[li]);
        auto yci = modulo(yc, tex.height[li]);

        auto d_u = Real(0);
        auto d_v = Real(0);
        for (int i = 0; i < channels; i++) {
            auto value_ff = texels[channels * (yfi * tex.width[li] + xfi) + i];
            auto value_cf = texels[channels * (yfi * tex.width[li] + xci) + i];
            auto value_fc = texels[channels * (yci * tex.width[li] + xfi) + i];
            auto value_cc = texels[channels * (yci * tex.width[li] + xci) + i];
            // output[i] = value_ff * (1.f - u) * (1.f - v) +
            //             value_fc * (1.f - u) *        v  +
            //             value_cf *        u  * (1.f - v) +
            //             value_cc *        u  *        v;
            // d_value_ff
            atomic_add(&d_texels[channels * (yfi * tex.width[li] + xfi) + i],
                       d_output[i] * (1.f - u) * (1.f - v));
            // d_value_fc
            atomic_add(&d_texels[channels * (yfi * tex.width[li] + xci) + i],
                       d_output[i] *        u  * (1.f - v));
            // d_value_cf
            atomic_add(&d_texels[channels * (yci * tex.width[li] + xfi) + i],
                       d_output[i] * (1.f - u) *        v );
            // d_value_cc
            atomic_add(&d_texels[channels * (yci * tex.width[li] + xci) + i],
                       d_output[i] *        u  *        v );
            d_u += sum(d_output[i] * (-value_ff * (1.f - v) +
                                       value_cf * (1.f - v) +
                                      -value_fc *        v  +
                                       value_cc *        v));
            d_v += sum(d_output[i] * (-value_ff * (1.f - u) +
                                      -value_cf *        u  +
                                       value_fc * (1.f - u) +
                                       value_cc *        u));
        }
        // du = dx, dv = dy
        // x = uv[0] * tex.width[li] - 0.5f
        // y = uv[1] * tex.height[li] - 0.5f
        d_uv[0] += d_u * tex.width[li];
        d_uv[1] += d_v * tex.height[li];
    } else {
        auto li = (int)floor(level);
        assert(li + 1 < tex.num_levels);
        assert(d_tex.num_levels == tex.num_levels);
        auto ld = level - li;
        auto texels0 = tex.texels[li];
        auto texels1 = tex.texels[li + 1];
        auto d_texels0 = d_tex.texels[li];
        auto d_texels1 = d_tex.texels[li + 1];

        auto x0 = uv[0] * tex.width[li] - 0.5f;
        auto y0 = uv[1] * tex.height[li] - 0.5f;
        auto xf0 = (int)floor(x0);
        auto yf0 = (int)floor(y0);
        auto xc0 = xf0 + 1;
        auto yc0 = yf0 + 1;
        auto u0 = x0 - xf0;
        auto v0 = y0 - yf0;
        auto xfi0 = modulo(xf0, tex.width[li]);
        auto yfi0 = modulo(yf0, tex.height[li]);
        auto xci0 = modulo(xc0, tex.width[li]);
        auto yci0 = modulo(yc0, tex.height[li]);

        auto x1 = uv[0] * tex.width[li + 1] - 0.5f;
        auto y1 = uv[1] * tex.height[li + 1] - 0.5f;
        auto xf1 = (int)floor(x1);
        auto yf1 = (int)floor(y1);
        auto xc1 = xf1 + 1;
        auto yc1 = yf1 + 1;
        auto u1 = x1 - xf1;
        auto v1 = y1 - yf1;
        auto xfi1 = modulo(xf1, tex.width[li + 1]);
        auto yfi1 = modulo(yf1, tex.height[li + 1]);
        auto xci1 = modulo(xc1, tex.width[li + 1]);
        auto yci1 = modulo(yc1, tex.height[li + 1]);

        auto d_u0 = Real(0);
        auto d_v0 = Real(0);
        auto d_u1 = Real(0);
        auto d_v1 = Real(0);
        for (int i = 0; i < channels; i++) {
            auto value_ff0 = texels0[channels * (yfi0 * tex.width[li] + xfi0) + i];
            auto value_cf0 = texels0[channels * (yfi0 * tex.width[li] + xci0) + i];
            auto value_fc0 = texels0[channels * (yci0 * tex.width[li] + xfi0) + i];
            auto value_cc0 = texels0[channels * (yci0 * tex.width[li] + xci0) + i];
            auto value_ff1 = texels1[channels * (yfi1 * tex.width[li + 1] + xfi1) + i];
            auto value_cf1 = texels1[channels * (yfi1 * tex.width[li + 1] + xci1) + i];
            auto value_fc1 = texels1[channels * (yci1 * tex.width[li + 1] + xfi1) + i];
            auto value_cc1 = texels1[channels * (yci1 * tex.width[li + 1] + xci1) + i];
            auto val0 = value_ff0 * (1.f - u0) * (1.f - v0) +
                        value_fc0 * (1.f - u0) *        v0  +
                        value_cf0 *        u0  * (1.f - v0) +
                        value_cc0 *        u0  *        v0;
            auto val1 = value_ff1 * (1.f - u1) * (1.f - v1) +
                        value_fc1 * (1.f - u1) *        v1  +
                        value_cf1 *        u1  * (1.f - v1) +
                        value_cc1 *        u1  *        v1;
            // output[i] = val0 * (1 - ld) + val1 * ld;
            auto d_val0 = d_output[i] * (1 - ld);
            auto d_val1 = d_output[i] *      ld;
            d_level += d_output[i] * (val1 - val0);
            // d_value_ff0
            atomic_add(&d_texels0[channels * (yfi0 * tex.width[li] + xfi0) + i],
                       d_val0 * (1.f - u0) * (1.f - v0));
            // d_value_fc0
            atomic_add(&d_texels0[channels * (yfi0 * tex.width[li] + xci0) + i],
                       d_val0 *        u0  * (1.f - v0));
            // d_value_cf0
            atomic_add(&d_texels0[channels * (yci0 * tex.width[li] + xfi0) + i],
                       d_val0 * (1.f - u0) *        v0 );
            // d_value_cc0
            atomic_add(&d_texels0[channels * (yci0 * tex.width[li] + xci0) + i],
                       d_val0 *        u0  *        v0 );
            // d_value_ff1
            atomic_add(&d_texels1[channels * (yfi1 * tex.width[li + 1] + xfi1) + i],
                       d_val1 * (1.f - u1) * (1.f - v1));
            // d_value_fc1
            atomic_add(&d_texels1[channels * (yfi1 * tex.width[li + 1] + xci1) + i],
                       d_val1 *        u1  * (1.f - v1));
            // d_value_cf1
            atomic_add(&d_texels1[channels * (yci1 * tex.width[li + 1] + xfi1) + i],
                       d_val1 * (1.f - u1) *        v1 );
            // d_value_cc1
            atomic_add(&d_texels1[channels * (yci1 * tex.width[li + 1] + xci1) + i],
                       d_val1 *        u1  *        v1 );
            d_u0 += d_val0 * (-value_ff0 * (1.f - v0) +
                               value_cf0 * (1.f - v0) +
                              -value_fc0 *        v0  +
                               value_cc0 *        v0);
            d_u1 += d_val1 * (-value_ff1 * (1.f - v1) +
                               value_cf1 * (1.f - v1) +
                              -value_fc1 *        v1  +
                               value_cc1 *        v1);
            d_v0 += d_val0 * (-value_ff0 * (1.f - u0) +
                              -value_cf0 *        u0  +
                               value_fc0 * (1.f - u0) +
                               value_cc0 *        u0);
            d_v1 += d_val1 * (-value_ff1 * (1.f - u1) +
                              -value_cf1 *        u1  +
                               value_fc1 * (1.f - u1) +
                               value_cc1 *        u1);
        }

        // du0 = dx0, dv0 = dy0
        // x1 = uv[0] * tex.width[li + 1] - 0.5f
        // y1 = uv[1] * tex.height[li + 1] - 0.5f
        d_uv[0] += d_u1 * tex.width[li + 1];
        d_uv[1] += d_v1 * tex.height[li + 1];

        // du0 = dx0, dv1 = dy1
        // x0 = uv[0] * tex.width[li] - 0.5f
        // y0 = uv[1] * tex.height[li] - 0.5f
        d_uv[0] += d_u0 * tex.width[li];
        d_uv[1] += d_v0 * tex.height[li];
    }
}

template <int N>
DEVICE
inline void get_texture_value_constant(const Texture<N> &tex,
                                       Real *output) {
    auto channels = N == -1 ? tex.channels : N;
    for (int i = 0; i < channels; i++) {
        output[i] = tex.texels[0][i];
    }
}

template <typename TextureType>
DEVICE
inline void get_texture_value(const TextureType &tex,
                              const Vector2 &uv_,
                              const Vector2 &du_dxy_,
                              const Vector2 &dv_dxy_,
                              Real *output) {
    if (tex.width[0] <= 0 && tex.height[0] <= 0) {
        // Constant texture
        get_texture_value_constant(tex, output);
    } else {
        // Trilinear interpolation
        auto uv_scale = Vector2f{tex.uv_scale[0], tex.uv_scale[1]};
        auto uv = uv_ * uv_scale;
        auto du_dxy = du_dxy_ * uv_scale[0];
        auto dv_dxy = dv_dxy_ * uv_scale[1];
        auto max_footprint = max(length(du_dxy) * tex.width[0], length(dv_dxy) * tex.height[0]);
        auto level = log2(max(max_footprint, Real(1e-8f)));
        trilinear_interp(tex, uv, level, output);
    }
}

template <int N>
DEVICE
inline void d_get_texture_value(const Texture<N> &tex,
                                const Vector2 &uv_,
                                const Vector2 &du_dxy_,
                                const Vector2 &dv_dxy_,
                                const Real *d_output,
                                Texture<N> &d_tex,
                                Vector2 &d_uv_,
                                Vector2 &d_du_dxy_,
                                Vector2 &d_dv_dxy_) {
    if (tex.width[0] <= 0 && tex.height[0] <= 0) {
        // Constant texture
        // output[i] = tex.texels[i]
        auto channels = N == -1 ? tex.channels : N;
        for (int i = 0; i < channels; i++) {
            atomic_add(d_tex.texels[0][i], d_output[i]);
        }
    } else {
        // Trilinear interpolation
        auto uv_scale = Vector2f{tex.uv_scale[0], tex.uv_scale[1]};
        auto uv = uv_ * uv_scale;
        auto du_dxy = du_dxy_ * uv_scale[0];
        auto dv_dxy = dv_dxy_ * uv_scale[1];
        auto u_footprint = length(du_dxy) * tex.width[0];
        auto v_footprint = length(dv_dxy) * tex.height[0];
        bool is_u_max = true;
        auto max_footprint = u_footprint;
        if (v_footprint > u_footprint) {
            is_u_max = false;
            max_footprint = v_footprint;
        }
        auto level = log2(max(max_footprint, Real(1e-8f)));
        auto d_uv = Vector2{0, 0};
        auto d_level = Real(0);
        d_trilinear_interp(tex, uv, level,
                           d_output, d_tex, d_uv, d_level);
        auto d_max_footprint = Real(0);
        // level = log2(max(max_footprint, Real(1e-8f)))
        if (max_footprint > Real(1e-8f)) {
            d_max_footprint += d_level / (max_footprint * log(Real(2)));
        }
        // max_footprint = max(length(du_dxy) * tex.width[0], length(dv_dxy) * tex.height[0])
        auto d_du_dxy = Vector2{0, 0};
        auto d_dv_dxy = Vector2{0, 0};
        if (max_footprint > Real(1e-8f)) {
            if (is_u_max) {
                d_du_dxy += d_length(du_dxy, d_max_footprint) * tex.width[0];
            } else {
                d_dv_dxy += d_length(dv_dxy, d_max_footprint) * tex.height[0];
            }
        }

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
