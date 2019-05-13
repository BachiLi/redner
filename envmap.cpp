#include "envmap.h"
#include "scene.h"
#include "parallel.h"

struct envmap_accumulator {
    DEVICE
    inline void operator()(int idx) {
        const auto &d_tex = d_envmap_vals[idx];
        auto xi = d_tex.xi;
        auto yi = d_tex.yi;
        auto texels = d_envmap->values.texels;
        if (xi < 0) {
            texels[0] += d_tex.t000[0];
            texels[1] += d_tex.t000[1];
            texels[2] += d_tex.t000[2];
        } else {
            auto w = d_envmap->values.width;
            auto h = d_envmap->values.height;
            auto num_levels = d_envmap->values.num_levels;
            auto xi0 = xi;
            auto xi1 = modulo(xi + 1, w);
            auto yi0 = yi;
            auto yi1 = modulo(yi + 1, h);
            auto level = d_tex.li;
            if (d_tex.li == -1) {
                level = 0;
            }
            auto lower_texels = texels + level * w * h * 3;
            // Different DTexture may overlap, so we need to use atomic updates
            // The probability of collision should be small in SIMD regime though
            atomic_add(lower_texels[3 * (yi0 * w + xi0) + 0], d_tex.t000[0]);
            atomic_add(lower_texels[3 * (yi0 * w + xi0) + 1], d_tex.t000[1]);
            atomic_add(lower_texels[3 * (yi0 * w + xi0) + 2], d_tex.t000[2]);
            atomic_add(lower_texels[3 * (yi0 * w + xi1) + 0], d_tex.t100[0]);
            atomic_add(lower_texels[3 * (yi0 * w + xi1) + 1], d_tex.t100[1]);
            atomic_add(lower_texels[3 * (yi0 * w + xi1) + 2], d_tex.t100[2]);
            atomic_add(lower_texels[3 * (yi1 * w + xi0) + 0], d_tex.t010[0]);
            atomic_add(lower_texels[3 * (yi1 * w + xi0) + 1], d_tex.t010[1]);
            atomic_add(lower_texels[3 * (yi1 * w + xi0) + 2], d_tex.t010[2]);
            atomic_add(lower_texels[3 * (yi1 * w + xi1) + 0], d_tex.t110[0]);
            atomic_add(lower_texels[3 * (yi1 * w + xi1) + 1], d_tex.t110[1]);
            atomic_add(lower_texels[3 * (yi1 * w + xi1) + 2], d_tex.t110[2]);
            if (d_tex.li >= 0 && d_tex.li < num_levels - 1) {
                auto higher_texels = texels + (level + 1) * w * h * 3;
                atomic_add(higher_texels[3 * (yi0 * w + xi0) + 0], d_tex.t001[0]);
                atomic_add(higher_texels[3 * (yi0 * w + xi0) + 1], d_tex.t001[1]);
                atomic_add(higher_texels[3 * (yi0 * w + xi0) + 2], d_tex.t001[2]);
                atomic_add(higher_texels[3 * (yi0 * w + xi1) + 0], d_tex.t101[0]);
                atomic_add(higher_texels[3 * (yi0 * w + xi1) + 1], d_tex.t101[1]);
                atomic_add(higher_texels[3 * (yi0 * w + xi1) + 2], d_tex.t101[2]);
                atomic_add(higher_texels[3 * (yi1 * w + xi0) + 0], d_tex.t011[0]);
                atomic_add(higher_texels[3 * (yi1 * w + xi0) + 1], d_tex.t011[1]);
                atomic_add(higher_texels[3 * (yi1 * w + xi0) + 2], d_tex.t011[2]);
                atomic_add(higher_texels[3 * (yi1 * w + xi1) + 0], d_tex.t111[0]);
                atomic_add(higher_texels[3 * (yi1 * w + xi1) + 1], d_tex.t111[1]);
                atomic_add(higher_texels[3 * (yi1 * w + xi1) + 2], d_tex.t111[2]);
            }
        }
    }

    const DTexture3 *d_envmap_vals;
    DEnvironmentMap *d_envmap;
};

void accumulate_envmap(const Scene &scene,
                       const BufferView<DTexture3> &d_envmap_vals,
                       const Matrix4x4 &d_world_to_env,
                       DEnvironmentMap &d_envmap) {
    parallel_for(envmap_accumulator{d_envmap_vals.begin(), &d_envmap},
        d_envmap_vals.size(), scene.use_gpu);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            d_envmap.world_to_env[4 * i + j] += d_world_to_env(i, j);
        }
    }
}
