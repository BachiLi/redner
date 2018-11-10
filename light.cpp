#include "light.h"
#include "parallel.h"

struct light_accumulator {
    DEVICE
    inline void operator()(int idx) {
        auto lid = d_light_insts[idx].light_id;
        d_lights[lid].intensity[0] += d_light_insts[idx].intensity[0];
        d_lights[lid].intensity[1] += d_light_insts[idx].intensity[1];
        d_lights[lid].intensity[2] += d_light_insts[idx].intensity[2];
    }

    const DLightInst *d_light_insts;
    DLight *d_lights;
};

void accumulate_light(const BufferView<DLightInst> &d_light_insts,
                      BufferView<DLight> d_lights,
                      bool use_gpu) {
    parallel_for(light_accumulator{
        d_light_insts.begin(), d_lights.begin()
    }, d_light_insts.size(), use_gpu);
}

