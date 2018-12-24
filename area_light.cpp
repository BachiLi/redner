#include "area_light.h"
#include "parallel.h"

struct area_light_accumulator {
    DEVICE
    inline void operator()(int idx) {
        auto lid = d_light_insts[idx].light_id;
        d_lights[lid].intensity[0] += d_light_insts[idx].intensity[0];
        d_lights[lid].intensity[1] += d_light_insts[idx].intensity[1];
        d_lights[lid].intensity[2] += d_light_insts[idx].intensity[2];
    }

    const DAreaLightInst *d_light_insts;
    DAreaLight *d_lights;
};

void accumulate_area_light(const BufferView<DAreaLightInst> &d_light_insts,
                           BufferView<DAreaLight> d_lights,
                           bool use_gpu) {
    parallel_for(area_light_accumulator{
        d_light_insts.begin(), d_lights.begin()
    }, d_light_insts.size(), use_gpu);
}
