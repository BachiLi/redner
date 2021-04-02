#include "warp_rr.h"

struct aux_sample_count_sampler {
    DEVICE void operator()(int idx) {
        const auto &pixel_id = active_pixels[idx];
        const auto num_samples = _aux_sample_sample_counts(
                                    kernel_parameters, aux_count_samples[pixel_id]);
        aux_sample_counts[pixel_id] = static_cast<uint>(num_samples);
    }

    const int* active_pixels;
    const KernelParameters kernel_parameters;
    const AuxCountSample* aux_count_samples;
    uint* aux_sample_counts;

};

void aux_sample_sample_counts( const KernelParameters& kernel_parameters,
                    const Scene& scene,
                    const BufferView<int> &active_pixels,
                    const BufferView<AuxCountSample> &aux_count_samples,
                    BufferView<uint> &aux_sample_counts) {
    parallel_for(aux_sample_count_sampler{
        active_pixels.begin(),
        kernel_parameters,
        aux_count_samples.begin(),
        aux_sample_counts.begin()
    }, active_pixels.size(), scene.use_gpu);
}