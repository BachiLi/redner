#pragma once

#include "sampler.h"

struct pcg32_state {
    uint64_t state;
    uint64_t inc;
};

struct PCGSampler : public Sampler {
    PCGSampler(bool use_gpu, uint64_t seed, int num_pixels);

    void next_camera_samples(BufferView<TCameraSample<float>> samples, bool sample_pixel_center) override;
    void next_camera_samples(BufferView<TCameraSample<double>> samples, bool sample_pixel_center) override;
    void next_light_samples(BufferView<TLightSample<float>> samples) override;
    void next_light_samples(BufferView<TLightSample<double>> samples) override;
    void next_bsdf_samples(BufferView<TBSDFSample<float>> samples) override;
    void next_bsdf_samples(BufferView<TBSDFSample<double>> samples) override;
    void next_primary_edge_samples(BufferView<TPrimaryEdgeSample<float>> samples) override;
    void next_primary_edge_samples(BufferView<TPrimaryEdgeSample<double>> samples) override;
    void next_secondary_edge_samples(BufferView<TSecondaryEdgeSample<float>> samples) override;
    void next_secondary_edge_samples(BufferView<TSecondaryEdgeSample<double>> samples) override;

    bool use_gpu;
    Buffer<pcg32_state> rng_states;
};
