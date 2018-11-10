#pragma once

#include "buffer.h"
#include "vector.h"
#include "camera.h"
#include "light.h"
#include "material.h"
#include "edge.h"

struct pcg32_state {
    uint64_t state;
    uint64_t inc;
};

struct Sampler {
    Sampler(bool use_gpu, uint64_t seed, int num_pixels);

    void next_camera_samples(
        BufferView<TCameraSample<float>> samples);
    void next_camera_samples(
        BufferView<TCameraSample<double>> samples);
    void next_light_samples(
        BufferView<TLightSample<float>> samples);
    void next_light_samples(
        BufferView<TLightSample<double>> samples);
    void next_bsdf_samples(
        BufferView<TBSDFSample<float>> samples);
    void next_bsdf_samples(
        BufferView<TBSDFSample<double>> samples);
    void next_primary_edge_samples(
        BufferView<TPrimaryEdgeSample<float>> samples);
    void next_primary_edge_samples(
        BufferView<TPrimaryEdgeSample<double>> samples);
    void next_secondary_edge_samples(
        BufferView<TSecondaryEdgeSample<float>> samples);
    void next_secondary_edge_samples(
        BufferView<TSecondaryEdgeSample<double>> samples);

    bool use_gpu;
    Buffer<pcg32_state> rng_states;
};
