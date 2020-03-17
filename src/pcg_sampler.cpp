#include "pcg_sampler.h"
#include "parallel.h"
#include "thrust_utils.h"

#include <thrust/fill.h>

// http://www.pcg-random.org/download.html
DEVICE inline uint32_t next_pcg32(pcg32_state *rng) {
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = uint32_t(oldstate >> 59u);
    return uint32_t((xorshifted >> rot) | (xorshifted << ((-rot) & 31)));
}

// https://github.com/wjakob/pcg32/blob/master/pcg32.h
DEVICE inline float next_pcg32_float(pcg32_state *rng) {
    union {
        uint32_t u;
        float f;
    } x;
    x.u = (next_pcg32(rng) >> 9) | 0x3f800000u;
    return x.f - 1.0f;
}

// https://github.com/wjakob/pcg32/blob/master/pcg32.h
DEVICE inline double next_pcg32_double(pcg32_state *rng) {
    union {
        uint64_t u;
        double d;
    } x;
    x.u = ((uint64_t) next_pcg32(rng) << 20) | 0x3ff0000000000000ULL;
    return x.d - 1.0;
}

// Initialize each pixel with a PCG rng with a different stream
struct pcg_initializer {
    DEVICE void operator()(int idx) {
        rng_states[idx].state = 0U;
        rng_states[idx].inc = (((uint64_t)idx + 1) << 1u) | 1u;
        next_pcg32(&rng_states[idx]);
        rng_states[idx].state += (0x853c49e6748fea9bULL + seed);
        next_pcg32(&rng_states[idx]);
    }

    uint64_t seed;
    pcg32_state *rng_states;
};

template <int spp>
struct pcg_sampler_float {
    DEVICE void operator()(int idx) {
        for (int i = 0; i < spp; i++) {
            samples[spp * idx + i] = next_pcg32_float(&rng_states[idx]);
        }
    }

    pcg32_state *rng_states;
    float *samples;
};

template <int spp>
struct pcg_sampler_double {
    DEVICE void operator()(int idx) {
        for (int i = 0; i < spp; i++) {
            samples[spp * idx + i] = next_pcg32_double(&rng_states[idx]);
        }
    }

    pcg32_state *rng_states;
    double *samples;
};

PCGSampler::PCGSampler(bool use_gpu,
                       uint64_t seed,
                       int num_pixels) : use_gpu(use_gpu) {
    rng_states = Buffer<pcg32_state>(use_gpu, num_pixels);
    parallel_for(pcg_initializer{seed, rng_states.begin()},
        rng_states.size(), use_gpu);
}

void PCGSampler::next_camera_samples(BufferView<TCameraSample<float>> samples, bool sample_pixel_center) {
    if (sample_pixel_center) {
        DISPATCH(use_gpu, thrust::fill,
            (float*)samples.begin(), (float*)samples.end(), 0.5f);
    } else {
        parallel_for(pcg_sampler_float<2>{rng_states.begin(),
            (float*)samples.begin()}, samples.size(), use_gpu);
    }
}

void PCGSampler::next_camera_samples(BufferView<TCameraSample<double>> samples, bool sample_pixel_center) {
    if (sample_pixel_center) {
        DISPATCH(use_gpu, thrust::fill,
            (double*)samples.begin(), (double*)samples.end(), 0.5);
    } else {
        parallel_for(pcg_sampler_double<2>{rng_states.begin(),
            (double*)samples.begin()}, samples.size(), use_gpu);
    }
}

void PCGSampler::next_light_samples(BufferView<TLightSample<float>> samples) {
    parallel_for(pcg_sampler_float<4>{rng_states.begin(),
        (float*)samples.begin()}, samples.size(), use_gpu);
}

void PCGSampler::next_light_samples(BufferView<TLightSample<double>> samples) {
    parallel_for(pcg_sampler_double<4>{rng_states.begin(),
        (double*)samples.begin()}, samples.size(), use_gpu);
}

void PCGSampler::next_bsdf_samples(BufferView<TBSDFSample<float>> samples) {
    parallel_for(pcg_sampler_float<3>{rng_states.begin(),
        (float*)samples.begin()}, samples.size(), use_gpu);
}

void PCGSampler::next_bsdf_samples(BufferView<TBSDFSample<double>> samples) {
    parallel_for(pcg_sampler_double<3>{rng_states.begin(),
        (double*)samples.begin()}, samples.size(), use_gpu);
}

void PCGSampler::next_primary_edge_samples(
        BufferView<TPrimaryEdgeSample<float>> samples) {
    parallel_for(pcg_sampler_float<2>{rng_states.begin(),
        (float*)samples.begin()}, samples.size(), use_gpu);
}

void PCGSampler::next_primary_edge_samples(
        BufferView<TPrimaryEdgeSample<double>> samples) {
    parallel_for(pcg_sampler_double<2>{rng_states.begin(),
        (double*)samples.begin()}, samples.size(), use_gpu);
}

void PCGSampler::next_secondary_edge_samples(
        BufferView<TSecondaryEdgeSample<float>> samples) {
    parallel_for(pcg_sampler_float<4>{rng_states.begin(),
        (float*)samples.begin()}, samples.size(), use_gpu);
}

void PCGSampler::next_secondary_edge_samples(
        BufferView<TSecondaryEdgeSample<double>> samples) {
    parallel_for(pcg_sampler_double<4>{rng_states.begin(),
        (double*)samples.begin()}, samples.size(), use_gpu);
}
