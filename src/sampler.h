#pragma once

#include "buffer.h"
#include "vector.h"
#include "camera.h"
#include "area_light.h"
#include "material.h"
#include "edge.h"

struct Sampler {
    virtual ~Sampler() {}
    virtual void begin_sample(int sample_id) {};

    virtual void next_camera_samples(BufferView<TCameraSample<float>> samples, bool sample_pixel_center) = 0;
    virtual void next_camera_samples(BufferView<TCameraSample<double>> samples, bool sample_pixel_center) = 0;
    virtual void next_light_samples(BufferView<TLightSample<float>> samples) = 0;
    virtual void next_light_samples(BufferView<TLightSample<double>> samples) = 0;
    virtual void next_bsdf_samples(BufferView<TBSDFSample<float>> samples) = 0;
    virtual void next_bsdf_samples(BufferView<TBSDFSample<double>> samples) = 0;
    virtual void next_primary_edge_samples(BufferView<TPrimaryEdgeSample<float>> samples) = 0;
    virtual void next_primary_edge_samples(BufferView<TPrimaryEdgeSample<double>> samples) = 0;
    virtual void next_secondary_edge_samples(BufferView<TSecondaryEdgeSample<float>> samples) = 0;
    virtual void next_secondary_edge_samples(BufferView<TSecondaryEdgeSample<double>> samples) = 0;
};
