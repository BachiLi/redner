#pragma once

#include "redner.h"
#include "ptr.h"
#include "channels.h"
#include <memory>

struct Scene;
struct DScene;

enum class SamplerType {
	independent,
	sobol
};

struct RenderOptions {
    uint64_t seed;
    int num_samples;
    int max_bounces;
    std::vector<Channels> channels;
    SamplerType sampler_type;
    bool sample_pixel_center;
};

void render(const Scene &scene,
            const RenderOptions &options,
            ptr<float> rendered_image,
            ptr<float> d_rendered_image,
            std::shared_ptr<DScene> d_scene,
            ptr<float> screen_gradient_image,
            ptr<float> debug_image);
