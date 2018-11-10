#pragma once

#include "redner.h"
#include "ptr.h"
#include <memory>

struct Scene;
struct DScene;

struct RenderOptions {
    uint64_t seed;
    int num_samples;
    int max_bounces;
};

void render(const Scene &scene,
            const RenderOptions &options,
            ptr<float> rendered_image,
            ptr<float> d_rendered_image,
            std::shared_ptr<DScene> d_scene,
            ptr<float> debug_image);
