#pragma once

#include "redner.h"
#include <vector>

enum class Channels {
    radiance,
    alpha,
    depth,
    position,
    geometry_normal,
    shading_normal,
    uv,
    diffuse_reflectance,
    specular_reflectance,
    roughness,
    shape_id,
    material_id
};

struct ChannelInfo {
    ChannelInfo(const std::vector<Channels> &channels, bool use_gpu);
    void free();

    Channels *channels;
    int num_channels;
    int num_total_dimensions;
    int radiance_dimension;
    bool use_gpu;
};

int compute_num_channels(const std::vector<Channels> &channels);
