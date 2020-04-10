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
    barycentric_coordinates,
    diffuse_reflectance,
    specular_reflectance,
    roughness,
    generic_texture,
    vertex_color,
    shape_id,
    triangle_id,
    material_id
};

struct ChannelInfo {
    ChannelInfo(const std::vector<Channels> &channels,
                bool use_gpu,
                int max_generic_texture_dimension);
    void free();

    Channels *channels;
    int num_channels;
    int num_total_dimensions;
    int radiance_dimension;
    int max_generic_texture_dimension;
    bool use_gpu;
};

int compute_num_channels(const std::vector<Channels> &channels,
                         int max_generic_texture_dimension);
