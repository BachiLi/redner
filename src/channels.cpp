#include "channels.h"
#include "cuda_utils.h"
#include <iostream>
#include <stdexcept>

ChannelInfo::ChannelInfo(const std::vector<Channels> &channels,
                         bool use_gpu,
                         int max_generic_texture_dimension) : use_gpu(use_gpu) {
    num_channels = (int)channels.size();
    radiance_dimension = -1;
    num_total_dimensions = compute_num_channels(channels, max_generic_texture_dimension);
    this->max_generic_texture_dimension = max_generic_texture_dimension;
    if (use_gpu) {
#ifdef __CUDACC__
        checkCuda(cudaMallocManaged(&this->channels, channels.size() * sizeof(Channels)));
#else
        assert(false);
#endif
    } else {
        this->channels = new Channels[channels.size()];
    }
    for (int i = 0; i < (int)channels.size(); i++) {
        if (channels[i] == Channels::radiance) {
            if (radiance_dimension != -1) {
                throw std::runtime_error("Duplicated radiance channel");
            }
            radiance_dimension = i;
        }
        if (use_gpu) {
            // We don't use unified memory to update here since another kernel might be running
#ifdef __CUDACC__
            checkCuda(cudaMemcpyAsync(&this->channels[i], &channels[i], sizeof(Channels), cudaMemcpyHostToDevice));
#else
            assert(false);
#endif
        } else {
            this->channels[i] = channels[i];
        }
    }
}

void ChannelInfo::free() {
    if (use_gpu) {
#ifdef __CUDACC__
        checkCuda(cudaFree(channels));
#else
        assert(false);
#endif
    } else {
        delete[] channels;
    }
}

int compute_num_channels(const std::vector<Channels> &channels,
                         int max_generic_texture_dimension) {
    int num_total_dimensions = 0;
    for (int i = 0; i < (int)channels.size(); i++) {
        switch(channels[i]) {
            case Channels::radiance: {
                num_total_dimensions += 3;
            } break;
            case Channels::alpha: {
                num_total_dimensions += 1;
            } break;
            case Channels::depth: {
                num_total_dimensions += 1;
            } break;
            case Channels::position: {
                num_total_dimensions += 3;
            } break;
            case Channels::geometry_normal: {
                num_total_dimensions += 3;
            } break;
            case Channels::shading_normal: {
                num_total_dimensions += 3;
            } break;
            case Channels::uv: {
                num_total_dimensions += 2;
            } break;
            case Channels::barycentric_coordinates: {
                num_total_dimensions += 2;
            } break;
            case Channels::diffuse_reflectance: {
                num_total_dimensions += 3;
            } break;
            case Channels::specular_reflectance: {
                num_total_dimensions += 3;
            } break;
            case Channels::roughness: {
                num_total_dimensions += 1;
            } break;
            case Channels::generic_texture: {
                num_total_dimensions += max_generic_texture_dimension;
            } break;
            case Channels::vertex_color: {
                num_total_dimensions += 3;
            } break;
            case Channels::shape_id: {
                num_total_dimensions += 1;
            } break;
            case Channels::triangle_id: {
                num_total_dimensions += 1;
            } break;
            case Channels::material_id: {
                num_total_dimensions += 1;
            } break;
            default: {
                assert(false);
            }
        }
    }
    return num_total_dimensions;
}
