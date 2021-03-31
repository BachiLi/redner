#pragma once

#include "redner.h"
#include "ptr.h"
#include "channels.h"
#include "pathtracer.h"
#include "warp_field.h"
#include <memory>

struct Scene;
struct DScene;

namespace vfield {
    enum class VarianceReduction {
        none,
        antithetic_variate,

        // Yet to be implemented
        control_variate
    };

    enum class BiasCorrection {
        none,

        // Yet to be implemented
        russian_roulette
    };

    enum class ImportanceSampling {
        cosine_hemisphere,

        // Yet to be implemented
        metropolis_mcmc,
        hamiltonian_mcmc,
    };

    struct VarianceReductionSettings {
        bool primary_antithetic_variates;// = true;
        bool aux_antithetic_variates;// = true;
        bool primary_control_variates;// = false;
        bool aux_control_variates;//= false;

        bool secondary_antithetic_variates;// = false;

        int num_control_samples;// = 0;
    };

    struct SamplingDebugSettings {
        bool disable_primary_edge_derivative = false;
        bool disable_primary_interior_derivative = false;
        bool disable_edge_derivative = false;
        bool disable_interior_derivative = false;

        bool secondary_override_enable = false;
        Vector3 secondary_ray = Vector3{0.0, 0.0, 1.0};

        bool secondary_aux_override_enable = false;
        Vector3 secondary_aux_ray = Vector3{0.0, 0.0, 1.0};
    };

    struct RenderOptions {
        uint64_t seed;
        int num_samples;
        int max_bounces;
        std::vector<Channels> channels;
        SamplerType sampler_type;
        SamplerType aux_sampler_type;
        VarianceReductionSettings variance_reduction_mode;
        ImportanceSampling importance_sampling_mode;
        KernelParameters kernel_parameters;
        bool enable_primary_warp_field;
        bool enable_secondary_warp_field;
        bool sample_pixel_center;
    };

    void render(Scene &scene,
                const RenderOptions &options,
                ptr<float> rendered_image,
                ptr<float> d_rendered_image,
                std::shared_ptr<DScene> d_scene,
                ptr<float> screen_gradient_image,
                ptr<float> debug_image);

};