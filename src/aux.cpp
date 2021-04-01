#include "aux.h"

struct aux_sampler {
    DEVICE void operator()(int idx) {
        const auto &pixel_id = active_pixels[idx];
        for(uint i = 0; i < aux_sample_counts[pixel_id]; i++) {
                const auto aux_ray = aux_sample(kernel_parameters, 
                                        materials[
                                            shapes[incoming_isects[pixel_id].shape_id].material_id
                                        ],
                                        shading_points[pixel_id],
                                        -incoming_rays[pixel_id].dir, // Flip incoming so it's in correct coords.
                                        primary_rays[pixel_id].dir,
                                        samples[pixel_id * kernel_parameters.numAuxillaryRays + i]);
                aux_samples[pixel_id * kernel_parameters.numAuxillaryRays + i] = aux_ray;
        }
    }
    const KernelParameters kernel_parameters;
    const Shape* shapes;
    const Material* materials;
    const int* active_pixels;
    const SurfacePoint* shading_points;
    const Ray* incoming_rays;
    const Intersection* incoming_isects;
    const Ray* primary_rays;
    const uint* aux_sample_counts;
    const AuxSample* samples;
    Ray* aux_samples;

};

void aux_bundle_sample( const KernelParameters& kernel_parameters,
                    const Scene &scene,
                    const BufferView<int> &active_pixels,
                    const BufferView<SurfacePoint> &shading_points,
                    const BufferView<Ray> &incoming_rays,
                    const BufferView<Intersection> &incoming_isects,
                    const BufferView<Ray> &primary_rays,
                    const BufferView<uint> &aux_sample_counts,
                    const BufferView<AuxSample> &samples,
                    BufferView<Ray> aux_samples) {
    parallel_for(aux_sampler{
        kernel_parameters,
        scene.shapes.data,
        scene.materials.data,
        active_pixels.begin(),
        shading_points.begin(),
        incoming_rays.begin(),
        incoming_isects.begin(),
        primary_rays.begin(),
        aux_sample_counts.begin(),
        samples.begin(),
        aux_samples.begin()
    }, active_pixels.size(), scene.use_gpu);
}


struct primary_aux_sampler {
    DEVICE void operator()(int idx) {
        const auto &pixel_id = active_pixels[idx];
        for(uint i = 0; i < aux_sample_counts[pixel_id]; i++) {
            // No incoming rays or local intersection point.
            // For primary rays.
            const auto aux_ray = aux_sample_primary( kernel_parameters,
                                        *camera,
                                        pixel_id,
                                        camera_samples[pixel_id],
                                        samples[pixel_id * kernel_parameters.numAuxillaryRays + i]);
            aux_samples[pixel_id * kernel_parameters.numAuxillaryRays + i] = aux_ray;
        }
    }

    const KernelParameters kernel_parameters;
    const Camera* camera;
    const int* active_pixels;
    const uint* aux_sample_counts;
    const Ray* primary_rays;
    const CameraSample* camera_samples;
    const AuxSample* samples;

    Ray* aux_samples;
};


void aux_bundle_sample_primary( const KernelParameters& kernel_parameters,
                    const Scene &scene,
                    const BufferView<int> &active_pixels,
                    const BufferView<uint> &aux_sample_counts,
                    const BufferView<Ray> &primary_rays,
                    const BufferView<CameraSample> &camera_samples,
                    const BufferView<AuxSample> &samples,
                    BufferView<Ray> aux_samples) {
        parallel_for(primary_aux_sampler{
            kernel_parameters,
            &scene.camera,
            active_pixels.begin(),
            aux_sample_counts.begin(),
            primary_rays.begin(),
            camera_samples.begin(),
            samples.begin(),
            aux_samples.begin(),
        }, active_pixels.size(), scene.use_gpu);
}


struct aux_antithetic_pair_generator {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];
        for(int i = 0; i < kernel_parameters.numAuxillaryRays; i += 2) {
            // For every other aux sample, overwrite it with the
            // anti sample of the previous aux sample.
            auto source_idx = pixel_id * kernel_parameters.numAuxillaryRays + i;
            auto target_idx = pixel_id * kernel_parameters.numAuxillaryRays + i + 1;
            samples[target_idx] = AuxSample{
                Vector2(samples[source_idx].uv[0], 
                    (
                        samples[source_idx].uv[1] + 0.5) > 1 ?
                        samples[source_idx].uv[1] - 0.5 :
                        samples[source_idx].uv[1] + 0.5
                    )
            };
        }
    }
    const KernelParameters kernel_parameters;
    const int* active_pixels;
    AuxSample *samples;
};

/* 
 *   Antithetic sampler.
 */
void aux_generate_antithetic_pairs( const KernelParameters& kernel_parameters,
    const BufferView<int> &active_pixels,
    BufferView<AuxSample> &aux_samples,
    bool use_gpu) {
    // Select an even number for sample count.
    assert(aux_samples.size() % 2 == 0);

    parallel_for(aux_antithetic_pair_generator{
                kernel_parameters, active_pixels.begin(), aux_samples.begin()
            },
            active_pixels.size(), use_gpu);
}