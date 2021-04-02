#pragma once

#include "redner.h"
#include "shape.h"
#include "camera.h"
#include "channels.h"

#include <memory>

struct Scene;
#include "scene.h"
#include "parallel.h"
#include "thrust_utils.h"
#include "ltc.inc"
#include "shape_adjacency.h"

#include "warp_common.h"

#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/remove.h>


DEVICE
inline
Vector2 aux_primary_local_pos( const KernelParameters& kernel_parameters,
                              const Camera &camera,
                              const Vector2& sample_local_pos,
                              const AuxSample &aux_sample) {
                                  
    const auto aux_s = aux_sample.uv;
    const auto auxPrimaryGaussianStddev = kernel_parameters.auxPrimaryGaussianStddev;

    auto normal_sample_x = (sqrt(-2 * log(aux_s[0])) * sin(M_PI * 2 * aux_s[1])) * auxPrimaryGaussianStddev;
    auto normal_sample_y = (sqrt(-2 * log(aux_s[0])) * cos(M_PI * 2 * aux_s[1])) * auxPrimaryGaussianStddev;
    
    auto local_pos = Vector2{normal_sample_x + sample_local_pos[0], normal_sample_y + sample_local_pos[1]};

    return local_pos;
}

/*
 * Computes a perturbed ray around wo by transforming aux_sample using the
 * von Mises-Fisher distribution.
 */
DEVICE
inline
Ray aux_sample(const KernelParameters &kernel_parameters,
                    const Material &material,
                    const SurfacePoint &shading_point,
                    const Vector3 &wi,
                    const Vector3 &wo,
                    const AuxSample &aux_sample) {
                        
    auto shading_frame = shading_point.shading_frame;
    if (has_normal_map(material)) {
        // Perturb shading frame
        shading_frame = perturb_shading_frame(material, shading_point);
    }
    auto geom_normal = shading_point.geom_normal;
    // Flip geometry normal to the same side of shading normal
    if (dot(geom_normal, shading_frame.n) < 0) {
        geom_normal = -geom_normal;
    }

    auto geom_wi = dot(geom_normal, wi);
    if (!material.two_sided) {
        // The surface doesn't reflect light on the other side of
        // the geometry normal.
        if (geom_wi < 0) {
            return Ray{shading_point.position, Vector3{0, 0, 0}};
        }
    }

    auto azimuth = Vector2{cos(aux_sample.uv.x * M_PI * 2), sin(aux_sample.uv.x * M_PI * 2)};
    
    auto k = kernel_parameters.vMFConcentration; // Concentration parameter.
    auto elevation = 1 + log(aux_sample.uv.y + exp(-2*k) * (1 - aux_sample.uv.y)) / k;

    auto sin_theta = sqrt(1 - elevation * elevation);
    // von Mises-Fisher unit vector sampled about the z-axis.
    auto vmf_zframe = normalize(Vector3{ sin_theta * azimuth.x, sin_theta * azimuth.y, elevation});

    // Rotate the vMF sample so that the mean is 'wo'.
    auto target_z = normalize(wo);
    Vector3 target_x, target_y;
    if(abs(dot(target_z, Vector3{0,0,1})) < 1e-6) {
        // wo is too close to z. set x and y to defaults.
        target_x = Vector3({1,0,0});
        target_y = Vector3({0,1,0});
    } else {
        // compute a simple orthonormal basis around z.
        target_x = cross(target_z, Vector3{0,0,1});
        target_x = normalize(target_x);
        target_y = cross(target_z, target_x);
    }

    Vector3 local_aux_dir = normalize(target_x * vmf_zframe.x +
                            target_y * vmf_zframe.y +
                            target_z * vmf_zframe.z);


    // Note that aux rays that are beyond the hemisphere bounds are OKAY.
    // These rays are requried to compute the contribution from the 
    // boundary of the hemisphere.
    // This is non-zero if the shading_point patch is rotating.
    return Ray{shading_point.position, local_aux_dir};
}


DEVICE
inline
Ray aux_sample_primary( const KernelParameters &kernel_parameters,
                    const Camera &camera,
                    const int idx,
                    const CameraSample& sample,
                    const AuxSample &aux_sample) {
                        
    Vector2 local_pos;
    sample_to_local_pos(camera,
                        sample,
                        local_pos);

    // Use the helper function to perform auxiliary sampling on the
    // pixel surface.
    auto aux_local_pos = aux_primary_local_pos(kernel_parameters, camera, local_pos, aux_sample);

    Vector2 aux_screen_pos;
    local_to_screen_pos(camera, idx, aux_local_pos, aux_screen_pos);

    // Generate new ray
    return sample_primary(camera, aux_screen_pos);
}

/*
 * Computes the pdf of the von Mises-Fisher distribution.
 */
DEVICE
inline
Real aux_pdf( const KernelParameters &kernel_parameters,
              const Ray &wo,
              const Ray &aux) {
    
    const auto k = kernel_parameters.vMFConcentration; // Concentration parameter.
    const auto norm_const = 2 * M_PI * (1 + exp(-2 * k)) / k;

    return exp(k * (dot(wo.dir,aux.dir) - 1)) / norm_const;
}

/*
 * Computes the pdf of the normal distribution with primary sample as mean.
 */
DEVICE
inline
Real aux_primary_pdf( const KernelParameters& kernel_parameters,
                    const Camera& camera,
                    const CameraSample& sample,
                    const AuxSample& aux_sample) {
    const auto auxPrimaryGaussianStddev = kernel_parameters.auxPrimaryGaussianStddev;
    const auto normalizer = 1/(Real(M_PI) * 2 * auxPrimaryGaussianStddev * auxPrimaryGaussianStddev);

    // Convert the samples to pixel space positions.
    Vector2 local_sample;
    sample_to_local_pos(camera, sample, local_sample);

    auto local_aux_sample = aux_primary_local_pos(kernel_parameters,
                                                camera, 
                                                local_sample, 
                                                aux_sample);

    auto logpdf = Real(0.5) * ((length_squared(local_sample - local_aux_sample)) / square(auxPrimaryGaussianStddev));
    auto pdf = exp(-logpdf);

    // We ever so slightly adjust the pdf to avoid infinities.
    auto adjustedPdf = (pdf * normalizer) + kernel_parameters.auxPdfEpsilonRegularizer;
    return adjustedPdf;
}

/*
 * Parallel version of aux_sample that works with ray buffers.
 */
void aux_bundle_sample( const KernelParameters &kernel_parameters,
                    const Scene& scene,
                    const BufferView<int> &active_pixels,
                    const BufferView<SurfacePoint> &shading_points,
                    const BufferView<Ray> &incoming_rays,
                    const BufferView<Intersection> &incoming_isects,
                    const BufferView<Ray> &primary_rays,
                    const BufferView<uint> &aux_sample_counts,
                    const BufferView<AuxSample> &aux_samples,
                    BufferView<Ray> aux_rays);

/*
 * Parallel version of aux_sample that works with ray buffers.
 */
void aux_bundle_sample_primary( const KernelParameters &kernel_parameters,
                    const Scene& scene,
                    const BufferView<int> &active_pixels,
                    const BufferView<uint> &aux_sample_counts,
                    const BufferView<Ray> &primary_rays,
                    const BufferView<CameraSample> &camera_samples,
                    const BufferView<AuxSample> &aux_samples,
                    BufferView<Ray> aux_rays);

/*
 * Transforms a set of samples to reduce variance.
 */
void aux_generate_antithetic_pairs( const KernelParameters& kernel_parameters,
                    const BufferView<int> &active_pixels,
                    BufferView<AuxSample> &aux_samples,
                    bool use_gpu);
