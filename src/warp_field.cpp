
#include "redner.h"
#include "shape.h"
#include "camera.h"
#include "channels.h"
#include "edge_tree.h"
#include "warp_field.h"

#include <memory>

struct Scene;
#include "edge.h"
#include "scene.h"
#include "parallel.h"
#include "thrust_utils.h"
#include "ltc.inc"
#include <memory>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/remove.h>


DEVICE
inline
int _aux_sample_sample_counts(KernelParameters kernel_parameters,
                                AuxCountSample sample) {
    // Sample N according to some distribution.
    int g = kernel_parameters.batch_size; // granularity.
    Real p = kernel_parameters.rr_geometric_p; // Geometric distribution probability

    int max_val = kernel_parameters.numAuxillaryRays;
    
    auto k = static_cast<int>(floor(log(1 - sample.u) / log(1 - p))) + 1;

    if(!isfinite(k)) // Handle numerical infs and nans..
        return max_val;

    // Clamp to maximum allocated space.
    // NOTE: Does not account for truncation bias! (This should be implemented in the future)
    return std::max(g, std::min(k * g, max_val));
}

DEVICE
inline
Real _aux_sample_sample_counts_pdf(KernelParameters kernel_parameters, 
                                    int num_samples) {
    // Return pdf of the distribution.
    int g = kernel_parameters.batch_size; // granularity.
    Real p = kernel_parameters.rr_geometric_p; // Geometric distribution probability

    int k = (num_samples) / g;

    return pow(1 - p, k - 1);
}

struct aux_sample_count_sampler {
    DEVICE void operator()(int idx) {
        const auto &pixel_id = active_pixels[idx];
        const auto num_samples = _aux_sample_sample_counts(
                                    kernel_parameters, aux_count_samples[pixel_id]);
        aux_sample_counts[pixel_id] = static_cast<uint>(num_samples);
    }

    const int* active_pixels;
    const KernelParameters kernel_parameters;
    const AuxCountSample* aux_count_samples;
    uint* aux_sample_counts;

};

void aux_sample_sample_counts( const KernelParameters& kernel_parameters,
                    const Scene& scene,
                    const BufferView<int> &active_pixels,
                    const BufferView<AuxCountSample> &aux_count_samples,
                    BufferView<uint> &aux_sample_counts) {
    parallel_for(aux_sample_count_sampler{
        active_pixels.begin(),
        kernel_parameters,
        aux_count_samples.begin(),
        aux_sample_counts.begin()
    }, active_pixels.size(), scene.use_gpu);
}

DEVICE
inline
Ray aux_sample(const KernelParameters& kernel_parameters, 
                    const Material &material,
                    const SurfacePoint &shading_point,
                    const Vector3 &wi,
                    const Vector3 &wo,
                    const AuxSample &aux_sample,
                    bool mirrored = false,
                    bool debug = false) {
    /*
     * Computes a perturbed ray around wo by transforming aux_sample using the
     * von Mises-Fisher distribution.
     */

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
Vector2 aux_sample_primary_local_pos( const KernelParameters& kernel_parameters,
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

DEVICE
inline
Ray aux_sample_primary( const KernelParameters& kernel_parameters,
                    const Camera &camera,
                    const int idx,
                    const CameraSample& sample,
                    const AuxSample &aux_sample,
                    bool debug = false) {

    Vector2 local_pos;
    sample_to_local_pos(camera,
                        sample,
                        local_pos);

    // Use the helper function to perform auxillary sampling on the
    // pixel surface.
    auto aux_local_pos = aux_sample_primary_local_pos(kernel_parameters, camera, local_pos, aux_sample);

    Vector2 aux_screen_pos;
    local_to_screen_pos(camera, idx, aux_local_pos, aux_screen_pos);

    Vector2 screen_pos;
    local_to_screen_pos(camera, idx, local_pos, screen_pos);

    // Generate new ray
    return sample_primary(camera, aux_screen_pos);
}

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


struct aux_correlated_pair_generator {
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

void aux_generate_correlated_pairs( const KernelParameters& kernel_parameters,
    const BufferView<int> &active_pixels,
    BufferView<AuxSample> &aux_samples,
    bool use_gpu) {
    // Select an even number for sample count.
    assert(aux_samples.size() % 2 == 0);

    parallel_for(aux_correlated_pair_generator{
                kernel_parameters, active_pixels.begin(), aux_samples.begin()
            },
            active_pixels.size(), use_gpu);
}

DEVICE
inline
Real aux_pdf( const KernelParameters& kernel_parameters,
                const Ray &wo,
                const Ray &aux,
                bool debug = false) {
    /*
     * Computes the pdf of the von Mises-Fisher distribution,
     * for secondary samples.
     */
    const auto k = kernel_parameters.vMFConcentration; // Concentration parameter.
    const auto norm_const = 2 * M_PI * (1 + exp(-2 * k)) / k;

    return exp(k * (dot(wo.dir,aux.dir) - 1)) / norm_const;
}

DEVICE
inline
Real aux_primary_pdf( const KernelParameters& kernel_parameters,
                    const Camera& camera,
                    const CameraSample& sample,
                    const AuxSample& aux_sample, 
                    bool debug = false) {
    /*
     * Computes the pdf of the gaussian distribution.
     * For primary samples.
     */
    const auto auxPrimaryGaussianStddev = kernel_parameters.auxPrimaryGaussianStddev;
    const auto normalizer = 1/(Real(M_PI) * 2 * auxPrimaryGaussianStddev * auxPrimaryGaussianStddev);

    // Convert the samples to pixel space positions.
    Vector2 local_sample;
    sample_to_local_pos(camera, sample, local_sample);

    // TODO: Split aux_sampler and the distribution into modules...
    // Current structure is very messy and entangled
    // Preferably move into a different file.
    auto local_aux_sample = aux_sample_primary_local_pos(kernel_parameters,
                                            camera, 
                                            local_sample, 
                                             aux_sample);

    auto logpdf = Real(0.5) * ((length_squared(local_sample - local_aux_sample)) / square(auxPrimaryGaussianStddev));
    auto pdf = exp(-logpdf);

    // We ever so slightly adjust the pdf to avoid infinities. 
    // Theoretically it can be proven that the error caused by
    // this regularizer decreases (atleast) linearly in epsilon.
    auto adjustedPdf = (pdf * normalizer) + kernel_parameters.auxPdfEpsilonRegularizer;
    return adjustedPdf;
}

DEVICE
template <typename T>
inline TVector3<T> v3(const T* data) {
    return TVector3<T>(data[0], data[1], data[2]);
}

DEVICE
template <typename T>
inline TVector3<T> expand(const TVector2<T> v2) {
    return TVector3<T>(v2.x, v2.y, static_cast<T>(0.f));
}

DEVICE
inline
Real warp_horizon_term(
                    const KernelParameters &kernel_parameters,
                    const Shape *shapes,
                    const ShapeAdjacency *shape_adjacencies,
                    const SurfacePoint &shading_point,
                    const SurfacePoint &aux_point,
                    const Intersection &aux_isect,
                    int rel_vidx) {

    // Also check for non-manifold edges.
    auto adj0 = shape_adjacencies[aux_isect.shape_id].adjacency[aux_isect.tri_id * 3 + rel_vidx];
    auto adj1 = shape_adjacencies[aux_isect.shape_id].adjacency[aux_isect.tri_id * 3 + (rel_vidx + 2)%3];
    if(adj0 == -1 || adj1 == -1) {
        return Real(1.0);
    }

    auto vidx = get_indices(shapes[aux_isect.shape_id], aux_isect.tri_id)[rel_vidx];
    std::vector<int> face_indices = shape_adjacencies[aux_isect.shape_id].vertex_adjacency[vidx];

    Real total_inner_product = Real(0.0);
    auto w = normalize(aux_point.position - shading_point.position);
    for(int i = 0; i < face_indices.size(); i++) {
        int face_idx = face_indices.at(i);
        auto vidxs = get_indices(shapes[aux_isect.shape_id], face_idx);
        
        auto a0 = v3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[0]]);
        auto a1 = v3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[1]]);
        auto a2 = v3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[2]]);

        auto an = normalize(cross(a0 - a1, a1 - a2));
        
        total_inner_product += dot(an, w);

        // If a single adjacent face point away from the source, then
        // this must be a boundary vertex.
        if((dot(an, w) * total_inner_product) < 0) return Real(1.0);
    }

    Real avg_inclination = total_inner_product / face_indices.size();
    auto alpha = 100;
    auto term = alpha / (1.0 - avg_inclination * avg_inclination) - (alpha - 1);

    return term;
}


DEVICE
inline
Real warp_boundary_term( const KernelParameters& kernel_parameters,
                  const Shape *shapes,
                  const ShapeAdjacency *shape_adjacencies,
                  const Ray &primary,
                  const Ray &auxillary,
                  const Intersection &aux_isect,
                  const SurfacePoint &aux_point,
                  const SurfacePoint &shading_point,
                  Real& horizon_term) {
    // Compute the boundary term S(x). 
    // This is simply exponentiated negative distance adjusted by the horizon term.
    auto boundary_term = Real(0.0);
    if (dot(shading_point.geom_normal, auxillary.dir) * dot(shading_point.geom_normal, primary.dir) < 1e-4) {
        // Outside the horizon ('hit' the black hemispherical occluder)
        horizon_term = 0.0;
        boundary_term = exp(-abs(dot(shading_point.geom_normal, auxillary.dir)) / kernel_parameters.asymptoteBoundaryTemp);
    } else if (aux_isect.valid()) {

        // Hit a valid surface (and isn't beyond horizon)
        auto vidxs = get_indices(shapes[aux_isect.shape_id], aux_isect.tri_id);
    
        auto p0 = v3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[0]]);
        auto p1 = v3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[1]]);
        auto p2 = v3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[2]]);
        auto p = aux_point.position;

        // Compute the edges.
        auto e0 = p0 - p1;
        auto e1 = p1 - p2;
        auto e2 = p2 - p0;

        // Compute distance to edges. 
        auto A = length(cross(e0, e1));
        auto w = length(cross(e0, p - p0)) / A; // barycentric of vtx2
        auto u = length(cross(e1, p - p1)) / A; // barycentric of vtx0
        auto v = length(cross(e2, p - p2)) / A; // barycentric of vtx1

        auto horiz0 = warp_horizon_term(kernel_parameters, 
                                        shapes, 
                                        shape_adjacencies, 
                                        shading_point, 
                                        aux_point, 
                                        aux_isect, 0);
        auto horiz1 = warp_horizon_term(kernel_parameters, 
                                        shapes, 
                                        shape_adjacencies, 
                                        shading_point, 
                                        aux_point,
                                        aux_isect, 1);
        auto horiz2 = warp_horizon_term(kernel_parameters, 
                                        shapes, 
                                        shape_adjacencies, 
                                        shading_point, 
                                        aux_point,
                                        aux_isect, 2);

        Real interpolated_inv_horiz_term = (u/horiz0 + v/horiz1 + w/horiz2) / (u + v + w);
        boundary_term = interpolated_inv_horiz_term;
        horizon_term = 1.0 / interpolated_inv_horiz_term;
        assert(boundary_term <= 1.f);
        assert(boundary_term >= 0.f);
        
    } else {
        // Hit environment map. No sharp boundaries..
        horizon_term = 0.0;
        boundary_term = 0.0;
    }

    assert(boundary_term <= 1.f);
    assert(boundary_term >= 0.f);
    return boundary_term;
}

DEVICE
inline
Real warp_weight( const KernelParameters& kernel_parameters,
                  const Shape *shapes,
                  const ShapeAdjacency *shape_adjacencies,
                  const Ray &primary,
                  const Ray &auxillary,
                  const Intersection &aux_isect,
                  const SurfacePoint &aux_point,
                  const SurfacePoint &shading_point,
                  Real &boundary_term,
                  Real &horizon_term) {
    /*
     * Computes the asymptotic weight of the auxillary ray's contribution to
     * the warp field. This is independent of the parameter.
     */
    
    if(kernel_parameters.isBasicNormal) {
        const auto k = kernel_parameters.vMFConcentration;
        return (k * exp(k * (dot(primary.dir, auxillary.dir) - 1))) / (2.0 * M_PI * (1 - exp(-2 * k)));
    }

    boundary_term = warp_boundary_term(kernel_parameters,
                 shapes,
                 shape_adjacencies,
                 primary,
                 auxillary,
                 aux_isect,
                 aux_point,
                 shading_point,
                 horizon_term);

    // Compute the inverse gaussian term.
    //auto inv_gauss = exp(square(1 - dot(primary.dir, auxillary.dir)) / float(kernel_parameters.asymptoteInvGaussSigma));
    
    const auto gamma = kernel_parameters.asymptoteGamma;
    const auto k = kernel_parameters.vMFConcentration / gamma;
    auto gauss = exp(k * (dot(primary.dir, auxillary.dir) - 1));

    // Compute the harmonic weight.
    auto harmonic = pow(gauss, gamma) / pow(1 - gauss * boundary_term, gamma);

    assert(harmonic >= 0.f);
    return harmonic;
}

DEVICE
inline
Real warp_weight_primary( const KernelParameters& kernel_parameters,
                  const Camera& camera,
                  const Shape* shapes,
                  const ShapeAdjacency *shape_adjacencies,
                  const Ray &primary,
                  const Ray &auxillary,
                  const Vector2 &local_sample,
                  const Vector2 &local_aux_sample,
                  const Intersection &aux_isect,
                  const SurfacePoint &aux_point,
                  const SurfacePoint &shading_point,
                  Real &boundary_term,
                  Real &horizon_term) {
    /*
     * Computes the asymptotic weight of the auxillary ray's contribution to
     * the warp field. This is independent of the parameter.
     */

    // Optionally we can use a simple gaussian kernel. This leads to biased results, however.
    if(kernel_parameters.isBasicNormal) {
        const auto sigma = kernel_parameters.auxPrimaryGaussianStddev * sqrt(kernel_parameters.asymptoteGamma);
        return exp(-0.5 * length_squared(local_sample - local_aux_sample) / 
                    float(sigma * sigma)) / 
                    (2.0 * M_PI * float(sigma * sigma));
    }

    const auto sigma = kernel_parameters.auxPrimaryGaussianStddev * sqrt(kernel_parameters.asymptoteGamma);

    boundary_term = warp_boundary_term(kernel_parameters,
                 shapes,
                 shape_adjacencies,
                 primary,
                 auxillary,
                 aux_isect,
                 aux_point,
                 shading_point,
                 horizon_term);

    // Compute the gaussian term.
    auto gauss = exp(-0.5 * length_squared(local_sample - local_aux_sample) / 
                float(sigma * sigma));
    
    // Compute the harmonic weight.
    auto harmonic = pow(gauss, kernel_parameters.asymptoteGamma) / pow(1 - gauss * boundary_term, kernel_parameters.asymptoteGamma);

    return harmonic;
}

DEVICE
inline
Vector3 warp_weight_grad( const KernelParameters& kernel_parameters,
                  const Shape* shapes,
                  const ShapeAdjacency *shape_adjacencies,
                  const Ray &primary,
                  const Ray &auxillary,
                  const Intersection &aux_isect,
                  const SurfacePoint &aux_point,
                  const SurfacePoint &shading_point ) {
    /*
     * Computes the derivative of the asymptotic weight of the 
     * auxillary ray's contribution to
     * the warp field. This is independent of the parameter.
     */
    if(kernel_parameters.isBasicNormal) {
        const auto k = kernel_parameters.vMFConcentration;
        return (k * k * exp(k * (dot(primary.dir, auxillary.dir) - 1)) * cross(primary.dir,cross(primary.dir, auxillary.dir)) ) / (2.0 * M_PI * (1 - exp(-2 * k)));
    }

    // Compute the boundary term S(x). 
    // This is simply inverse exponential distance product.
    // Has no gradient in x. Only dependent on aux.
    Real horizon_term = 0.0;
    auto boundary_term = warp_boundary_term(kernel_parameters,
                 shapes,
                 shape_adjacencies,
                 primary,
                 auxillary,
                 aux_isect,
                 aux_point,
                 shading_point,
                 horizon_term);

    // Compute the inverse gaussian term.
    //auto inv_gauss = exp(square(1 - dot(primary.dir, auxillary.dir)) / asymptoteInvGaussSigma);
    const auto gamma = kernel_parameters.asymptoteGamma;
    const auto k = kernel_parameters.vMFConcentration / gamma;

    //auto gauss = exp(-square(1 - dot(primary.dir, auxillary.dir)) / kernel_parameters.asymptoteInvGaussSigma);
    auto gauss = exp(k * (dot(primary.dir, auxillary.dir) - 1));

    // Gradient of inverse gaussian.
    auto inv_gauss_grad = exp(k * (1 - dot(primary.dir, auxillary.dir))) * k * cross(primary.dir,cross(primary.dir, auxillary.dir));

    // Compute the harmonic weight.
    auto harmonic = -gamma * pow(gauss, gamma + 1) / pow(1 - boundary_term * gauss, gamma + 1);

    // Compute the gradient of the harmonic weight w.r.t 'wo'.
    auto harmonic_gradient = harmonic * inv_gauss_grad;

    // (Note that this is a vector quantity)
    return harmonic_gradient;
}

DEVICE
inline
Vector3 warp_weight_grad_primary( const KernelParameters& kernel_parameters,
                  const Camera& camera,
                  const Shape* shapes,
                  const ShapeAdjacency *shape_adjacencies,
                  const Ray &primary,
                  const Ray &auxillary,
                  const Vector2 &local_pos,
                  const Vector2 &aux_local_pos,
                  const Intersection &aux_isect,
                  const SurfacePoint &aux_point,
                  const SurfacePoint &shading_point ) {

    /*
     * Computes the derivative of the asymptotic weight of the 
     * auxillary ray's contribution to
     * the warp field. This is independent of the parameter.
     */

    if(kernel_parameters.isBasicNormal) {
        const auto sigma = kernel_parameters.auxPrimaryGaussianStddev * sqrt(kernel_parameters.asymptoteGamma);
        const Vector2 grad = (exp(-0.5 * length_squared(local_pos - aux_local_pos) / 
                    float(sigma * sigma)) / 
                    (2.0 * M_PI * float(sigma * sigma))) * (aux_local_pos - local_pos) / (float(sigma * sigma));
        return Vector3 {grad.x, grad.y, 0.0};
    }

    // Compute the boundary term S(x). 
    // This is simply inverse exponential distance product.
    // Has no gradient in x. Only dependent on aux.
    Real horizon_term = 0.0;
    auto boundary_term = warp_boundary_term(kernel_parameters,
                 shapes,
                 shape_adjacencies,
                 primary,
                 auxillary,
                 aux_isect,
                 aux_point,
                 shading_point,
                 horizon_term);

    const auto sigma = kernel_parameters.auxPrimaryGaussianStddev * sqrt(kernel_parameters.asymptoteGamma);

    const auto asymptoteGamma = kernel_parameters.asymptoteGamma;
    // Compute the gaussian term.
    auto gauss = exp(-0.5 * length_squared(local_pos - aux_local_pos) / (sigma * sigma));
    auto gauss_gamma = pow(gauss, asymptoteGamma);

    // Gradient of gaussian.
    auto gauss_grad = gauss_gamma * (local_pos - aux_local_pos) / float(sigma * sigma);

    // Gradient of gamma-gaussian
    auto gauss_grad_gamma = gauss_grad * float(asymptoteGamma);

    // Compute the harmonic weight.
    auto harmonic = 1.0 / pow(1 - gauss * boundary_term, asymptoteGamma + 1);

    // Compute the gradient of the harmonic weight w.r.t 'wo'.
    auto harmonic_gradient = -harmonic * gauss_grad_gamma;

    // (Note that this is a vector quantity)
    // Add a dummy dimension to have some type uniformity for primary and secondary
    // samples.
    // We now return grad_w/w instead of just grad_w owing to numerical instability
    // in the latter method.
    return Vector3{harmonic_gradient.x, harmonic_gradient.y, 0.0};
}


DEVICE
inline
void warp_jacobian( const KernelParameters& kernel_parameters,
                    const SurfacePoint& shading_point,
                    const SurfacePoint& aux_point,
                    const Intersection& aux_isect,
                    Matrix3x3 &d_dir_d_xg ) {
    /*
     * Computes the jacobian of local sample coordinates
     * w.r.t the intersection point
     */

    if (!aux_isect.valid()) {
        // No intersection => no global coordinate derivative.
        d_dir_d_xg = Matrix3x3::zeros();
        return;
    }

    // Compute distance vector from current point to the auxillary intersection.
    auto aux_vector = aux_point.position - shading_point.position;

    auto dist = length(aux_vector);

    // Compute the local-global jacobian matrix here.
    d_dir_d_xg = Matrix3x3::identity() * (1 / dist) - 
                outer_product(aux_vector, aux_vector) / (dist * dist * dist);
    
}


DEVICE
inline
void warp_jacobian_primary(const KernelParameters& kernel_parameters,
                            const Camera& camera,
                            const Vector2& local_pos,
                            const int pixel_id,
                            const SurfacePoint& aux_point,
                            const Intersection& aux_isect,
                            Matrix3x3& d_xy_d_xg,
                            Matrix3x3& d_xy_d_dir,
                            Matrix3x3& d_xy_d_org) {
    /*
     * Computes the jacobian (matrix) of local sample coordinates
     * w.r.t the intersection point
     * For primary rays, the local sample coordinates are the 2D uniform
     * sample in the square pixel grid.
     * Note that even though the local samples are 2 dimensional, we return a
     * 3x3 jacobian (instead of 2x3), because redner doesn't support arbitrary
     * matrices.
     */

    // Call camera-specific jacobian functions from camera.h for d_xy_d_dir and d_xy_d_org
    screen_dir_jacobian(camera, pixel_id, local_pos, d_xy_d_dir);
    screen_org_jacobian(camera, pixel_id, local_pos, d_xy_d_org);

    // Compute dw/dxg.
    if(aux_isect.valid()) {
        Vector3 d_x_d_lxg(0,0,0);
        Vector3 d_y_d_lxg(0,0,0);

        Vector3 d_x_d_xg(0,0,0);
        Vector3 d_y_d_xg(0,0,0);

        Vector3 xfm_pt = xfm_point(camera.world_to_cam, aux_point.position);

        Vector2 d_local_x(0, 0);
        Vector2 d_local_y(0, 0);
        d_screen_to_local_pos(camera, pixel_id, Vector2(1, 0), d_local_x);
        d_screen_to_local_pos(camera, pixel_id, Vector2(0, 1), d_local_y);

        // Compute derivatives of screen coords w.r.t camera-space point 'lxg'.
        DCamera null_camera = DCamera::null();
        d_camera_to_screen(camera, xfm_pt, d_local_x.x, d_local_x.y, null_camera, d_x_d_lxg);
        d_camera_to_screen(camera, xfm_pt, d_local_y.x, d_local_y.y, null_camera, d_y_d_lxg);

        Matrix4x4 temp0, temp1;
        // Compute derivatives of screen coords w.r.t world-space point 'xg'.
        d_xfm_point(camera.world_to_cam, aux_point.position, d_x_d_lxg, temp0, d_x_d_xg);
        d_xfm_point(camera.world_to_cam, aux_point.position, d_y_d_lxg, temp1, d_y_d_xg);

        d_xy_d_xg = Matrix3x3{
            d_x_d_xg[0], d_x_d_xg[1], d_x_d_xg[2],
            d_y_d_xg[0], d_y_d_xg[1], d_y_d_xg[2],
            0.0, 0.0, 0.0
        };

    } else {
        d_xy_d_xg = Matrix3x3::zeros();
    }
}


/*
 * (Utility method for control variates)
 * Computes the spatial derivatives of the warp_jacobian.
 * This is used for a control variate with linear assumptions to sharply reduce the variance.
 */
DEVICE
inline
void warp_gradient_tensor( const KernelParameters& kernel_parameters,
                                 const Shape* shapes,
                                 const Camera& camera,
                                 const Vector2& local_pos,
                                 const int pixel_id,
                                 const SurfacePoint& point,
                                 const Intersection& isect,
                                 const Ray& ray,
                                 const float epslion, // default.. eps=1e-4
                                 Matrix3x3* d_xy_d_xg_d_barycentrics,
                                 Vector3& barycentrics) {

    auto vidxs = get_indices(shapes[isect.shape_id], isect.tri_id);

    // Find the triangle vertices and compute alpha, beta and gamma.
    TVector3<double> p0 = TVector3<double>(v3(&shapes[isect.shape_id].vertices[3 * vidxs[0]]));
    TVector3<double> p1 = TVector3<double>(v3(&shapes[isect.shape_id].vertices[3 * vidxs[1]]));
    TVector3<double> p2 = TVector3<double>(v3(&shapes[isect.shape_id].vertices[3 * vidxs[2]]));

    auto u_dxy = Vector2{0,0};
    auto v_dxy = Vector2{0,0};
    auto t_dxy = Vector2{0,0};
    TRayDifferential<double> zero_d_ray_differential{
                    TVector3<double>{Real(0), Real(0), Real(0)}, TVector3<double>{Real(0), Real(0), Real(0)},
                  TVector3<double>{Real(0), Real(0), Real(0)}, TVector3<double>{Real(0), Real(0), Real(0)}};
    auto uvt = intersect(p0, p1, p2, ray, zero_d_ray_differential, u_dxy, v_dxy, t_dxy);

    barycentrics = Vector3{(1 - (uvt[0] + uvt[1])), uvt[0], uvt[1]};
    // Interpolate the triangle vertices with alpha +- eps, beta +- eps and gamma +- eps.

    for(int i = 1; i < 3; i++) {
        Vector3 eps_select = Vector3{0.0, 0.0, 0.0};
        eps_select[i] = 1; // i=1, 2 are independent variables.
        eps_select[0] = -1; // Dependent variable..

        auto barycentrics_eps_plus = barycentrics + eps_select * epslion;
        auto barycentrics_eps_minus = barycentrics - eps_select * epslion;

        auto xg_eps_plus = barycentrics_eps_plus.x * p0 + 
                           barycentrics_eps_plus.y * p1 + 
                           barycentrics_eps_plus.z * p2;

        auto xg_eps_minus = barycentrics_eps_minus.x * p0 + 
                            barycentrics_eps_minus.y * p1 + 
                            barycentrics_eps_minus.z * p2;

        SurfacePoint sp_eps_plus = point;
        SurfacePoint sp_eps_minus = point;
        sp_eps_plus.position = xg_eps_plus;
        sp_eps_minus.position = xg_eps_minus;
        // Compute individual gradients.
        Matrix3x3 _dw_dxg_plus, _dw_dxg_minus, _dw_ddir, _dw_dorg;
        warp_jacobian_primary(kernel_parameters, 
                                camera, 
                                local_pos, 
                                pixel_id,
                                sp_eps_plus, 
                                isect,
                                _dw_dxg_plus, _dw_ddir, _dw_dorg);

        warp_jacobian_primary(kernel_parameters, 
                                camera, 
                                local_pos, 
                                pixel_id, 
                                sp_eps_minus, 
                                isect,
                                _dw_dxg_minus, _dw_ddir, _dw_dorg);

        d_xy_d_xg_d_barycentrics[i] = (_dw_dxg_plus - _dw_dxg_minus) / (2.0 * epslion);
    }

}

/*
 * This accumulator is intended to be called AFTER path_contribution.
 * It relies on d_bsdf_wos and d_light_wos
 * which are computed in the d_path_contribution pass. 
 */
struct warp_derivatives_accumulator {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];

        //const auto numAuxillaryRays = kernel_parameters.numAuxillaryRays;
        const auto numAuxillaryRays = aux_sample_counts[pixel_id];

        const auto maxAuxillaryRays = kernel_parameters.numAuxillaryRays;
        // Load 'origin' surface point.
        const auto &shading_point = (shading_points != nullptr) ? 
                                    shading_points[pixel_id] :
                                    SurfacePoint{
                                        camera->position,
                                        primary_rays[pixel_id].dir,
                                        Frame{Vector3{0, 0, 0},
                                        Vector3{0, 0, 0},
                                        Vector3{0, 0, 0}},
                                        Vector3{0, 0, 0},
                                        Vector2{0, 0},
                                        Vector2{0, 0}, Vector2{0, 0},
                                        Vector3{0, 0, 0}, Vector3{0, 0, 0}
                                    };

        // Load primary ray data.
        const auto &primary_ray = primary_rays[pixel_id];
        const auto &primary_isect = primary_isects[pixel_id];
        const auto &primary_point = primary_points[pixel_id];

        // Load full path contribution.
        auto path_contrib = path_contribs[pixel_id];

        // Load and transform camera sample to local area.
        Vector2 local_pos;
        Vector2 screen_pos;
        if (camera_samples != nullptr) {
            sample_to_local_pos(*camera,
                                camera_samples[pixel_id],
                                local_pos);

            local_to_screen_pos(*camera,
                                pixel_id,
                                local_pos, screen_pos);
        }

        // Load auxillary ray data.
        std::vector<Ray> v_aux_rays;
        std::vector<Intersection> v_aux_isects;
        std::vector<SurfacePoint> v_aux_points;
        std::vector<AuxSample> v_aux_samples;
        v_aux_rays.assign(&aux_rays[maxAuxillaryRays * pixel_id], &aux_rays[maxAuxillaryRays * pixel_id + (maxAuxillaryRays)]);
        v_aux_isects.assign(&aux_isects[maxAuxillaryRays * pixel_id], &aux_isects[maxAuxillaryRays * pixel_id + (maxAuxillaryRays)]);
        v_aux_points.assign(&aux_points[maxAuxillaryRays * pixel_id], &aux_points[maxAuxillaryRays * pixel_id + (maxAuxillaryRays)]);
        if(camera_samples != nullptr) {
            v_aux_samples.assign(&aux_samples[maxAuxillaryRays * pixel_id], &aux_samples[maxAuxillaryRays * pixel_id + (maxAuxillaryRays)]);
        }

        // Number of auxillary weights from sampled rays.
        const auto sampledAuxWeights = numAuxillaryRays;

        // Sampled auxillary weights + weights from known boundary conditions.
        // (e.g. pixel integral limits)
        const auto totalAuxWeights = numAuxillaryRays;

        // Compute auxillary ray properties.
        std::vector<Real> v_aux_weights(totalAuxWeights, 0);
        std::vector<Vector3> v_aux_div_weights(totalAuxWeights, Vector3(0.0, 0.0, 0.0));

        // TODO : For debugging. remove later.
        std::vector<Real> v_aux_boundary_terms(totalAuxWeights, 0);
        std::vector<Real> v_aux_horizon_terms(totalAuxWeights, 0);

        for(uint i = 0; i < sampledAuxWeights; i++) {

            if(shading_isects != nullptr) {
                // Ignore degerate rays.
                auto shading_isect = shading_isects[pixel_id];
                auto aux_isect = v_aux_isects.at(i);
                if (shading_isect == aux_isect) {
                    continue;
                }
            }

            if (camera_samples != nullptr) {

                auto aux_local_pos = aux_sample_primary_local_pos( kernel_parameters,
                                                            *camera, 
                                                            local_pos, 
                                                            v_aux_samples[i]);
                Real boundary_term = 0.0;
                Real horizon_term = 0.0;

                v_aux_weights.at(i) = warp_weight_primary( kernel_parameters,
                                            *camera,
                                            shapes,
                                            adjacencies,
                                            primary_ray,
                                            v_aux_rays.at(i),
                                            local_pos,
                                            aux_local_pos,
                                            v_aux_isects.at(i),
                                            v_aux_points.at(i),
                                            shading_point,
                                            boundary_term,
                                            horizon_term);

                v_aux_div_weights.at(i) = warp_weight_grad_primary( kernel_parameters,
                                            *camera,
                                            shapes,
                                            adjacencies,
                                            primary_ray,
                                            v_aux_rays.at(i),
                                            local_pos,
                                            aux_local_pos,
                                            v_aux_isects.at(i),
                                            v_aux_points.at(i),
                                            shading_point);

                v_aux_boundary_terms.at(i) = boundary_term;
                v_aux_horizon_terms.at(i) = horizon_term;

            } else {
                Real boundary_term = 0.0;
                Real horizon_term = 0.0;

                v_aux_weights.at(i) = warp_weight( kernel_parameters,
                                            shapes,
                                            adjacencies,
                                            primary_ray,
                                            v_aux_rays.at(i),
                                            v_aux_isects.at(i),
                                            v_aux_points.at(i),
                                            shading_point,
                                            boundary_term,
                                            horizon_term);

                v_aux_div_weights.at(i) = warp_weight_grad( kernel_parameters,
                                            shapes,
                                            adjacencies,
                                            primary_ray,
                                            v_aux_rays.at(i),
                                            v_aux_isects.at(i),
                                            v_aux_points.at(i),
                                            shading_point);

                v_aux_boundary_terms.at(i) = boundary_term;
                v_aux_horizon_terms.at(i) = horizon_term;
            }
        }

        // Compute aux pdfs.
        std::vector<Real> v_aux_pdfs(totalAuxWeights, 0);
        
        for(uint i = 0; i < sampledAuxWeights; i++) {
            if (camera_samples == nullptr)
                // Computes von Mises-Fisher pdf.
                v_aux_pdfs.at(i) = aux_pdf( kernel_parameters, v_aux_rays.at(i), primary_ray );
            else
                // Computes gaussian PDF.
                v_aux_pdfs.at(i) = aux_primary_pdf( kernel_parameters,
                                                   *camera, 
                                                   camera_samples[pixel_id],
                                                   v_aux_samples.at(i));
        }

        Real normalization = 0;
        Vector3 div_normalization(0, 0, 0);
        // Compute the normalizers Z(x) and div.Z(x) (independent of parameter)
        for(uint i = 0; i < totalAuxWeights; i++) {
            normalization += v_aux_weights.at(i) / v_aux_pdfs.at(i);
            div_normalization += v_aux_div_weights.at(i) / v_aux_pdfs.at(i);
        }

        // Compute the inverse normalization. This is the main source of bias.
        // To handle this we provide two modes simple monte carlo (biased) and 
        // RR (Russian Roulette) (unbiased but higher variance)

        std::vector<Real> inv_normalization(totalAuxWeights, 0);
        std::vector<Vector3> q_normalization(totalAuxWeights, Vector3{0.0, 0.0, 0.0});

        if(kernel_parameters.isBasicNormal){
            // Switch to unbiased mode if we use simple normal distribution weights.
            normalization = kernel_parameters.numAuxillaryRays;
            div_normalization = Vector3{0, 0, 0};
        }

        // Russian roulette estimation.
        // Estimates quantites using Russian roulette that are otherwise biased:
        // (i) reciprocal of the weight integral (normalization) 1/\int_{x'}(w(x, x'))
        // (ii) derivative of this reciprocal (\int{x'}grad_w(x, x'))/\int_{x'}(w^2(x, x'))
        if (kernel_parameters.rr_enabled) {
            std::vector<Real> _acc_wt_sum = std::vector<Real>(totalAuxWeights, 0.0);
            std::vector<Vector3> _acc_grad_wt_sum = std::vector<Vector3>(totalAuxWeights, Vector3{0.0, 0.0, 0.0});
            for(int i = 0; i < totalAuxWeights; i++) {
                _acc_wt_sum.at(i) = ((i != 0) ? _acc_wt_sum.at(i - 1) : 0) + (v_aux_weights.at(i) / v_aux_pdfs.at(i));
                _acc_grad_wt_sum.at(i) = ((i != 0) ? _acc_grad_wt_sum.at(i - 1) : Vector3{0.0, 0.0, 0.0}) + (v_aux_div_weights.at(i) / v_aux_pdfs.at(i));
            }

            Real Z = 0.0;
            Vector3 grad_Z = Vector3{0.0, 0.0, 0.0};
            int batchsz = kernel_parameters.batch_size;
            for(int k = totalAuxWeights - 1; k >= 0; k--) {
                // Compute the estiamtor values cumulatively for each batch.
                if (k % batchsz == 0) {
                    // Compute the harmonic difference of the pdfs of DeltaX_i and DeltaX_i+1.
                    Real pdf_i = _aux_sample_sample_counts_pdf(kernel_parameters, k + batchsz);
                    Real pdf_next_i = _aux_sample_sample_counts_pdf(kernel_parameters, k + 2 * batchsz);
                    Real effective_pdf = (pdf_i * pdf_next_i) / (pdf_next_i - pdf_i);

                    // The last element in the sequence occurs only once. The effective pdf is the same as the pdf.
                    if (k == totalAuxWeights - batchsz)
                        effective_pdf = pdf_i;

                    int kidx = k + batchsz - 1;
                    Z = Z + (1.0 / (_acc_wt_sum.at(kidx))) / effective_pdf;
                    grad_Z = grad_Z + ( (_acc_grad_wt_sum.at(kidx)) / ((_acc_wt_sum.at(kidx)) * (_acc_wt_sum.at(kidx))) ) / effective_pdf;

                    for(int offset = 0; offset < batchsz; offset++){
                        inv_normalization.at(k + offset) = Z;
                        q_normalization.at(k + offset) = grad_Z;
                    }
                }
            }

        } else {
            for(int i = 0; i < totalAuxWeights; i++){
                inv_normalization.at(i) =  1.0 / normalization;
                q_normalization.at(i) = div_normalization * (1.0 / normalization) * (1.0 / normalization);
            }
        }

        auto nd = channel_info.num_total_dimensions;
        auto d = channel_info.radiance_dimension;

        // Compute pixel weight w.r.t the current loss function.
        auto df_d_path_contrib = weight *
        Vector3{d_rendered_image[nd * pixel_id + d    ],
                d_rendered_image[nd * pixel_id + d + 1],
                d_rendered_image[nd * pixel_id + d + 2]};

        // Contribution of the control variate to u, v and w coefficients.
        Vector3 dv_d_local_pos(0.0,0.0,0.0);
        Vector3 du_d_local_pos(0.0,0.0,0.0);
        Vector3 dw_d_local_pos(0.0,0.0,0.0);

        // Compute control variates for denoising (if enabled)
        // This section is only for variance reduction and does not affect the bias of the esimator
        // The current implementation is only for primary rays.
        // TODO: Move to its own block.
        if(enable_control_variates 
            && camera_samples != nullptr
            && primary_isect.valid() ) {
            // Compute the A matrix such that w_i^T A w_i is the control variate.

            // We need to find the derivative of the coefficients alpha, beta and gamma w.r.t omega

            // Grab the primary isect and it's vertices.
            auto vidxs = get_indices(shapes[primary_isect.shape_id], primary_isect.tri_id);

            TVector3<double> p0 = TVector3<double>(v3(&shapes[primary_isect.shape_id].vertices[3 * vidxs[0]]));
            TVector3<double> p1 = TVector3<double>(v3(&shapes[primary_isect.shape_id].vertices[3 * vidxs[1]]));
            TVector3<double> p2 = TVector3<double>(v3(&shapes[primary_isect.shape_id].vertices[3 * vidxs[2]]));

            TVector3<double> d_v0{Real(0), Real(0), Real(0)};
            TVector3<double> d_v1{Real(0), Real(0), Real(0)};
            TVector3<double> d_v2{Real(0), Real(0), Real(0)};
            DTRay<double> d_ray_du{
                TVector3<double>{Real(0), Real(0), Real(0)},
                TVector3<double>{Real(0), Real(0), Real(0)}
            };

            TRayDifferential<double> zero_d_ray_differential{
                    TVector3<double>{Real(0), Real(0), Real(0)}, TVector3<double>{Real(0), Real(0), Real(0)},
                  TVector3<double>{Real(0), Real(0), Real(0)}, TVector3<double>{Real(0), Real(0), Real(0)}};

            // Compute the first term in A, i.e. the change in the local velocity if the jacobian remains constant

            d_intersect(p0, p1, p2, 
                    primary_ray, zero_d_ray_differential, 
                    TVector3<double>{Real(1.0), Real(0), Real(0)}, 
                    TVector2<double>{Real(0), Real(0)},
                    TVector2<double>{Real(0), Real(0)},
                    TVector2<double>{Real(0), Real(0)},
                    d_v1, d_v1, d_v2,
                    d_ray_du,
                    zero_d_ray_differential
                );

            DTRay<double> d_ray_dv{
                TVector3<double>{Real(0), Real(0), Real(0)},
                TVector3<double>{Real(0), Real(0), Real(0)}
            };
            d_intersect(p0, p1, p2, 
                    primary_ray, zero_d_ray_differential, 
                    TVector3<double>{Real(0), Real(1.0), Real(0)}, 
                    TVector2<double>{Real(0), Real(0)},
                    TVector2<double>{Real(0), Real(0)},
                    TVector2<double>{Real(0), Real(0)},
                    d_v1, d_v1, d_v2,
                    d_ray_dv,
                    zero_d_ray_differential
                );

            DCamera d_camera(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
            Vector2 du_d_screen_pos;
            d_sample_primary_ray(*camera, screen_pos, d_ray_du, d_camera, &du_d_screen_pos);
            Vector2 dv_d_screen_pos;
            d_sample_primary_ray(*camera, screen_pos, d_ray_dv, d_camera, &dv_d_screen_pos);

            dv_d_local_pos = expand(dv_d_screen_pos * (Real(1.0) / Vector2{camera->width, camera->height}));
            du_d_local_pos = expand(du_d_screen_pos * (Real(1.0) / Vector2{camera->width, camera->height}));
            dw_d_local_pos = - (dv_d_local_pos + du_d_local_pos);

            Matrix3x3 primary_dw_dxg = Matrix3x3::zeros();
            Matrix3x3 primary_dw_ddir = Matrix3x3::zeros();
            Matrix3x3 primary_dw_dorg = Matrix3x3::zeros();
            warp_jacobian_primary(kernel_parameters, 
                                      *camera,
                                      local_pos,
                                      pixel_id,
                                      primary_point,
                                      primary_isect,
                                      primary_dw_dxg,
                                      primary_dw_ddir,
                                      primary_dw_dorg);

            // Compute the second term, i.e change in the jacobian if the constant velocity remains constant.

            Matrix3x3 d_jacobian_d_barycentrics[3];
            d_jacobian_d_barycentrics[0] = Matrix3x3::zeros();
            d_jacobian_d_barycentrics[1] = Matrix3x3::zeros();
            d_jacobian_d_barycentrics[2] = Matrix3x3::zeros();

            Vector3 barycentrics{0.f, 0.f, 0.f}; // used to store (1 - (u + v), u, v)

            warp_gradient_tensor(kernel_parameters, 
                        shapes,
                        *camera,
                        local_pos,
                        pixel_id,
                        primary_point,
                        primary_isect,
                        primary_ray,
                        1e-2,
                        d_jacobian_d_barycentrics,
                        barycentrics);

            // Compute intermediate matrix.
            auto t2_A_x = //outer_product(d_jacobian_d_barycentrics[0].col(0), dw_d_local_pos) + 
                          outer_product(d_jacobian_d_barycentrics[1].col(0), du_d_local_pos) + 
                          outer_product(d_jacobian_d_barycentrics[2].col(0), dv_d_local_pos);

            auto t2_A_y = //outer_product(d_jacobian_d_barycentrics[0].col(1), dw_d_local_pos) + 
                          outer_product(d_jacobian_d_barycentrics[1].col(1), du_d_local_pos) + 
                          outer_product(d_jacobian_d_barycentrics[2].col(1), dv_d_local_pos);

            auto t2_A_z = //outer_product(d_jacobian_d_barycentrics[0].col(2), dw_d_local_pos) + 
                          outer_product(d_jacobian_d_barycentrics[1].col(2), du_d_local_pos) + 
                          outer_product(d_jacobian_d_barycentrics[2].col(2), dv_d_local_pos);

            // Scatter the contributions to the three vertices of the triangle.
            auto A_u_x = outer_product(primary_dw_dxg.col(0), du_d_local_pos) + t2_A_x * barycentrics[1];
            auto A_u_y = outer_product(primary_dw_dxg.col(1), du_d_local_pos) + t2_A_y * barycentrics[1];
            auto A_u_z = outer_product(primary_dw_dxg.col(2), du_d_local_pos) + t2_A_z * barycentrics[1];

            auto A_v_x = outer_product(primary_dw_dxg.col(0), dv_d_local_pos) + t2_A_x * barycentrics[2];
            auto A_v_y = outer_product(primary_dw_dxg.col(1), dv_d_local_pos) + t2_A_y * barycentrics[2];
            auto A_v_z = outer_product(primary_dw_dxg.col(2), dv_d_local_pos) + t2_A_z * barycentrics[2];

            auto A_w_x = outer_product(primary_dw_dxg.col(0), dw_d_local_pos) + t2_A_x * barycentrics[0];
            auto A_w_y = outer_product(primary_dw_dxg.col(1), dw_d_local_pos) + t2_A_y * barycentrics[0];
            auto A_w_z = outer_product(primary_dw_dxg.col(2), dw_d_local_pos) + t2_A_z * barycentrics[0];

            const auto sigma = kernel_parameters.auxPrimaryGaussianStddev;
            std::vector<Vector3> d_primary_v_p(3, Vector3{0.0, 0.0, 0.0});
            for(uint i = 0; i < sampledAuxWeights; i++) {
                auto aux_local_pos = aux_sample_primary_local_pos(kernel_parameters,
                                    *camera, 
                                    local_pos, 
                                    v_aux_samples[i]);

                // TODO: Figure this sign out at some point.
                Vector3 w_i = -expand(aux_local_pos - local_pos) / sigma;

                Vector3 control_p0 = Vector3{
                        dot(w_i * A_w_x, w_i),
                        dot(w_i * A_w_y, w_i),
                        dot(w_i * A_w_z, w_i)
                };
                Vector3 control_mean_p0 = Vector3{ trace(A_w_x), trace(A_w_y), trace(A_w_z) };

                Vector3 control_p1 = Vector3{
                        dot(w_i * A_u_x, w_i),
                        dot(w_i * A_u_y, w_i),
                        dot(w_i * A_u_z, w_i)
                };
                Vector3 control_mean_p1 = Vector3{ trace(A_u_x), trace(A_u_y), trace(A_u_z) };

                Vector3 control_p2 = Vector3{
                        dot(w_i * A_v_x, w_i),
                        dot(w_i * A_v_y, w_i),
                        dot(w_i * A_v_z, w_i)
                };
                Vector3 control_mean_p2 = Vector3{ trace(A_v_x), trace(A_v_y), trace(A_v_z) };

                d_primary_v_p[0] += (control_p0 - control_mean_p0) * 
                                    inv_normalization.at(sampledAuxWeights - 1) * 
                                    Real(M_PI) * 2 * sigma * sigma * 
                                    sum(df_d_path_contrib * path_contrib);

                d_primary_v_p[1] += (control_p1 - control_mean_p1) * 
                                    inv_normalization.at(sampledAuxWeights - 1) * 
                                    Real(M_PI) * 2 * sigma * sigma * 
                                    sum(df_d_path_contrib * path_contrib);

                d_primary_v_p[2] += (control_p2 - control_mean_p2) * 
                                    inv_normalization.at(sampledAuxWeights - 1) * 
                                    Real(M_PI) * 2 * sigma * sigma * 
                                    sum(df_d_path_contrib * path_contrib);
            }

            // Backpropagate into the shape.
            atomic_add(&d_shapes[primary_isect.shape_id].vertices[3 * vidxs[0]],
                    -d_primary_v_p[0]);
            atomic_add(&d_shapes[primary_isect.shape_id].vertices[3 * vidxs[1]],
                    -d_primary_v_p[1]);
            atomic_add(&d_shapes[primary_isect.shape_id].vertices[3 * vidxs[2]],
                    -d_primary_v_p[2]);
            
            /* TODO: DEBUG IMAGE */
            if (primary_isect.shape_id == SHAPE_SELECT && debug_image != nullptr)
                debug_image[pixel_id] += -(d_primary_v_p[0][DIM_SELECT] +
                                           d_primary_v_p[1][DIM_SELECT] +
                                           d_primary_v_p[2][DIM_SELECT]);

        }

        std::vector<SurfacePoint> d_aux_points(sampledAuxWeights, SurfacePoint());

        auto primary_vec = primary_point.position - primary_ray.org;
        auto dist = length(primary_vec);

        // This is the derivative of the contribution w.r.t 
        // spatial domain variable.
        // The spatial domain variable for secondary bounces is the space of
        // outgoing directions in the hemisphere 'wo'
        // The spatial domain variable for primary bounce is the 2D square
        // covering the pixel.
        auto df_dw = (d_wos != nullptr) ?
                        -d_wos[pixel_id] : // Secondary bounce. Domain is unit 3-vector wo.
                        Vector3 {          // First bounce. Domain is unit square.
                            df_d_local_pos[pixel_id].x,
                            df_d_local_pos[pixel_id].y,
                            0.0
                        };

        // Slightly hacky.
        if (primary_ray.tmax <= 0 && d_wos != nullptr) {
            // Occluded NEE ray, the gradient must be auto set to 0.
            df_dw = Vector3{0,0,0};
        }

        auto raw_df_dw = df_dw;

        // -----------------
        /*
            Accumulate control variate metrics.
            These are used by accumulate_control_variates() to compute its contribution.
        */
        auto w_i = expand(local_pos);
        if(control_sample_covariance != nullptr && camera_samples != nullptr) {
            control_sample_covariance[pixel_id] = 
                control_sample_covariance[pixel_id] + outer_product(w_i, w_i);
            control_mean_contrib[pixel_id] = 
                control_mean_contrib[pixel_id] + sum(df_d_path_contrib * path_contrib);
            control_mean_grad_contrib[pixel_id] = 
                control_mean_grad_contrib[pixel_id] + df_dw;
        }
        // ----------------------------
        
        Vector3 kernel_score(0,0,0);
        Real filter_weight;
        Vector3 F_gradK(0, 0, 0);
        if(camera_samples != nullptr) {
            // For primary samples..
            // An additional term shows up in the derivative
            // of contrib w.r.t local pos, due to the recon
            // filter.
            Vector2 local_pos;
            sample_to_local_pos(*camera, camera_samples[pixel_id], local_pos);

            Vector2 d_filter_d_local_pos;

            screen_filter_grad(*camera,
                               pixel_id,
                               local_pos,
                               d_filter_d_local_pos,
                               filter_weight);

            // Add the filter gradient score function to the computation.
            if (filter_weight != 0) {
                kernel_score =
                     (Vector3{d_filter_d_local_pos[0], d_filter_d_local_pos[1], 0.0}
                       / filter_weight);
                F_gradK = sum(path_contrib *
                     df_d_path_contrib) *
                     kernel_score;
            } else {
                kernel_score = Vector3(0.f, 0.f, 0.f);
            }
        }

        // Compute the warp contribution to the derivative of the shape parameters.
        for(uint i = 0; i < sampledAuxWeights; i++) {
            const auto &ray = v_aux_rays[i];
            if(dot(shading_point.geom_normal, ray.dir) * dot(shading_point.geom_normal, primary_ray.dir) <= 1e-4) {
                // These rays are handled in a subsequent loop.
                continue;
            }

            // Special case: 
            // Check if the intersections at either end of the ray
            // are the same.
            if(shading_isects != nullptr) {
                auto shading_isect = shading_isects[pixel_id];
                auto aux_isect = v_aux_isects.at(i);
                if (shading_isect == aux_isect) {
                    continue;
                }
            }

            Matrix3x3 dw_dxg,  // Jacobian of domain coords w.r.t intersection point.
                      dw_ddir, // Jacobian of domain coords w.r.t ray direction
                      dw_dorg; // Jacobian of domain coords w.r.t ray origin

            if (camera_samples == nullptr) {
                // Secondary sampling.
                warp_jacobian(
                            kernel_parameters,
                            shading_point,
                            v_aux_points.at(i),
                            v_aux_isects.at(i),
                            dw_dxg);

                // These aren't necessary for secondary sampling since
                // all of our parameters only affect the contribution
                // function AFTER the computation of intersection xg.
                // This means dw_dxg is sufficient.
                dw_ddir = Matrix3x3::identity();
                dw_dorg = Matrix3x3::identity();
            } else {

                // We compute jacobian matrices for both intersections and
                // dir and org, which occur earlier in the pipeline. 
                // This is to efficiently compute the warp contrib for 
                // camera parameters which affect xg through dir and org.
                Vector2 local_pos;
                sample_to_local_pos(*camera, camera_samples[pixel_id], local_pos);
                warp_jacobian_primary(kernel_parameters, 
                                      *camera,
                                      local_pos,
                                      pixel_id,
                                      v_aux_points.at(i),
                                      v_aux_isects.at(i),
                                      dw_dxg,
                                      dw_ddir,
                                      dw_dorg);
            }

            // Compute warp field contribution from this auxillary ray.
            // 3x3 (3D vector field in domain, 3 spatial parameters)
            auto vMultiplier = (v_aux_weights.at(i) / v_aux_pdfs.at(i)) * inv_normalization.at(i);

            auto V_xg =  dw_dxg  * vMultiplier;
            auto V_dir = dw_ddir * vMultiplier;
            auto V_org = dw_dorg * vMultiplier;

            // Compute contribution to the divergence of warp field.
            // 3x1 (scalar field in domain, 3 spatial parameters)
            auto divVMultiplier = (v_aux_div_weights.at(i) / v_aux_pdfs.at(i)) * inv_normalization.at(i) -
                                  (v_aux_weights.at(i) / v_aux_pdfs.at(i)) * q_normalization.at(i);

            auto divV_xg  = divVMultiplier * dw_dxg;
            auto divV_dir = divVMultiplier * dw_ddir;
            auto divV_org = divVMultiplier * dw_dorg;

            // Gradient w.r.t intersection point (world space).
            auto gradF_dot_V_xg = df_dw * V_xg + F_gradK * V_xg; // gradF.K.V + F.gradK.V
            auto F_mul_div_V_xg = sum(df_d_path_contrib * path_contrib) * divV_xg; // F.K.divV
            
            // Gradient w.r.t ray direction.
            auto gradF_dot_V_dir = df_dw * V_dir + F_gradK * V_dir; // gradF.K.V + F.gradK.V
            auto F_mul_div_V_dir = sum(df_d_path_contrib * path_contrib) * divV_dir; // F.K.divV

            // Gradient w.r.t ray origin.
            // NOTE: Incorrect for orthographic.
            auto gradF_dot_V_org = df_dw * V_org + F_gradK * V_org; // gradF.K.V + F.gradK.V
            auto F_mul_div_V_org = sum(df_d_path_contrib * path_contrib) * divV_org; // F.K.divV

            // Compute final gradients.
            auto grad_xg = gradF_dot_V_xg + F_mul_div_V_xg;
            auto grad_org = gradF_dot_V_org + F_mul_div_V_org;
            auto grad_dir = gradF_dot_V_dir + F_mul_div_V_dir;
            // ---

            if(camera_samples != nullptr) {
                Vector2 screen_pos;
                local_to_screen_pos(*camera, pixel_id, 
                                    local_pos, screen_pos);

                Vector2 d_screen_pos{0, 0}; // Intermediate value..

                // Pass the derivatives w.r.t ray onto the camera parameters.
                d_sample_primary_ray(*camera, 
                                    screen_pos,
                                    DRay(-grad_xg, // NOTE: Incorrect for orthographic.
                                         -grad_dir),
                                    *d_camera,
                                    &d_screen_pos);

                /* TODO: Debuging code. Remove.
                if(pixel_id == 125222) {
                    auto aux_isect = v_aux_isects.at(i);
                    std::cout << "screen_pos: " << screen_pos.x << ", " << screen_pos.y << std::endl;
                    std::cout << "primary dir: " << primary_ray.dir.x << ", " << primary_ray.dir.y << ", " << primary_ray.dir.z << std::endl;
                    std::cout << "primary isect: " << primary_isect.shape_id << ", " << primary_isect.tri_id << std::endl;
                    std::cout << "aux-ray-id: " << i << "/" << sampledAuxWeights << std::endl;
                    std::cout << "aux-ray dir: " << v_aux_rays.at(i).dir.x << ", " << v_aux_rays.at(i).dir.y << ", " << v_aux_rays.at(i).dir.z << std::endl;
                    std::cout << "isect: " << aux_isect.shape_id << ", " << aux_isect.tri_id << std::endl;
                    std::cout << "grad_dir: " << grad_dir.x << ", " << grad_dir.y << ", " << grad_dir.z << std::endl;
                    std::cout << "grad_org: " << grad_org.x << ", " << grad_org.y << ", " << grad_org.z << std::endl;
                    std::cout << "d_screen_pos: " << d_screen_pos.x << ", " << d_screen_pos.y << std::endl;
                    auto d = df_d_path_contrib * path_contrib;
                    std::cout << "grad_f: " << d.x << ", " << d.y << ", " << d.z << std::endl;
                    std::cout << "df_dw: " << df_dw.x << ", " << df_dw.y << ", " << df_dw.z << std::endl;
                    std::cout << "F_gradK: " << F_gradK.x << ", " << F_gradK.y << ", " << F_gradK.z << std::endl;
                    std::cout << "multiplier: " << vMultiplier << std::endl;
                    std::cout << "aux_boundary: " << v_aux_boundary_terms.at(i) << std::endl;
                    std::cout << "dxy_d_dir: " << dw_ddir(0, 0) << ", " << dw_ddir(0, 1) << ", " << dw_ddir(0, 2) << std::endl
                                               << dw_ddir(1, 0) << ", " << dw_ddir(1, 1) << ", " << dw_ddir(1, 2) << std::endl
                                               << dw_ddir(2, 0) << ", " << dw_ddir(2, 1) << ", " << dw_ddir(2, 2) << std::endl;
                }
                */

                if (screen_gradient_image != nullptr) {
                    screen_gradient_image[2 * pixel_id + 0] += d_screen_pos.x;
                    screen_gradient_image[2 * pixel_id + 1] += d_screen_pos.y;
                }
            }

            if(d_shading_points != nullptr) {
                d_shading_points[pixel_id].position -= (F_mul_div_V_xg + gradF_dot_V_xg);
            }

            auto &t_aux_isect = v_aux_isects.at(i);
            if(t_aux_isect.valid() && 
                ((camera_samples != nullptr) ||
                (d_shading_points != nullptr))
            ) {
                Vector3 d_aux_v_p[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                Vector3 ignore_d_aux_v_n[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                Vector2 ignore_d_aux_v_uv[3] = {Vector2{0, 0}, Vector2{0, 0}};

                // TODO: Ray differential is not handled. Figure out.
                RayDifferential zero_d_ray_differential{
                    Vector3{0, 0, 0}, Vector3{0, 0, 0},
                  Vector3{0, 0, 0}, Vector3{0, 0, 0}};

                // Instead of using d_intersect_shape (which differentiates only 't' directly),
                // we're going about this in a slightly more straightforward manner.
                // This still works because the spatial derivative is distinct from the
                // 'temporal' (w.r.t parameter) derivative.
                auto aux_tri_index = get_indices(shapes[t_aux_isect.shape_id], t_aux_isect.tri_id);
                auto v0 = Vector3{get_vertex(shapes[t_aux_isect.shape_id], aux_tri_index[0])};
                auto v1 = Vector3{get_vertex(shapes[t_aux_isect.shape_id], aux_tri_index[1])};
                auto v2 = Vector3{get_vertex(shapes[t_aux_isect.shape_id], aux_tri_index[2])};
                auto u_dxy = Vector2{0,0};
                auto v_dxy = Vector2{0,0};
                auto t_dxy = Vector2{0,0};
                auto uvt = intersect(v0, v1, v2, ray, zero_d_ray_differential, u_dxy, v_dxy, t_dxy);

                // Propagate the xg derivative through to points manually.
                // For this we assume (falsely, but without consequence) that
                // the intersection was originally computed using u * v0 + v * v1 + (1 - (u+v)) * v2
                // instead of o + r*t.
                // This keeps the original point, but changes it's dependencies to what we actually need.
                d_aux_v_p[0] = (grad_xg) * (1 - (uvt[0] + uvt[1]));
                d_aux_v_p[1] = (grad_xg) * uvt[0];
                d_aux_v_p[2] = (grad_xg) * uvt[1];

                /* TODO: DEBUG IMAGE */
                if (t_aux_isect.shape_id == SHAPE_SELECT && debug_image != nullptr)
                    debug_image[pixel_id] += grad_xg[DIM_SELECT];

                atomic_add(&d_shapes[t_aux_isect.shape_id].vertices[3 * aux_tri_index[0]],
                    d_aux_v_p[0]);
                atomic_add(&d_shapes[t_aux_isect.shape_id].vertices[3 * aux_tri_index[1]],
                    d_aux_v_p[1]);
                atomic_add(&d_shapes[t_aux_isect.shape_id].vertices[3 * aux_tri_index[2]],
                    d_aux_v_p[2]);
            }
        }

    }

    // Global data
    const Shape *shapes;
    const int *active_pixels;

    // Common data (primary + auxillary)
    const SurfacePoint *shading_points;
    const Intersection *shading_isects;
    
    // Primary ray data.
    const Ray *primary_rays;
    const Intersection *primary_isects;
    const SurfacePoint *primary_points;

    // Auxillary ray(s) data.
    const uint *aux_sample_counts;
    const Ray *aux_rays;
    const Intersection *aux_isects;
    const SurfacePoint *aux_points;
    const AuxSample *aux_samples; // Only for primary bounce.
    
    // Data required for computation of backward derivatives.
    const Vector3 *path_contribs; // Contrib for the current path. don't confuse with integral.

    // Derivative inputs.
    //SurfacePoint *d_primary_points;
    const Vector3 *d_wos;
    const Real weight;
    const ChannelInfo channel_info;
    const KernelParameters kernel_parameters; // Info for asymptotic sampling and weighting.
    const float* d_rendered_image;

    // Enables control variates for the computation of the warp divergence.
    const bool enable_control_variates;

    // Derivative outputs.
    SurfacePoint *d_shading_points; // If point exists.
    DShape *d_shapes;

    CameraSample *camera_samples;
    const Vector2* df_d_local_pos;
    DCamera      *d_camera; // If first bounce.
    const Camera       *camera;

    // Shape edge-adjacency information, can be used to quickly
    // compute if an edge is silhouette or not
    ShapeAdjacency* adjacencies;

    float* debug_image;
    float* screen_gradient_image;

    // Accumulate some population statistics to compute a control
    // variate that sharply reduces variance in case of linear variation.
    Vector3* control_mean_grad_contrib;
    Real* control_mean_contrib;
    Matrix3x3* control_sample_covariance;

};

void accumulate_warp_derivatives(const Scene &scene,
                                           const BufferView<int> &active_pixels,
                                           const BufferView<SurfacePoint> &shading_points,
                                           const BufferView<Intersection> &shading_isects,
                                           const BufferView<Ray> &primary_rays,
                                           const BufferView<Intersection> &primary_isects,
                                           const BufferView<SurfacePoint> &primary_points,
                                           const BufferView<uint> &aux_sample_counts,
                                           const BufferView<Ray> &aux_rays,
                                           const BufferView<Intersection> &aux_isects,
                                           const BufferView<SurfacePoint> &aux_points,
                                           const BufferView<AuxSample> &aux_samples,
                                           const BufferView<Vector3> &path_contribs,
                                           const BufferView<Vector3> &d_wos,
                                           const Real weight,
                                           const ChannelInfo channel_info,
                                           const KernelParameters &kernel_parameters,
                                           const float *d_rendered_image,
                                           const bool enable_control_variates,
                                           BufferView<SurfacePoint> d_shading_points,
                                           BufferView<DShape> d_shapes,
                                           BufferView<CameraSample> camera_samples,
                                           BufferView<Vector2> df_d_local_pos,
                                           ShapeAdjacency* adjacency,
                                           DScene* d_scene,
                                           float* debug_image,
                                           float* screen_gradient_image,
                                           BufferView<Vector3> control_mean_grad_contrib,
                                           BufferView<Real> control_mean_contrib,
                                           BufferView<Matrix3x3> control_sample_covariance){
    parallel_for(warp_derivatives_accumulator{
        scene.shapes.data,
        active_pixels.begin(),
        shading_points.begin(),
        shading_isects.begin(),
        primary_rays.begin(),
        primary_isects.begin(),
        primary_points.begin(),
        aux_sample_counts.begin(),
        aux_rays.begin(),
        aux_isects.begin(),
        aux_points.begin(),
        aux_samples.begin(),
        path_contribs.begin(),
        d_wos.begin(),
        weight,
        channel_info,
        kernel_parameters,
        d_rendered_image,
        enable_control_variates,
        d_shading_points.begin(),
        d_shapes.begin(),
        camera_samples.begin(),
        df_d_local_pos.begin(),
        &d_scene->camera,
        &scene.camera,
        adjacency,
        debug_image,
        screen_gradient_image,
        control_mean_grad_contrib.begin(),
        control_mean_contrib.begin(),
        control_sample_covariance.begin(),
    }, active_pixels.size(), scene.use_gpu);
}

struct control_variates_accumulator{

    DEVICE void operator()(int idx) { 
        const int pixel_id = active_pixels[idx];

        auto control_point = control_points[pixel_id];
        auto control_isect = control_isects[pixel_id];
        auto control_ray = control_rays[pixel_id];

        if (!control_isect.valid()) {
            // No intersection. PCV is 0.
            return;
        }

        // Grab the primary isect and it's vertices.
        auto vidxs = get_indices(scene.shapes[control_isect.shape_id], control_isect.tri_id);

        TVector3<double> p0 = TVector3<double>(v3(&scene.shapes[control_isect.shape_id].vertices[3 * vidxs[0]]));
        TVector3<double> p1 = TVector3<double>(v3(&scene.shapes[control_isect.shape_id].vertices[3 * vidxs[1]]));
        TVector3<double> p2 = TVector3<double>(v3(&scene.shapes[control_isect.shape_id].vertices[3 * vidxs[2]]));

        TVector3<double> d_v0{Real(0), Real(0), Real(0)};
        TVector3<double> d_v1{Real(0), Real(0), Real(0)};
        TVector3<double> d_v2{Real(0), Real(0), Real(0)};
        DTRay<double> d_ray_du{
            TVector3<double>{Real(0), Real(0), Real(0)},
            TVector3<double>{Real(0), Real(0), Real(0)}
        };

        TRayDifferential<double> zero_d_ray_differential{
                TVector3<double>{Real(0), Real(0), Real(0)}, TVector3<double>{Real(0), Real(0), Real(0)},
                TVector3<double>{Real(0), Real(0), Real(0)}, TVector3<double>{Real(0), Real(0), Real(0)}};

        // Compute the first term in A, i.e. the change in the local velocity if the jacobian remains constant.

        d_intersect(p0, p1, p2, 
                control_ray, zero_d_ray_differential, 
                TVector3<double>{Real(1.0), Real(0), Real(0)}, 
                TVector2<double>{Real(0), Real(0)},
                TVector2<double>{Real(0), Real(0)},
                TVector2<double>{Real(0), Real(0)},
                d_v1, d_v1, d_v2,
                d_ray_du,
                zero_d_ray_differential
            );

        DTRay<double> d_ray_dv{
            TVector3<double>{Real(0), Real(0), Real(0)},
            TVector3<double>{Real(0), Real(0), Real(0)}
        };
        d_intersect(p0, p1, p2, 
                control_ray, zero_d_ray_differential, 
                TVector3<double>{Real(0), Real(1.0), Real(0)}, 
                TVector2<double>{Real(0), Real(0)},
                TVector2<double>{Real(0), Real(0)},
                TVector2<double>{Real(0), Real(0)},
                d_v1, d_v1, d_v2,
                d_ray_dv,
                zero_d_ray_differential
            );

        Vector2 local_pos;
        Vector2 screen_pos;
        sample_to_local_pos(scene.camera, control_samples[pixel_id], local_pos);
        local_to_screen_pos(scene.camera,
                            pixel_id,
                            local_pos, screen_pos);

        DCamera d_camera(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        Vector2 du_d_screen_pos;
        d_sample_primary_ray(scene.camera, screen_pos, d_ray_du, d_camera, &du_d_screen_pos);
        Vector2 dv_d_screen_pos;
        d_sample_primary_ray(scene.camera, screen_pos, d_ray_dv, d_camera, &dv_d_screen_pos);

        
        // TODO: rewrite using d_local_to_screen_pos
        Vector2 _du_d_local_pos;
        d_local_to_screen_pos(scene.camera, pixel_id, du_d_screen_pos, _du_d_local_pos);
        Vector2 _dv_d_local_pos;
        d_local_to_screen_pos(scene.camera, pixel_id, dv_d_screen_pos, _dv_d_local_pos);

        auto dv_d_local_pos = expand(_dv_d_local_pos);
        auto du_d_local_pos = expand(_du_d_local_pos);
        auto dw_d_local_pos = - (dv_d_local_pos + du_d_local_pos);

        Matrix3x3 dw_dxg = Matrix3x3::zeros();

        Matrix3x3 dw_dxg_0 = Matrix3x3::zeros();
        Matrix3x3 dw_dxg_1 = Matrix3x3::zeros();
        Matrix3x3 dw_dxg_2 = Matrix3x3::zeros();

        Matrix3x3 dw_ddir = Matrix3x3::zeros();
        Matrix3x3 dw_dorg = Matrix3x3::zeros();
        warp_jacobian_primary(kernel_parameters, 
                                    scene.camera,
                                    local_pos,
                                    pixel_id,
                                    control_point,
                                    control_isect,
                                    dw_dxg,
                                    dw_ddir,
                                    dw_dorg);
        
        SurfacePoint control_point_p0 = control_point;
        SurfacePoint control_point_p1 = control_point;
        SurfacePoint control_point_p2 = control_point;

        control_point_p0.position = p0;
        control_point_p1.position = p1;
        control_point_p2.position = p2;
        warp_jacobian_primary(kernel_parameters, 
                                    scene.camera,
                                    local_pos,
                                    pixel_id,
                                    control_point_p0,
                                    control_isect,
                                    dw_dxg_0,
                                    dw_ddir,
                                    dw_dorg);

        warp_jacobian_primary(kernel_parameters, 
                                    scene.camera,
                                    local_pos,
                                    pixel_id,
                                    control_point_p1,
                                    control_isect,
                                    dw_dxg_1,
                                    dw_ddir,
                                    dw_dorg);

        warp_jacobian_primary(kernel_parameters, 
                                    scene.camera,
                                    local_pos,
                                    pixel_id,
                                    control_point_p2,
                                    control_isect,
                                    dw_dxg_2,
                                    dw_ddir,
                                    dw_dorg);

        // Compute the second term, i.e change in the jacobian if the constant velocity remains constant.

        Matrix3x3 d_jacobian_d_barycentrics[3];
        d_jacobian_d_barycentrics[0] = Matrix3x3::zeros();
        d_jacobian_d_barycentrics[1] = Matrix3x3::zeros();
        d_jacobian_d_barycentrics[2] = Matrix3x3::zeros();

        Vector3 barycentrics{0.f, 0.f, 0.f}; // used to store (1 - (u + v), u, v)

        warp_gradient_tensor(kernel_parameters, 
                scene.shapes.begin(),
                scene.camera,
                local_pos,
                pixel_id,
                control_point,
                control_isect,
                control_ray,
                1e-2,
                d_jacobian_d_barycentrics,
                barycentrics);

        Vector3 control_grad_contrib = control_mean_grad_contrib[pixel_id];

        // Compute intermediate matrix.
        auto t2_A_x = //outer_product(d_jacobian_d_barycentrics[0].col(0), dw_d_local_pos) + 
                        outer_product(d_jacobian_d_barycentrics[1].col(0), du_d_local_pos) + 
                        outer_product(d_jacobian_d_barycentrics[2].col(0), dv_d_local_pos);

        auto t2_A_y = //outer_product(d_jacobian_d_barycentrics[0].col(1), dw_d_local_pos) + 
                        outer_product(d_jacobian_d_barycentrics[1].col(1), du_d_local_pos) + 
                        outer_product(d_jacobian_d_barycentrics[2].col(1), dv_d_local_pos);

        auto t2_A_z = //outer_product(d_jacobian_d_barycentrics[0].col(2), dw_d_local_pos) + 
                        outer_product(d_jacobian_d_barycentrics[1].col(2), du_d_local_pos) + 
                        outer_product(d_jacobian_d_barycentrics[2].col(2), dv_d_local_pos);

        // Scatter the contributions to the three vertices of the triangle.
        auto B_u_x = outer_product(dw_dxg_1.col(0), du_d_local_pos);
        auto B_u_y = outer_product(dw_dxg_1.col(1), du_d_local_pos);
        auto B_u_z = outer_product(dw_dxg_1.col(2), du_d_local_pos);

        auto B_v_x = outer_product(dw_dxg_2.col(0), dv_d_local_pos);
        auto B_v_y = outer_product(dw_dxg_2.col(1), dv_d_local_pos);
        auto B_v_z = outer_product(dw_dxg_2.col(2), dv_d_local_pos);

        auto B_w_x = outer_product(dw_dxg_0.col(0), dw_d_local_pos);
        auto B_w_y = outer_product(dw_dxg_0.col(1), dw_d_local_pos);
        auto B_w_z = outer_product(dw_dxg_0.col(2), dw_d_local_pos);

        auto mean_contrib = control_mean_contrib[pixel_id];
        auto A_u_x = outer_product(control_grad_contrib, dw_dxg.col(0)) * -barycentrics[1] + B_u_x * mean_contrib;
        auto A_u_y = outer_product(control_grad_contrib, dw_dxg.col(1)) * -barycentrics[1] + B_u_y * mean_contrib;
        auto A_u_z = outer_product(control_grad_contrib, dw_dxg.col(2)) * -barycentrics[1] + B_u_z * mean_contrib;

        auto A_v_x = outer_product(control_grad_contrib, dw_dxg.col(0)) * -barycentrics[2] + B_v_x * mean_contrib;
        auto A_v_y = outer_product(control_grad_contrib, dw_dxg.col(1)) * -barycentrics[2] + B_v_y * mean_contrib;
        auto A_v_z = outer_product(control_grad_contrib, dw_dxg.col(2)) * -barycentrics[2] + B_v_z * mean_contrib;

        auto A_w_x = outer_product(control_grad_contrib, dw_dxg.col(0)) * -barycentrics[0] + B_w_x * mean_contrib;
        auto A_w_y = outer_product(control_grad_contrib, dw_dxg.col(1)) * -barycentrics[0] + B_w_y * mean_contrib;
        auto A_w_z = outer_product(control_grad_contrib, dw_dxg.col(2)) * -barycentrics[0] + B_w_z * mean_contrib;
        std::vector<Vector3> d_control_v_p(3, Vector3{0,0,0});
        Matrix3x3 sample_covariance = control_sample_covariance[pixel_id];
        constexpr auto sigma = sqrt(SCREEN_FILTER_VARIANCE);
        sample_covariance = (sample_covariance / (sigma * sigma)) * weight;

        Vector3 control_variate_p0 = Vector3{
            collapse(hadamard_product(A_w_x, sample_covariance)),
            collapse(hadamard_product(A_w_y, sample_covariance)),
            collapse(hadamard_product(A_w_z, sample_covariance))
        };
        Vector3 control_mean_p0 = Vector3{
            trace(A_w_x), trace(A_w_y), trace(A_w_z)
        };

        Vector3 control_variate_p1 = Vector3{
            collapse(hadamard_product(A_u_x, sample_covariance)),
            collapse(hadamard_product(A_u_y, sample_covariance)),
            collapse(hadamard_product(A_u_z, sample_covariance))
        };
        Vector3 control_mean_p1 = Vector3{
            trace(A_u_x), trace(A_u_y), trace(A_u_z)
        };

        Vector3 control_variate_p2 = Vector3{
            collapse(hadamard_product(A_v_x, sample_covariance)),
            collapse(hadamard_product(A_v_y, sample_covariance)),
            collapse(hadamard_product(A_v_z, sample_covariance))
        };
        Vector3 control_mean_p2 = Vector3{
            trace(A_v_x), trace(A_v_y), trace(A_v_z)
        };

        d_control_v_p[0] = control_variate_p0 - control_mean_p0;
        d_control_v_p[1] = control_variate_p1 - control_mean_p1;
        d_control_v_p[2] = control_variate_p2 - control_mean_p2;

        atomic_add(&d_shapes[control_isect.shape_id].vertices[3 * vidxs[0]],
                -d_control_v_p[0]);
        atomic_add(&d_shapes[control_isect.shape_id].vertices[3 * vidxs[1]],
                -d_control_v_p[1]);
        atomic_add(&d_shapes[control_isect.shape_id].vertices[3 * vidxs[2]],
                -d_control_v_p[2]);

        if (control_isect.shape_id == SHAPE_SELECT && debug_image != nullptr) {
            debug_image[pixel_id] += -(d_control_v_p[0][DIM_SELECT] + 
                                       d_control_v_p[1][DIM_SELECT] + 
                                       d_control_v_p[2][DIM_SELECT]);
        }
    }

    const Scene& scene;

    const KernelParameters& kernel_parameters;

    const int* active_pixels;
    const SurfacePoint* control_points;
    const Intersection* control_isects;
    const Ray* control_rays;
    const CameraSample* control_samples;

    const Vector3* control_mean_grad_contrib;
    const Real* control_mean_contrib;
    const Matrix3x3* control_sample_covariance;

    const DShape* d_shapes;

    const Real weight;

    float* debug_image;
};


void accumulate_control_variates(const Scene& scene,
                    const KernelParameters& kernel_parameters,
                    const BufferView<int>& active_pixels,

                    const BufferView<SurfacePoint>& control_points,
                    const BufferView<Intersection>& control_isects,
                    const BufferView<Ray>& control_rays,
                    const BufferView<CameraSample>& control_samples,

                    const BufferView<Vector3>& control_mean_grad_contrib,
                    const BufferView<Real>& control_mean_contrib,
                    const BufferView<Matrix3x3>& control_sample_covariance,

                    const BufferView<DShape>& d_shapes,
                    const Real weight,
                    
                    float* debug_image) {
    parallel_for(control_variates_accumulator{
        scene,
        kernel_parameters,
        active_pixels.begin(),

        control_points.begin(),
        control_isects.begin(),
        control_rays.begin(),
        control_samples.begin(),

        control_mean_grad_contrib.begin(),
        control_mean_contrib.begin(),
        control_sample_covariance.begin(),

        d_shapes.begin(),

        weight,
        debug_image
    }, active_pixels.size(), scene.use_gpu);
}