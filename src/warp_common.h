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

template <typename T>
struct TKernelParameters {
    // Concentration of the von Mises-Fisher distribution used 
    // to sample the auxillary rays.
    T vMFConcentration;

    T auxPrimaryGaussianStddev;
    T auxPdfEpsilonRegularizer;

    // Std-dev of the inverse gaussian used when computing the
    // asymptotic weights.
    T asymptoteInvGaussSigma;

    // Temperature of the boundary term.
    T asymptoteBoundaryTemp;

    // Gamma power for the weight term.
    // This changes if the number of dimensions change.
    int asymptoteGamma;

    // Weight multiplier for pixel boundaries
    T pixelBoundaryMultiplier;

    // Maximum number of auxillary rays to trace.
    int numAuxillaryRays;

    // RR options. 
    bool rr_enabled;
    T rr_geometric_p;
    // Average runtime equivalent is (batch_size / p)
    int batch_size;

    // If this flag is true, use a simple gaussian kernel.
    bool isBasicNormal;

    TKernelParameters(
        T vMFConcentration,
        T auxPrimaryGaussianStddev,
        T auxPdfEpsilonRegularizer,
        T asymptoteInvGaussSigma,
        T asymptoteBoundaryTemp,
        int asymptoteGamma,
        T pixelBoundaryMultiplier,
        int numAuxillaryRays,
        bool rr_enabled,
        T rr_geometric_p,
        int batch_size,
        bool isBasicNormal) : vMFConcentration(vMFConcentration),
                            auxPrimaryGaussianStddev(auxPrimaryGaussianStddev),
                            auxPdfEpsilonRegularizer(auxPdfEpsilonRegularizer),
                            asymptoteInvGaussSigma(asymptoteInvGaussSigma),
                            asymptoteBoundaryTemp(asymptoteBoundaryTemp),
                            asymptoteGamma(asymptoteGamma),
                            pixelBoundaryMultiplier(pixelBoundaryMultiplier),
                            numAuxillaryRays(numAuxillaryRays),
                            rr_enabled(rr_enabled),
                            rr_geometric_p(rr_geometric_p),
                            batch_size(batch_size),
                            isBasicNormal(isBasicNormal) { }
};

template <typename T>
struct TAuxSample {
    TVector2<T> uv;
};

template <typename T>
struct TAuxCountSample {
    T u;
};

using AuxSample = TAuxSample<Real>;
using AuxCountSample = TAuxCountSample<Real>;
using KernelParameters = TKernelParameters<Real>;


/* Vector3 helpers */

template <typename T>
DEVICE
inline TVector3<T> array_as_vec3(const T* data) {
    return TVector3<T>(data[0], data[1], data[2]);
}

template <typename T>
DEVICE
inline TVector3<T> vec2_as_vec3(const TVector2<T> v2) {
    return TVector3<T>(v2.x, v2.y, static_cast<T>(0.f));
}


/* 
    Utility functions to help with estimating warp quantities
*/


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
    // NOTE: Does not account for truncation bias. (This should be fixed in a future commit)
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
        
        auto a0 = array_as_vec3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[0]]);
        auto a1 = array_as_vec3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[1]]);
        auto a2 = array_as_vec3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[2]]);

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

/* 
 * Compute the boundary term S(x). 
 * This is simply exponentiated negative distance adjusted by the horizon term.
 */
DEVICE
inline
Real warp_boundary_term(const KernelParameters& kernel_parameters,
                  const Shape *shapes,
                  const ShapeAdjacency *shape_adjacencies,
                  const Ray &primary,
                  const Ray &auxillary,
                  const Intersection &aux_isect,
                  const SurfacePoint &aux_point,
                  const SurfacePoint &shading_point,
                  Real& horizon_term) {

    auto boundary_term = Real(0.0);
    if (dot(shading_point.geom_normal, auxillary.dir) * dot(shading_point.geom_normal, primary.dir) < 1e-4) {
        // Outside the horizon ('hit' the black hemispherical occluder)
        horizon_term = 0.0;
        boundary_term = exp(-abs(dot(shading_point.geom_normal, auxillary.dir)) / kernel_parameters.asymptoteBoundaryTemp);
    } else if (aux_isect.valid()) {
        // Hit a valid surface (and isn't beyond horizon)
        auto vidxs = get_indices(shapes[aux_isect.shape_id], aux_isect.tri_id);
    
        auto p0 = array_as_vec3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[0]]);
        auto p1 = array_as_vec3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[1]]);
        auto p2 = array_as_vec3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[2]]);
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

/*
* Computes the asymptotic weight of the auxillary ray's contribution to
* the warp field. This is independent of the parameter.
*/
DEVICE
inline
Real warp_weight(const KernelParameters& kernel_parameters,
                  const Shape *shapes,
                  const ShapeAdjacency *shape_adjacencies,
                  const Ray &primary,
                  const Ray &auxillary,
                  const Intersection &aux_isect,
                  const SurfacePoint &aux_point,
                  const SurfacePoint &shading_point,
                  Real &boundary_term,
                  Real &horizon_term) {

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
    
    const auto gamma = kernel_parameters.asymptoteGamma;
    const auto k = kernel_parameters.vMFConcentration / gamma;
    auto gauss = exp(k * (dot(primary.dir, auxillary.dir) - 1));

    // Compute the harmonic weight.
    auto harmonic = pow(gauss, gamma) / pow(1 - gauss * boundary_term, gamma);

    assert(harmonic >= 0.f);
    return harmonic;
}

/*
 * Computes the asymptotic weight of the auxillary ray's contribution to
 * the warp field. This is independent of the parameter.
 */
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

/*
 * Computes the derivative of the asymptotic weight of the 
 * auxillary ray's contribution to
 * the warp field. This is independent of the parameter.
 */
DEVICE
inline
Vector3 warp_weight_grad( 
                  const KernelParameters& kernel_parameters,
                  const Shape* shapes,
                  const ShapeAdjacency *shape_adjacencies,
                  const Ray &primary,
                  const Ray &auxillary,
                  const Intersection &aux_isect,
                  const SurfacePoint &aux_point,
                  const SurfacePoint &shading_point) {

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


/*
 * Computes the derivative of the asymptotic weight of the 
 * auxillary ray's contribution to
 * the warp field. This is independent of the parameter.
 */
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


/*
 * Computes the jacobian of local sample coordinates
 * w.r.t the intersection point
 */
DEVICE
inline
void warp_jacobian(const KernelParameters& kernel_parameters,
                    const SurfacePoint& shading_point,
                    const SurfacePoint& aux_point,
                    const Intersection& aux_isect,
                    Matrix3x3 &d_dir_d_xg ) {


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


/*
* Computes the jacobian (matrix) of local sample coordinates
* w.r.t the intersection point
* For primary rays, the local sample coordinates are the 2D uniform
* sample in the square pixel grid.
* Note that even though the local samples are 2 dimensional, we return a
* 3x3 jacobian (instead of 2x3), because redner doesn't support arbitrary
* matrices.
*/
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
 * (Utility method for variance reduction)
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
    TVector3<Real> p0 = TVector3<Real>(array_as_vec3(&shapes[isect.shape_id].vertices[3 * vidxs[0]]));
    TVector3<Real> p1 = TVector3<Real>(array_as_vec3(&shapes[isect.shape_id].vertices[3 * vidxs[1]]));
    TVector3<Real> p2 = TVector3<Real>(array_as_vec3(&shapes[isect.shape_id].vertices[3 * vidxs[2]]));

    auto u_dxy = Vector2{0,0};
    auto v_dxy = Vector2{0,0};
    auto t_dxy = Vector2{0,0};
    TRayDifferential<Real> zero_d_ray_differential{
                    TVector3<Real>(0, 0, 0), TVector3<Real>(0, 0, 0),
                  TVector3<Real>(0, 0, 0), TVector3<Real>(0, 0, 0)};
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