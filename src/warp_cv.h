#pragma once

#include "redner.h"
#include "shape.h"
#include "camera.h"

struct Scene;
#include "scene.h"

#include "parallel.h"
#include "thrust_utils.h"
#include "ltc.inc"

#include "warp_common.h"


void accumulate_primary_control_variates(const Scene& scene,
                                 const KernelParameters& kernel_parameters,
                                 const BufferView<int>& active_pixels,
                                 const BufferView<SurfacePoint>& control_points,
                                 const BufferView<Intersection>& control_isects,
                                 const BufferView<Ray>& control_rays,
                                 const BufferView<RayDifferential>& control_ray_differentials,
                                 const BufferView<CameraSample>& control_samples,
                                 const BufferView<Vector3>& control_mean_grad_contrib,
                                 const BufferView<Real>& control_mean_contrib,
                                 const BufferView<Matrix3x3>& control_sample_covariance,
                                 const BufferView<DShape>& d_shapes,
                                 const Real weight,
                                 float* debug_image);


DEVICE
inline
void accumulate_aux_control_variate(
        const Scene& scene,
        const KernelParameters& kernel_parameters,
        const int& pixel_id,

        const CameraSample& camera_sample,
        const Intersection& primary_point,
        const Intersection& primary_isect,
        const Ray& primary_ray,
        const Real weight,
        const Camera* camera,
        const std::vector<AuxSample>& v_aux_samples,
        const int& num_aux_rays,
        const Real loss_contrib,

        DShape* d_shapes,
        float* debug_image
) {
    // Compute the A matrix such that w_i^T A w_i is the control variate.

    Vector2 local_pos;
    Vector2 screen_pos;
    if (is_first_bounce) {
        sample_to_local_pos(*camera,
                            camera_sample,
                            local_pos);

        local_to_screen_pos(*camera,
                            pixel_id,
                            local_pos, screen_pos);
    }

    // Grab the primary intersection and it's triangle vertices.
    auto vidxs = get_indices(shapes[primary_isect.shape_id], primary_isect.tri_id);

    Vector3 p0 = Vector3(array_as_vec3(&shapes[primary_isect.shape_id].vertices[3 * vidxs[0]]));
    Vector3 p1 = Vector3(array_as_vec3(&shapes[primary_isect.shape_id].vertices[3 * vidxs[1]]));
    Vector3 p2 = Vector3(array_as_vec3(&shapes[primary_isect.shape_id].vertices[3 * vidxs[2]]));

    Vector3 d_v0(0, 0, 0);
    Vector3 d_v1(0, 0, 0);
    Vector3 d_v2(0, 0, 0);
    DTRay<double> du_d_ray{
        Vector3(0, 0, 0),
        Vector3(0, 0, 0)
    };

    TRayDifferential<double> zero_d_ray_differential{
            Vector3(0, 0, 0), Vector3(0, 0, 0),
            Vector3(0, 0, 0), Vector3(0, 0, 0)};

    // Compute the first term in A, i.e. the change in the local velocity if the jacobian remains constant

    d_intersect(p0, p1, p2, 
            primary_ray, zero_d_ray_differential, 
            Vector3{Real(1.0), Real(0), Real(0)}, 
            Vector2{0, 0},
            Vector2{0, 0},
            Vector2{0, 0},
            d_v1, d_v1, d_v2,
            du_d_ray,
            zero_d_ray_differential
        );

    DTRay<double> dv_d_ray{
        Vector3(0, 0, 0),
        Vector3(0, 0, 0)
    };
    d_intersect(p0, p1, p2, 
            primary_ray, zero_d_ray_differential, 
            Vector3{Real(0), Real(1.0), Real(0)}, 
            Vector2{0, 0},
            Vector2{0, 0},
            Vector2{0, 0},
            d_v1, d_v1, d_v2,
            dv_d_ray,
            zero_d_ray_differential
        );

    DCamera d_camera(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    Vector2 du_d_screen_pos(0, 0);
    d_sample_primary_ray(*camera, screen_pos, du_d_ray, d_camera, &du_d_screen_pos);
    Vector2 dv_d_screen_pos(0, 0);
    d_sample_primary_ray(*camera, screen_pos, dv_d_ray, d_camera, &dv_d_screen_pos);

    // Contribution of the control variate to u, v and w coefficients.
    Vector3 dv_d_local_pos(0.0,0.0,0.0);
    Vector3 du_d_local_pos(0.0,0.0,0.0);
    Vector3 dw_d_local_pos(0.0,0.0,0.0);

    dv_d_local_pos = vec2_as_vec3(dv_d_screen_pos * (Real(1.0) / Vector2{camera->width, camera->height}));
    du_d_local_pos = vec2_as_vec3(du_d_screen_pos * (Real(1.0) / Vector2{camera->width, camera->height}));
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
    for(uint i = 0; i < num_aux_rays; i++) {
        auto aux_local_pos = aux_primary_local_pos(kernel_parameters,
                            *camera, 
                            local_pos, 
                            v_aux_samples[i]);

        Vector3 w_i = -vec2_as_vec3(aux_local_pos - local_pos) / sigma;

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
                            inv_normalization.at(num_aux_rays - 1) * 
                            Real(M_PI) * 2 * sigma * sigma * 
                            loss_contrib;

        d_primary_v_p[1] += (control_p1 - control_mean_p1) * 
                            inv_normalization.at(num_aux_rays - 1) * 
                            Real(M_PI) * 2 * sigma * sigma * 
                            loss_contrib;

        d_primary_v_p[2] += (control_p2 - control_mean_p2) * 
                            inv_normalization.at(num_aux_rays - 1) * 
                            Real(M_PI) * 2 * sigma * sigma * 
                            loss_contrib;
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
