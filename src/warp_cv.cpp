#include "warp_cv.h"

struct control_variates_accumulator{

    DEVICE void operator()(int idx) { 
        const int pixel_id = active_pixels[idx];

        auto control_point = control_points[pixel_id];
        auto control_isect = control_isects[pixel_id];
        auto control_ray = control_rays[pixel_id];
        auto control_ray_differential = control_ray_differentials[pixel_id];

        if (!control_isect.valid()) {
            // No intersection. PCV is 0.
            return;
        }

        // Grab the primary isect and it's vertices.
        auto vidxs = get_indices(scene.shapes[control_isect.shape_id], control_isect.tri_id);

        Vector3 p0 = Vector3(array_as_vec3(&scene.shapes[control_isect.shape_id].vertices[3 * vidxs[0]]));
        Vector3 p1 = Vector3(array_as_vec3(&scene.shapes[control_isect.shape_id].vertices[3 * vidxs[1]]));
        Vector3 p2 = Vector3(array_as_vec3(&scene.shapes[control_isect.shape_id].vertices[3 * vidxs[2]]));

        Vector3 d_v0(0, 0, 0);
        Vector3 d_v1(0, 0, 0);
        Vector3 d_v2(0, 0, 0);
        DTRay<Real> du_d_ray{ Vector3(0, 0, 0), Vector3(0, 0, 0) };
        TRayDifferential<Real> d_ray_differential{
                Vector3(0, 0, 0), Vector3(0, 0, 0),
                Vector3(0, 0, 0), Vector3(0, 0, 0)};

        // Compute the first term in A, i.e. the change in the local velocity if the jacobian remains constant.

        d_intersect(p0, p1, p2, 
                control_ray, control_ray_differential, 
                Vector3{Real(1.0), Real(0), Real(0)}, 
                TVector2<Real>{Real(0), Real(0)},
                TVector2<Real>{Real(0), Real(0)},
                TVector2<Real>{Real(0), Real(0)},
                d_v1, d_v1, d_v2,
                du_d_ray,
                d_ray_differential
            );

        DTRay<Real> dv_d_ray{ Vector3(0, 0, 0), Vector3(0, 0, 0) };
        d_intersect(p0, p1, p2, 
                control_ray, control_ray_differential, 
                Vector3{Real(0), Real(1.0), Real(0)}, 
                TVector2<Real>{Real(0), Real(0)},
                TVector2<Real>{Real(0), Real(0)},
                TVector2<Real>{Real(0), Real(0)},
                d_v1, d_v1, d_v2,
                dv_d_ray,
                d_ray_differential
            );

        Vector2 local_pos;
        Vector2 screen_pos;
        sample_to_local_pos(scene.camera, control_samples[pixel_id], local_pos);
        local_to_screen_pos(scene.camera,
                            pixel_id,
                            local_pos, screen_pos);

        DCamera d_camera(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        Vector2 du_d_screen_pos(0.0, 0.0);
        d_sample_primary_ray(scene.camera, screen_pos, du_d_ray, d_camera, &du_d_screen_pos);
        Vector2 dv_d_screen_pos(0.0, 0.0);
        d_sample_primary_ray(scene.camera, screen_pos, dv_d_ray, d_camera, &dv_d_screen_pos);

        Vector2 _du_d_local_pos(0.0, 0.0);
        d_local_to_screen_pos(scene.camera, pixel_id, du_d_screen_pos, _du_d_local_pos);
        Vector2 _dv_d_local_pos(0.0, 0.0);
        d_local_to_screen_pos(scene.camera, pixel_id, dv_d_screen_pos, _dv_d_local_pos);

        auto dv_d_local_pos = vec2_as_vec3(_dv_d_local_pos);
        auto du_d_local_pos = vec2_as_vec3(_du_d_local_pos);
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

        // Compute the second term, i.e change in the jacobian if the velocity remains constant.

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
            reduce_sum(element_wise_product(A_w_x, sample_covariance)),
            reduce_sum(element_wise_product(A_w_y, sample_covariance)),
            reduce_sum(element_wise_product(A_w_z, sample_covariance))
        };
        Vector3 control_mean_p0 = Vector3{
            trace(A_w_x), trace(A_w_y), trace(A_w_z)
        };

        Vector3 control_variate_p1 = Vector3{
            reduce_sum(element_wise_product(A_u_x, sample_covariance)),
            reduce_sum(element_wise_product(A_u_y, sample_covariance)),
            reduce_sum(element_wise_product(A_u_z, sample_covariance))
        };
        Vector3 control_mean_p1 = Vector3{
            trace(A_u_x), trace(A_u_y), trace(A_u_z)
        };

        Vector3 control_variate_p2 = Vector3{
            reduce_sum(element_wise_product(A_v_x, sample_covariance)),
            reduce_sum(element_wise_product(A_v_y, sample_covariance)),
            reduce_sum(element_wise_product(A_v_z, sample_covariance))
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
    const RayDifferential* control_ray_differentials;
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
                    const BufferView<RayDifferential>& control_ray_differentials,
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
        control_ray_differentials.begin(),
        control_samples.begin(),

        control_mean_grad_contrib.begin(),
        control_mean_contrib.begin(),
        control_sample_covariance.begin(),

        d_shapes.begin(),
        weight,

        debug_image
    }, active_pixels.size(), scene.use_gpu);
}