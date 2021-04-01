
#include "redner.h"
#include "shape.h"
#include "camera.h"
#include "channels.h"
#include "edge_tree.h"
#include "warp_field.h"

#include <memory>

struct Scene;
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

/*
 * This accumulator is intended to be called AFTER path_contribution.
 * It relies on d_bsdf_wos and d_light_wos
 * which are computed in the d_path_contribution pass. 
 */
struct warp_derivatives_accumulator {
    DEVICE void operator()(int idx) {
        auto pixel_id = active_pixels[idx];

        const auto num_aux_rays = aux_sample_counts[pixel_id];
        const auto max_aux_rays = kernel_parameters.numAuxillaryRays;

        /* 
         * Flag to indicate if we're handling primary or secondary bounce.
         * We use pixel space coordinates for primary rays and 
         * hemispherical coordinates for the secondary bounces.
         * The fundamanetal mathematical technique remains the same. 
         * However, the computation of weights, jacobians, require different functions to be used.
         */
        bool is_first_bounce = (camera_samples != nullptr);

        // Load 'origin' surface point.
        const auto &shading_point = (!is_first_bounce) ? 
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
        if (is_first_bounce) {
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
        v_aux_rays.assign(&aux_rays[max_aux_rays * pixel_id], &aux_rays[max_aux_rays * pixel_id + (max_aux_rays)]);
        v_aux_isects.assign(&aux_isects[max_aux_rays * pixel_id], &aux_isects[max_aux_rays * pixel_id + (max_aux_rays)]);
        v_aux_points.assign(&aux_points[max_aux_rays * pixel_id], &aux_points[max_aux_rays * pixel_id + (max_aux_rays)]);
        if(is_first_bounce) {
            v_aux_samples.assign(&aux_samples[max_aux_rays * pixel_id], &aux_samples[max_aux_rays * pixel_id + (max_aux_rays)]);
        }

        // Compute auxillary ray properties.
        std::vector<Real> v_aux_weights(num_aux_rays, 0);
        std::vector<Vector3> v_aux_div_weights(num_aux_rays, Vector3(0.0, 0.0, 0.0));

        // TODO: Buffers for debugging.
        std::vector<Real> v_aux_boundary_terms(num_aux_rays, 0);
        std::vector<Real> v_aux_horizon_terms(num_aux_rays, 0);

        for(uint i = 0; i < num_aux_rays; i++) {
            
            /* Filter degenerate rays */
            if(shading_isects != nullptr) {
                auto shading_isect = shading_isects[pixel_id];
                auto aux_isect = v_aux_isects.at(i);
                if (shading_isect == aux_isect) {
                    continue;
                }
            }

            /* Compute weights w(x,x') */
            if (is_first_bounce) {
                auto aux_local_pos = aux_primary_local_pos(kernel_parameters,
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
        std::vector<Real> v_aux_pdfs(num_aux_rays, 0);
        for(uint i = 0; i < num_aux_rays; i++) {
            // TODO: Make this more readable.
            if (!is_first_bounce)
                // Computes von Mises-Fisher pdf.
                v_aux_pdfs.at(i) = aux_pdf(kernel_parameters, v_aux_rays.at(i), primary_ray);
            else
                // Computes gaussian PDF.
                v_aux_pdfs.at(i) = aux_primary_pdf(kernel_parameters,
                                                   *camera, 
                                                   camera_samples[pixel_id],
                                                   v_aux_samples.at(i));
        }

        Real normalization = 0;
        Vector3 div_normalization(0, 0, 0);
        // Compute the normalizers Z(x) and div.Z(x) (independent of parameter)
        for(uint i = 0; i < num_aux_rays; i++) {
            normalization += v_aux_weights.at(i) / v_aux_pdfs.at(i);
            div_normalization += v_aux_div_weights.at(i) / v_aux_pdfs.at(i);
        }

        // Compute the inverse normalization. This is the main source of bias.
        // To handle this we provide two modes simple monte carlo (biased) and 
        // RR (Russian Roulette) (unbiased but higher variance)

        std::vector<Real> inv_normalization(num_aux_rays, 0);
        std::vector<Vector3> grad_inv_normalization(num_aux_rays, Vector3{0.0, 0.0, 0.0});

        if(kernel_parameters.isBasicNormal){
            // Normalization is 1 (known distribution) if the distribution is standard normal.
            normalization = kernel_parameters.numAuxillaryRays;
            div_normalization = Vector3{0, 0, 0};
        }

        // Russian roulette estimation.
        // Estimates quantites using Russian roulette that are otherwise biased:
        // (i) 'inv_normalization' := reciprocal of the weight integral (normalization) 1/\int_{x'}(w(x, x'))
        // (ii) 'grad_inv_normalization' := derivative of this reciprocal (\int{x'}grad_w(x, x'))/\int_{x'}(w^2(x, x'))
        if (kernel_parameters.rr_enabled) {
            std::vector<Real> _acc_wt_sum = std::vector<Real>(num_aux_rays, 0.0);
            std::vector<Vector3> _acc_grad_wt_sum = std::vector<Vector3>(num_aux_rays, Vector3{0.0, 0.0, 0.0});
            for(int i = 0; i < num_aux_rays; i++) {
                _acc_wt_sum.at(i) = ((i != 0) ? _acc_wt_sum.at(i - 1) : 0) + (v_aux_weights.at(i) / v_aux_pdfs.at(i));
                _acc_grad_wt_sum.at(i) = ((i != 0) ? _acc_grad_wt_sum.at(i - 1) : Vector3{0.0, 0.0, 0.0}) + (v_aux_div_weights.at(i) / v_aux_pdfs.at(i));
            }

            Real Z = 0.0;
            Vector3 grad_Z = Vector3{0.0, 0.0, 0.0};
            int batchsz = kernel_parameters.batch_size;
            for(int k = num_aux_rays - 1; k >= 0; k--) {
                // Compute the estimator values cumulatively for each batch.
                if (k % batchsz == 0) {
                    // Compute the harmonic difference of the pdfs of DeltaX_i and DeltaX_i+1.
                    Real pdf_i = _aux_sample_sample_counts_pdf(kernel_parameters, k + batchsz);
                    Real pdf_next_i = _aux_sample_sample_counts_pdf(kernel_parameters, k + 2 * batchsz);
                    Real effective_pdf = (pdf_i * pdf_next_i) / (pdf_next_i - pdf_i);

                    // The last element in the sequence occurs only once. The effective pdf is the same as the pdf.
                    if (k == num_aux_rays - batchsz)
                        effective_pdf = pdf_i;

                    int kidx = k + batchsz - 1;
                    Z = Z + (1.0 / (_acc_wt_sum.at(kidx))) / effective_pdf;
                    grad_Z = grad_Z + ( (_acc_grad_wt_sum.at(kidx)) / ((_acc_wt_sum.at(kidx)) * (_acc_wt_sum.at(kidx))) ) / effective_pdf;

                    for(int offset = 0; offset < batchsz; offset++){
                        inv_normalization.at(k + offset) = Z;
                        grad_inv_normalization.at(k + offset) = grad_Z;
                    }
                }
            }

        } else {
            for(int i = 0; i < num_aux_rays; i++){
                inv_normalization.at(i) =  1.0 / normalization;
                grad_inv_normalization.at(i) = div_normalization * (1.0 / normalization) * (1.0 / normalization);
            }
        }

        auto nd = channel_info.num_total_dimensions;
        auto d = channel_info.radiance_dimension;

        // Compute pixel weight w.r.t the current loss function.
        auto df_d_path_contrib = weight *
        Vector3{d_rendered_image[nd * pixel_id + d    ],
                d_rendered_image[nd * pixel_id + d + 1],
                d_rendered_image[nd * pixel_id + d + 2]};

        // Compute control variates for denoising (if enabled)
        // This section is only for variance reduction and does not affect the bias of the esimator
        // The current implementation is only for primary rays.
        // TODO: Move to its own block.
        if(enable_aux_control_variates 
            && is_first_bounce
            && primary_isect.valid() ) {
            // Compute the A matrix such that w_i^T A w_i is the control variate.

            // We need to find the derivative of the coefficients alpha, beta and gamma w.r.t omega

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
                                    sum(df_d_path_contrib * path_contrib);

                d_primary_v_p[1] += (control_p1 - control_mean_p1) * 
                                    inv_normalization.at(num_aux_rays - 1) * 
                                    Real(M_PI) * 2 * sigma * sigma * 
                                    sum(df_d_path_contrib * path_contrib);

                d_primary_v_p[2] += (control_p2 - control_mean_p2) * 
                                    inv_normalization.at(num_aux_rays - 1) * 
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

        // This is the derivative of the contribution w.r.t 
        // spatial domain variable.
        // The spatial domain variable for secondary bounces is the space of
        // outgoing directions in the hemisphere 'wo'
        // The spatial domain variable for primary bounce is the 2D square
        // covering the pixel.
        auto df_dw = (!is_first_bounce) ?
                        -df_dwos[pixel_id] : // Secondary bounce. Domain is unit 3-vector wo.
                        Vector3 {          // First bounce. Domain is unit square.
                            df_d_local_pos[pixel_id].x,
                            df_d_local_pos[pixel_id].y,
                            0.0
                        };

        // Slightly hacky.
        if (primary_ray.tmax <= 0 && df_dwos != nullptr) {
            // Occluded NEE ray, the gradient must be auto set to 0.
            df_dw = Vector3{0,0,0};
        }

        // -----------------
        /*
            Accumulate aggregate values.
            These are used by accumulate_control_variates() to compute its contribution.
        */
        auto w_i = vec2_as_vec3(local_pos);
        if(sample_covariance != nullptr && is_first_bounce) {
            sample_covariance[pixel_id] = 
                sample_covariance[pixel_id] + outer_product(w_i, w_i);
            mean_contrib[pixel_id] = 
                mean_contrib[pixel_id] + sum(df_d_path_contrib * path_contrib);
            mean_grad_contrib[pixel_id] = 
                mean_grad_contrib[pixel_id] + df_dw;
        }
        // ----------------------------
        
        Vector3 kernel_score(0,0,0);
        Real filter_weight;
        Vector3 F_gradK(0, 0, 0);
        if(is_first_bounce) {
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
        for(uint i = 0; i < num_aux_rays; i++) {
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

            if (!is_first_bounce) {
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
                                  (v_aux_weights.at(i) / v_aux_pdfs.at(i)) * grad_inv_normalization.at(i);

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
            auto gradF_dot_V_org = df_dw * V_org + F_gradK * V_org; // gradF.K.V + F.gradK.V
            auto F_mul_div_V_org = sum(df_d_path_contrib * path_contrib) * divV_org; // F.K.divV

            // Compute final gradients.
            auto grad_xg = gradF_dot_V_xg + F_mul_div_V_xg;
            auto grad_org = gradF_dot_V_org + F_mul_div_V_org;
            auto grad_dir = gradF_dot_V_dir + F_mul_div_V_dir;
            // ---

            if(is_first_bounce) {
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

                if (screen_gradient_image != nullptr) {
                    screen_gradient_image[2 * pixel_id + 0] += d_screen_pos.x;
                    screen_gradient_image[2 * pixel_id + 1] += d_screen_pos.y;
                }
            }

            if(!is_first_bounce && d_shading_points != nullptr) {
                d_shading_points[pixel_id].position -= (F_mul_div_V_xg + gradF_dot_V_xg);
            }

            auto &t_aux_isect = v_aux_isects.at(i);
            if(t_aux_isect.valid() && 
                (is_first_bounce ||
                (d_shading_points != nullptr))
            ) {
                Vector3 d_aux_v_p[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                Vector3 ignore_d_aux_v_n[3] = {Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
                Vector2 ignore_d_aux_v_uv[3] = {Vector2{0, 0}, Vector2{0, 0}};

                // TODO: Ray differential is not properly handled.
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
    const Vector3 *df_dwos;
    const Real weight;
    const ChannelInfo channel_info;
    const KernelParameters kernel_parameters; // Info for asymptotic sampling and weighting.
    const float* d_rendered_image;

    // Enables control variates for the computation of the warp divergence.
    const bool enable_aux_control_variates;

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

    // Images for debug output.
    float* debug_image;
    float* screen_gradient_image;

    // Aggregate buffers used by accumulate_control_variates() to build 
    // its linear approximation to reduce variance.
    Vector3* mean_grad_contrib;
    Real* mean_contrib;
    Matrix3x3* sample_covariance;
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
                                           const BufferView<Vector3> &df_dwos,
                                           const Real weight,
                                           const ChannelInfo channel_info,
                                           const KernelParameters &kernel_parameters,
                                           const float *d_rendered_image,
                                           const bool enable_aux_control_variates,
                                           BufferView<SurfacePoint> d_shading_points,
                                           BufferView<DShape> d_shapes,
                                           BufferView<CameraSample> camera_samples,
                                           BufferView<Vector2> df_d_local_pos,
                                           ShapeAdjacency* adjacency,
                                           DScene* d_scene,
                                           float* debug_image,
                                           float* screen_gradient_image,
                                           BufferView<Vector3> mean_grad_contrib,
                                           BufferView<Real> mean_contrib,
                                           BufferView<Matrix3x3> sample_covariance){
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
        df_dwos.begin(),
        weight,
        channel_info,
        kernel_parameters,
        d_rendered_image,
        enable_aux_control_variates,
        d_shading_points.begin(),
        d_shapes.begin(),
        camera_samples.begin(),
        df_d_local_pos.begin(),
        &d_scene->camera,
        &scene.camera,
        adjacency,
        debug_image,
        screen_gradient_image,
        mean_grad_contrib.begin(),
        mean_contrib.begin(),
        sample_covariance.begin(),
    }, active_pixels.size(), scene.use_gpu);
}
