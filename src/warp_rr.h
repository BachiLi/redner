#pragma once

#include "redner.h"
#include "shape.h"
#include "camera.h"
#include "channels.h"
#include "edge_tree.h"

#include <memory>

struct Scene;
#include "edge.h"
#include "scene.h"
#include "parallel.h"
#include "thrust_utils.h"
#include "ltc.inc"
#include "shape_adjacency.h"

#include "warp_aux.h"

#include <memory>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/remove.h>


/*
 * Samples a stopping count 'N' for the 
 * number of auxillary rays to process.
 * This process is part of the Russian Roulette debiasing method.
 */

void aux_sample_sample_counts(const KernelParameters& kernel_parameters,
                              const Scene& scene,
                              const BufferView<int> &active_pixels,
                              const BufferView<AuxCountSample> &aux_count_samples,
                              BufferView<uint> &aux_sample_counts);


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

// TODO: Add comments and citation link.
DEVICE
inline
void compute_rr_debiased_normalization(
    const KernelParameters& kernel_parameters,
    const int num_aux_rays,
    const std::vector<Real>& v_aux_weights,
    const std::vector<Real>& v_aux_pdfs,
    const std::vector<Vector3>& v_aux_div_weights,
    
    std::vector<Real>& inv_normalization,
    std::vector<Vector3>& grad_inv_normalization
) {
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

}