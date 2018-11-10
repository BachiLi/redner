#pragma once

#include "redner.h"
#include "vector.h"
#include "matrix.h"

template <typename T>
DEVICE
inline TVector3<T> xfm_point(const TMatrix4x4<T> &xform,
                             const TVector3<T> &pt) {
    auto tpt = TVector4<T>{
        xform(0, 0) * pt[0] + xform(0, 1) * pt[1] + xform(0, 2) * pt[2] + xform(0, 3),
        xform(1, 0) * pt[0] + xform(1, 1) * pt[1] + xform(1, 2) * pt[2] + xform(1, 3),
        xform(2, 0) * pt[0] + xform(2, 1) * pt[1] + xform(2, 2) * pt[2] + xform(2, 3),
        xform(3, 0) * pt[0] + xform(3, 1) * pt[1] + xform(3, 2) * pt[2] + xform(3, 3)};
    auto inv_w = 1.f / tpt[3];
    return TVector3<T>{tpt[0], tpt[1], tpt[2]} * inv_w;
}

template <typename T>
DEVICE
inline auto xfm_vector(const TMatrix4x4<T> &xform,
                       const TVector3<T> &vec) {
    return TVector3<T>{
        xform(0, 0) * vec[0] + xform(0, 1) * vec[1] + xform(0, 2) * vec[2],
        xform(1, 0) * vec[0] + xform(1, 1) * vec[1] + xform(1, 2) * vec[2],
        xform(2, 0) * vec[0] + xform(2, 1) * vec[1] + xform(2, 2) * vec[2]};
}

template <typename T>
DEVICE
inline void d_xfm_point(const TMatrix4x4<T> &xform,
                        const TVector3<T> &pt,
                        const TVector3<T> &d_out,
                        TMatrix4x4<T> &d_xform,
                        TVector3<T> &d_pt) {
    auto tpt = TVector4<T>{
        xform(0, 0) * pt[0] + xform(0, 1) * pt[1] + xform(0, 2) * pt[2] + xform(0, 3),
        xform(1, 0) * pt[0] + xform(1, 1) * pt[1] + xform(1, 2) * pt[2] + xform(1, 3),
        xform(2, 0) * pt[0] + xform(2, 1) * pt[1] + xform(2, 2) * pt[2] + xform(2, 3),
        xform(3, 0) * pt[0] + xform(3, 1) * pt[1] + xform(3, 2) * pt[2] + xform(3, 3)};
    auto inv_w = 1.f / tpt[3];
    // out = TVector3<T>{tpt[0], tpt[1], tpt[2]} * inv_w
    auto d_tpt03 = d_out * inv_w;
    auto d_inv_w = sum(d_out * TVector3<T>{tpt[0], tpt[1], tpt[2]});
    // inv_w = 1.f / tpt[3]
    auto d_tpt3 = -d_inv_w * inv_w / tpt[3];
    auto d_tpt = TVector4<T>{d_tpt03[0], d_tpt03[1], d_tpt03[2], d_tpt3};
    // auto tpt = TVector4<T>{
    //     xform(0, 0) * pt[0] + xform(0, 1) * pt[1] + xform(0, 2) * pt[2] + xform(0, 3),
    //     xform(1, 0) * pt[0] + xform(1, 1) * pt[1] + xform(1, 2) * pt[2] + xform(1, 3),
    //     xform(2, 0) * pt[0] + xform(2, 1) * pt[1] + xform(2, 2) * pt[2] + xform(2, 3),
    //     xform(3, 0) * pt[0] + xform(3, 1) * pt[1] + xform(3, 2) * pt[2] + xform(3, 3)};
    d_xform(0, 0) += d_tpt[0] * pt[0];
    d_xform(0, 1) += d_tpt[0] * pt[1];
    d_xform(0, 2) += d_tpt[0] * pt[2];
    d_xform(0, 3) += d_tpt[0];
    d_xform(1, 0) += d_tpt[1] * pt[0];
    d_xform(1, 1) += d_tpt[1] * pt[1];
    d_xform(1, 2) += d_tpt[1] * pt[2];
    d_xform(1, 3) += d_tpt[1];
    d_xform(2, 0) += d_tpt[2] * pt[0];
    d_xform(2, 1) += d_tpt[2] * pt[1];
    d_xform(2, 2) += d_tpt[2] * pt[2];
    d_xform(2, 3) += d_tpt[2];
    d_xform(3, 0) += d_tpt[3] * pt[0];
    d_xform(3, 1) += d_tpt[3] * pt[1];
    d_xform(3, 2) += d_tpt[3] * pt[2];
    d_xform(3, 3) += d_tpt[3];
    d_pt[0] += d_tpt[0] * xform(0, 0) +
               d_tpt[1] * xform(1, 0) +
               d_tpt[2] * xform(2, 0) +
               d_tpt[3] * xform(3, 0);
    d_pt[1] += d_tpt[0] * xform(0, 1) +
               d_tpt[1] * xform(1, 1) +
               d_tpt[2] * xform(2, 1) +
               d_tpt[3] * xform(3, 1);
    d_pt[2] += d_tpt[0] * xform(0, 2) +
               d_tpt[1] * xform(1, 2) +
               d_tpt[2] * xform(2, 2) +
               d_tpt[3] * xform(3, 2);
}

template <typename T>
DEVICE
inline void d_xfm_vector(const TMatrix4x4<T> &xform,
                         const TVector3<T> &vec,
                         const TVector3<T> &d_out,
                         TMatrix4x4<T> &d_xform,
                         TVector3<T> &d_vec) {
    // out =
    // xform(0, 0) * vec[0] + xform(0, 1) * vec[1] + xform(0, 2) * vec[2]
    // xform(1, 0) * vec[0] + xform(1, 1) * vec[1] + xform(1, 2) * vec[2]
    // xform(2, 0) * vec[0] + xform(2, 1) * vec[1] + xform(2, 2) * vec[2]
    d_xform(0, 0) += d_out[0] * vec[0];
    d_xform(0, 1) += d_out[0] * vec[1];
    d_xform(0, 2) += d_out[0] * vec[2];
    d_xform(1, 0) += d_out[1] * vec[0];
    d_xform(1, 1) += d_out[1] * vec[1];
    d_xform(1, 2) += d_out[1] * vec[2];
    d_xform(2, 0) += d_out[2] * vec[0];
    d_xform(2, 1) += d_out[2] * vec[1];
    d_xform(2, 2) += d_out[2] * vec[2];
    d_vec[0] += d_out[0] * xform(0, 0) +
                d_out[1] * xform(1, 0) +
                d_out[2] * xform(2, 0);
    d_vec[1] += d_out[0] * xform(0, 1) +
                d_out[1] * xform(1, 1) +
                d_out[2] * xform(2, 1);
    d_vec[2] += d_out[0] * xform(0, 2) +
                d_out[1] * xform(1, 2) +
                d_out[2] * xform(2, 2);
}
