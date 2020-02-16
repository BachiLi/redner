#pragma once

#include "redner.h"
#include "vector.h"
#include "cuda_utils.h"

#include <limits>

template <typename T>
struct TRay {
    DEVICE TRay() {}
    template <typename T2>
    DEVICE
    TRay(const TVector3<T2> &org, const TVector3<T2> &dir, T2 tmin = 1e-3f)
        : org(org), tmin(tmin), dir(dir), tmax(infinity<T>()) {}
    template <typename T2>
    DEVICE
    TRay(const TRay<T2> &ray)
        : org(ray.org), tmin(ray.tmin), dir(ray.dir), tmax(ray.tmax) {}
    template <typename T2>
    DEVICE
    TRay(const TVector3<T2> &org, const TVector3<T2> &dir, T2 tmin, T2 tmax)
        : org(org), tmin(tmin), dir(dir), tmax(tmax) {}

    // When T == float, this exactly matches Optix prime's ray format
    TVector3<T> org;
    T tmin;
    TVector3<T> dir;
    T tmax;
};

template <typename T>
struct TRayDifferential {
    TVector3<T> org_dx, org_dy;
    TVector3<T> dir_dx, dir_dy;
};

template <typename T>
struct DTRay {
    DEVICE DTRay(const TVector3<T> &o = TVector3<T>{0,0,0},
                 const TVector3<T> &d = TVector3<T>{0,0,0})
        : org(o), dir(d) {}
    TVector3<T> org;
    TVector3<T> dir;
};

struct OptiXHit {
    float t;
    int   tri_id;
    int   inst_id;
};

using Ray = TRay<Real>;
using RayDifferential = TRayDifferential<Real>;
using DRay = DTRay<Real>;
using OptiXRay = TRay<float>;
