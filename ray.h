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
        : org(org), tmin(T(1e-3)), dir(dir), tmax(infinity<T>()) {}
    template <typename T2>
    DEVICE
    TRay(const TRay<T2> &ray)
        : org(ray.org), tmin(ray.tmin), dir(ray.dir), tmax(ray.tmax) {}

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
#ifdef WIN32
	DTRay(TVector3<T> o = TVector3<T>{ 0, 0, 0 }, TVector3<T> d = TVector3<T>{ 0, 0, 0 })
		:
		org(o),
		dir(d)
	{
	}
#endif
    TVector3<T> org = TVector3<T>{0, 0, 0};
    TVector3<T> dir = TVector3<T>{0, 0, 0};
};

using Ray = TRay<Real>;
using RayDifferential = TRayDifferential<Real>;
using DRay = DTRay<Real>;
using OptiXRay = TRay<float>;
