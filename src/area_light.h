#pragma once

#include "redner.h"
#include "vector.h"
#include "buffer.h"
#include "ptr.h"

struct AreaLight {
    AreaLight() {}

    AreaLight(int shape_id,
              const ptr<float> intensity_data,
              bool two_sided,
              bool directly_visible) :
            shape_id(shape_id),
            two_sided(two_sided),
            directly_visible(directly_visible) {
        intensity[0] = intensity_data[0];
        intensity[1] = intensity_data[1];
        intensity[2] = intensity_data[2];
    }

    AreaLight(int shape_id,
              const Vector3f &intensity,
              bool two_sided,
              bool directly_visible) :
        shape_id(shape_id),
        intensity(intensity),
        two_sided(two_sided),
        directly_visible(directly_visible) {}

    int shape_id;
    Vector3f intensity;
    bool two_sided;
    bool directly_visible;
};

struct DAreaLight {
    DAreaLight(ptr<float> intensity_data) : intensity(intensity_data.get()) {
    }

    float *intensity;
};

template <typename T>
struct TLightSample {
    T light_sel;
    T tri_sel;
    TVector2<T> uv;
};

using LightSample = TLightSample<Real>;
