#pragma once

#include "redner.h"
#include "vector.h"
#include "buffer.h"
#include "ptr.h"

struct Light {
    Light() {}

    Light(int shape_id,
          const ptr<float> intensity_data) : shape_id(shape_id) {
        intensity[0] = intensity_data[0];
        intensity[1] = intensity_data[1];
        intensity[2] = intensity_data[2];
    }

    Light(int shape_id,
          const Vector3f &intensity) :
        shape_id(shape_id), intensity(intensity) {}

    int shape_id;
    Vector3f intensity;
};

struct DLight {
    DLight(ptr<float> intensity_data) : intensity(intensity_data.get()) {
    }

    float *intensity;
};

struct DLightInst {
    int light_id = -1;
    Vector3 intensity = Vector3{0, 0, 0};

    DEVICE inline bool operator<(const DLightInst &other) const {
        return light_id < other.light_id;
    }

    DEVICE inline bool operator==(const DLightInst &other) const {
        return light_id == other.light_id;
    }

    DEVICE inline DLightInst operator+(const DLightInst &other) const {
        return DLightInst{light_id, intensity + other.intensity};
    }
};

template <typename T>
struct TLightSample {
    T light_sel;
    T tri_sel;
    TVector2<T> uv;
};

using LightSample = TLightSample<Real>;

void accumulate_light(const BufferView<DLightInst> &d_light_insts,
                      BufferView<DLight> d_lights,
                      bool use_gpu);

