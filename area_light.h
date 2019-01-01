#pragma once

#include "redner.h"
#include "vector.h"
#include "buffer.h"
#include "ptr.h"

struct AreaLight {
    AreaLight() {}

    AreaLight(int shape_id,
              const ptr<float> intensity_data,
              bool two_sided) : shape_id(shape_id), two_sided(two_sided) {
        intensity[0] = intensity_data[0];
        intensity[1] = intensity_data[1];
        intensity[2] = intensity_data[2];
    }

    AreaLight(int shape_id,
              const Vector3f &intensity,
              bool two_sided) :
        shape_id(shape_id), intensity(intensity), two_sided(two_sided) {}

    int shape_id;
    Vector3f intensity;
    bool two_sided;
};

struct DAreaLight {
    DAreaLight(ptr<float> intensity_data) : intensity(intensity_data.get()) {
    }

    float *intensity;
};

struct DAreaLightInst {
#ifdef WIN32
	DAreaLightInst(int li = -1, Vector3& i = Vector3{ 0, 0, 0 })
		:
		light_id(li), intensity(i)
	{
	}
#endif
    int light_id = -1;
    Vector3 intensity = Vector3{0, 0, 0};

    DEVICE inline bool operator<(const DAreaLightInst &other) const {
        return light_id < other.light_id;
    }

    DEVICE inline bool operator==(const DAreaLightInst &other) const {
        return light_id == other.light_id;
    }

    DEVICE inline DAreaLightInst operator+(const DAreaLightInst &other) const {
        return DAreaLightInst{light_id, intensity + other.intensity};
    }
};

template <typename T>
struct TLightSample {
    T light_sel;
    T tri_sel;
    TVector2<T> uv;
};

using LightSample = TLightSample<Real>;

void accumulate_area_light(const BufferView<DAreaLightInst> &d_light_insts,
                           BufferView<DAreaLight> d_lights,
                           bool use_gpu);
