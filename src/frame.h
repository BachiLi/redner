#pragma once

#include "redner.h"
#include "vector.h"

template <typename T>
struct TFrame {
    DEVICE TFrame() {}

    template <typename T2>
    DEVICE
    TFrame(const TVector3<T2> &x,
           const TVector3<T2> &y,
           const TVector3<T2> &n)
        : x(x), y(y), n(n) {}

    template <typename T2>
    DEVICE
    TFrame(const TVector3<T2> &n) : n(n) {
        coordinate_system(n, x, y);
    }

    DEVICE
    inline TVector3<T>& operator[](int i) {
        return *(&x + i);
    }
    DEVICE
    inline const TVector3<T>& operator[](int i) const {
        return *(&x + i);
    }
    TVector3<T> x, y, n;
};

using Frame = TFrame<Real>;

template <typename T>
DEVICE
inline TFrame<T> operator+=(TFrame<T> &f0, const TFrame<T> &f1) {
    f0.x += f1.x;
    f0.y += f1.y;
    f0.n += f1.n;
    return f0;
}

template <typename T>
DEVICE
inline TFrame<T> operator-(const TFrame<T> &frame) {
    return TFrame<T>{-frame.x, -frame.y, -frame.n};
}

template <typename T0, typename T1>
DEVICE
inline auto to_local(const TFrame<T0> &frame,
                     const TVector3<T1> &v) -> TVector3<decltype(dot(v, frame[0]))> {
    return TVector3<decltype(dot(v, frame[0]))>{
        dot(v, frame[0]),
        dot(v, frame[1]),
        dot(v, frame[2])};
}

template <typename T0, typename T1>
DEVICE
inline auto to_world(const TFrame<T0> &frame,
                     const TVector3<T1> &v) -> decltype(frame[0] * v[0]) {
    return frame[0] * v[0] +
           frame[1] * v[1] +
           frame[2] * v[2];
}

template <typename T>
DEVICE
inline void d_to_world(const TFrame<T> &frame,
                       const TVector3<T> &v,
                       const TVector3<T> &d_dir,
                       TFrame<T> &d_frame,
                       TVector3<T> &d_v) {
    // dir[i] = frame[0][i] * v[0] +
    //          frame[1][i] * v[1] +
    //          frame[2][i] * v[2]
    d_frame[0] += d_dir * v[0];
    d_frame[1] += d_dir * v[1];
    d_frame[2] += d_dir * v[2];
    d_v[0] += sum(d_dir * frame[0]);
    d_v[1] += sum(d_dir * frame[1]);
    d_v[2] += sum(d_dir * frame[2]);
}
