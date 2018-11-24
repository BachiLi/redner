#pragma once

#include "redner.h"

/**
 *  Specialized 2D dual number implementation for screen space derivatives
 */

template <typename T>
struct DualNumber {
    DEVICE
    DualNumber(const T &value) : value(value), dx(T(0)), dy(T(0)) {}
    DEVICE
    DualNumber(const T &value, const T &dx, const T &dy) : value(value), dx(dx), dy(dy) {}

    T value, dx, dy;
};

template <typename T>
DEVICE
DualNumber<T> operator+(const DualNumber<T> &v0, const DualNumber<T> &v1) {
    return DualNumber<T>{v0.value + v1.value,
                         v0.dx + v1.dx,
                         v0.dy + v1.dy};
}

template <typename T>
DEVICE
DualNumber<T> operator+(const T &v0, const DualNumber<T> &v1) {
    return DualNumber<T>{v0 + v1.value,
                         v1.dx,
                         v1.dy};
}

template <typename T>
DEVICE
DualNumber<T> operator+(const DualNumber<T> &v0, const T &v1) {
    return DualNumber<T>{v0.value + v1,
                         v0.dx,
                         v0.dy};
}

template <typename T>
DEVICE
DualNumber<T>& operator+=(DualNumber<T> &v0, const T &v1) {
    v0.value += v1;
    return v0;
}

template <typename T>
DEVICE
DualNumber<T>& operator+=(DualNumber<T> &v0, const DualNumber<T> &v1) {
    v0.value += v1.value;
    v0.dx += v1.dx;
    v0.dy += v1.dy;
    return v0;
}


template <typename T>
DEVICE
DualNumber<T> operator-(const DualNumber<T> &v0) {
    return DualNumber<T>{-v0.value,
                         -v0.dx,
                         -v0.dy};
}

template <typename T>
DEVICE
DualNumber<T> operator-(const DualNumber<T> &v0, const DualNumber<T> &v1) {
    return DualNumber<T>{v0.value - v1.value,
                         v0.dx - v1.dx,
                         v0.dy - v1.dy};
}

template <typename T>
DEVICE
DualNumber<T> operator-(const T &v0, const DualNumber<T> &v1) {
    return DualNumber<T>{v0 - v1.value,
                         -v1.dx,
                         -v1.dy};
}

template <typename T>
DEVICE
DualNumber<T> operator-(const DualNumber<T> &v0, const T &v1) {
    return DualNumber<T>{v0.value - v1,
                         v0.dx,
                         v0.dy};
}

template <typename T>
DEVICE
DualNumber<T>& operator-=(DualNumber<T> &v0, const T &v1) {
    v0.value -= v1;
    return v0;
}

template <typename T>
DEVICE
DualNumber<T>& operator-=(DualNumber<T> &v0, const DualNumber<T> &v1) {
    v0.value -= v1.value;
    v0.dx -= v1.dx;
    v0.dy -= v1.dy;
    return v0;
}


template <typename T>
DEVICE
DualNumber<T> operator*(const DualNumber<T> &v0, const DualNumber<T> &v1) {
    return DualNumber<T>{v0.value * v1.value,
                         v0.dx * v1.value + v0.value * v1.dx,
                         v0.dy * v1.value + v0.value * v1.dy};
}

template <typename T>
DEVICE
DualNumber<T> operator*(const T &v0, const DualNumber<T> &v1) {
    return DualNumber<T>{v0 * v1.value,
                         v0 * v1.dx,
                         v0 * v1.dy};
}

template <typename T>
DEVICE
DualNumber<T> operator*(const DualNumber<T> &v0, const T &v1) {
    return DualNumber<T>{v0.value * v1,
                         v0.dx * v1,
                         v0.dy * v1};
}

template <typename T>
DEVICE
DualNumber<T>& operator*=(DualNumber<T> &v0, const T &v1) {
    v0.value *= v1;
    return v0;
}

template <typename T>
DEVICE
DualNumber<T>& operator*=(DualNumber<T> &v0, const DualNumber<T> &v1) {
    v0.value *= v1.value;
    v0.dx = v0.dx * v1.value + v0.value * v1.dx;
    v0.dy = v0.dy * v1.value + v0.value * v1.dy;
    return v0;
}


template <typename T>
DEVICE
DualNumber<T> operator/(const DualNumber<T> &v0, const DualNumber<T> &v1) {
    return DualNumber<T>{v0.value / v1.value,
                         (v0.dx * v1.value - v0.value * v1.dx) / (v1.value * v1.value),
                         (v0.dy * v1.value - v0.value * v1.dy) / (v1.value * v1.value)};
}

template <typename T>
DEVICE
DualNumber<T> operator/(const T &v0, const DualNumber<T> &v1) {
    return DualNumber<T>{v0 / v1.value,
                         (- v0 * v1.dx) / (v1.value * v1.value),
                         (- v0 * v1.dy) / (v1.value * v1.value)};
}

template <typename T>
DEVICE
DualNumber<T> operator/(const DualNumber<T> &v0, const T &v1) {
    return DualNumber<T>{v0.value / v1,
                         v0.dx / v1,
                         v0.dy / v1};
}

template <typename T>
DEVICE
DualNumber<T>& operator/=(DualNumber<T> &v0, const T &v1) {
    v0.value /= v1;
    return v0;
}

template <typename T>
DEVICE
DualNumber<T>& operator/=(DualNumber<T> &v0, const DualNumber<T> &v1) {
    v0.value /= v1.value;
    v0.dx = (v0.dx * v1.value - v0.value * v1.dx) / (v1.value * v1.value);
    v0.dy = (v0.dy * v1.value - v0.value * v1.dy) / (v1.value * v1.value);
    return v0;
}

template <typename T>
DEVICE
DualNumber<T> sqrt(const DualNumber<T> &v) {
    auto sqrt_value = sqrt(v.value);
    return DualNumber<T>{sqrt_value,
                         T(0.5) * v.dx / sqrt_value,
                         T(0.5) * v.dy / sqrt_value};
}

template <typename T>
DEVICE
DualNumber<T> pow(const DualNumber<T> &v0, const DualNumber<T> &v1) {
    return DualNumber<T>{pow(v0.value, v1.value),
                         pow(v0.value, v1.value - 1) * 
                            (v1.value * v0.dx + v0.value * log(v0.value) * v1.dx),
                         pow(v0.value, v1.value - 1) * 
                            (v1.value * v0.dy + v0.value * log(v0.value) * v1.dy)};
}

template <typename T>
DEVICE
DualNumber<T> pow(const DualNumber<T> &v0, const T &v1) {
    return DualNumber<T>{pow(v0.value, v1),
                         v0.dx * v1 * pow(v0.value, v1 - 1),
                         v0.dy * v1 * pow(v0.value, v1 - 1)};
}

template <typename T>
DEVICE
DualNumber<T> exp(const DualNumber<T> &v) {
    auto exp_value = exp(v.value);
    return DualNumber<T>{exp_value,
                         v.dx * exp_value,
                         v.dy * exp_value};
}

template <typename T>
DEVICE
DualNumber<T> log(const DualNumber<T> &v) {
    return DualNumber<T>{log(v.value),
                         v.dx / v.value,
                         v.dy / v.value};
}

template <typename T>
DEVICE
DualNumber<T> sin(const DualNumber<T> &v) {
    return DualNumber<T>{sin(v.value),
                         v.dx * cos(v.value),
                         v.dy * cos(v.value)};
}

template <typename T>
DEVICE
DualNumber<T> cos(const DualNumber<T> &v) {
    return DualNumber<T>{cos(v.value),
                         -v.dx * sin(v.value),
                         -v.dy * sin(v.value)};
}

template <typename T>
DEVICE
DualNumber<T> acos(const DualNumber<T> &v) {
    return DualNumber<T>{acos(v.value),
                         -v.dx / sqrt(1 - v.value * v.value),
                         -v.dy / sqrt(1 - v.value * v.value)};
}

template <typename T>
DEVICE
DualNumber<T> asin(const DualNumber<T> &v) {
    return DualNumber<T>{asin(v.value),
                         v.dx / sqrt(1 - v.value * v.value),
                         v.dy / sqrt(1 - v.value * v.value)};
}

template <typename T>
DEVICE
DualNumber<T> atan2(const DualNumber<T> &y, const DualNumber<T> &x) {
    auto r2 = y.value * y.value + x.value * x.value;
    return DualNumber<T>{atan2(y, x),
                         y.dx * (x.value / r2) - x.dx * (y.value / r2),
                         y.dy * (x.value / r2) - x.dy * (y.value / r2)};
}
