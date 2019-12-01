#pragma once

#include "vector.h"
#include "frame.h"

template <typename T>
struct TMatrix3x3 {
    DEVICE
    TMatrix3x3() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                data[i][j] = T(0);
            }
        }
    }

    template <typename T2>
    DEVICE
    TMatrix3x3(T2 *arr) {
    	data[0][0] = arr[0];
    	data[0][1] = arr[1];
    	data[0][2] = arr[2];
    	data[1][0] = arr[3];
    	data[1][1] = arr[4];
    	data[1][2] = arr[5];
    	data[2][0] = arr[6];
    	data[2][1] = arr[7];
    	data[2][2] = arr[8];
    }
    DEVICE
    TMatrix3x3(T v00, T v01, T v02,
               T v10, T v11, T v12,
               T v20, T v21, T v22) {
        data[0][0] = v00;
        data[0][1] = v01;
        data[0][2] = v02;
        data[1][0] = v10;
        data[1][1] = v11;
        data[1][2] = v12;
        data[2][0] = v20;
        data[2][1] = v21;
        data[2][2] = v22;
    }

    template <typename T2>
    DEVICE
    TMatrix3x3(const TFrame<T2> &f) {
        data[0][0] = f[0][0];
        data[0][1] = f[0][1];
        data[0][2] = f[0][2];
        data[1][0] = f[1][0];
        data[1][1] = f[1][1];
        data[1][2] = f[1][2];
        data[2][0] = f[2][0];
        data[2][1] = f[2][1];
        data[2][2] = f[2][2];
    }
    DEVICE
    const T& operator()(int i, int j) const {
        return data[i][j];
    }
    DEVICE
    T& operator()(int i, int j) {
        return data[i][j];
    }
    DEVICE
    static TMatrix3x3<T> identity() {
        TMatrix3x3<T> m(1, 0, 0,
                        0, 1, 0,
                        0, 0, 1);
        return m;
    }

    T data[3][3];
};

using Matrix3x3 = TMatrix3x3<Real>;
using Matrix3x3f = TMatrix3x3<float>;

template <typename T>
struct TMatrix4x4 {
    DEVICE TMatrix4x4() {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                data[i][j] = T(0);
            }
        }
    }

    template <typename T2>
    DEVICE TMatrix4x4(const T2 *arr) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                data[i][j] = (T)arr[i * 4 + j];
            }
        }
    }

    template <typename T2>
    DEVICE TMatrix4x4(const TMatrix4x4<T2> &m) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                data[i][j] = T(m.data[i][j]);
            }
        }
    }

    template <typename T2>
    DEVICE TMatrix4x4(T2 v00, T2 v01, T2 v02, T2 v03,
                      T2 v10, T2 v11, T2 v12, T2 v13,
                      T2 v20, T2 v21, T2 v22, T2 v23,
                      T2 v30, T2 v31, T2 v32, T2 v33) {
        data[0][0] = (T)v00;
        data[0][1] = (T)v01;
        data[0][2] = (T)v02;
        data[0][3] = (T)v03;
        data[1][0] = (T)v10;
        data[1][1] = (T)v11;
        data[1][2] = (T)v12;
        data[1][3] = (T)v13;
        data[2][0] = (T)v20;
        data[2][1] = (T)v21;
        data[2][2] = (T)v22;
        data[2][3] = (T)v23;
        data[3][0] = (T)v30;
        data[3][1] = (T)v31;
        data[3][2] = (T)v32;
        data[3][3] = (T)v33;
    }

    DEVICE
    const T& operator()(int i, int j) const {
        return data[i][j];
    }

    DEVICE
    T& operator()(int i, int j) {
        return data[i][j];
    }

    DEVICE
    static TMatrix4x4<T> identity() {
        TMatrix4x4<T> m(1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1);
        return m;
    }

    T data[4][4];
};

using Matrix4x4 = TMatrix4x4<Real>;
using Matrix4x4f = TMatrix4x4<float>;

template <typename T0, typename T1>
DEVICE
inline auto operator+(const TMatrix3x3<T0> &m0, const TMatrix3x3<T1> &m1)
        -> TMatrix3x3<decltype(T0(0) * T1(0))> {
    TMatrix3x3<decltype(m0(0, 0) + m1(0, 0))> m;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            m(i, j) = m0(i, j) + m1(i, j);
        }
    }
    return m;
}

template <typename T0, typename T1>
DEVICE
inline auto operator-(const TMatrix3x3<T0> &m0, const TMatrix3x3<T1> &m1) 
        -> TMatrix3x3<decltype(T0(0) - T1(0))>{
    TMatrix3x3<decltype(m0(0, 0) - m1(0, 0))> m;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            m(i, j) = m0(i, j) - m1(i, j);
        }
    }
    return m;
}

template <typename T>
DEVICE
inline TMatrix3x3<T> operator*(const TMatrix3x3<T> &m0, const TMatrix3x3<T> &m1) {
    TMatrix3x3<T> ret;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            ret(i, j) = T(0);
            for (int k = 0; k < 3; k++) {
                ret(i, j) += m0(i, k) * m1(k, j);
            }
        }
    }
    return ret;
}

template <typename T>
DEVICE
inline TVector3<T> operator*(const TVector3<T> &v, const TMatrix3x3<T> &m) {
    TVector3<T> ret;
    for (int i = 0; i < 3; i++) {
        ret[i] = T(0);
        for (int j = 0; j < 3; j++) {
            ret[i] += v[j] * m(j, i);
        }
    }
    return ret;
}

template <typename T>
DEVICE
inline TVector3<T> operator*(const TMatrix3x3<T> &m, const TVector3<T> &v) {
    TVector3<T> ret;
    for (int i = 0; i < 3; i++) {
        ret[i] = 0.f;
        for (int j = 0; j < 3; j++) {
            ret[i] += m(i, j) * v[j];
        }
    }
    return ret;
}

template <typename T>
DEVICE
inline TMatrix3x3<T> inverse(const TMatrix3x3<T> &m) {
    // computes the inverse of a matrix m
    auto det = m(0, 0) * (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) -
               m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
               m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));

    auto invdet = 1 / det;

    auto m_inv = TMatrix3x3<T>{};
    m_inv(0, 0) = (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) * invdet;
    m_inv(0, 1) = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * invdet;
    m_inv(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * invdet;
    m_inv(1, 0) = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * invdet;
    m_inv(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * invdet;
    m_inv(1, 2) = (m(1, 0) * m(0, 2) - m(0, 0) * m(1, 2)) * invdet;
    m_inv(2, 0) = (m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1)) * invdet;
    m_inv(2, 1) = (m(2, 0) * m(0, 1) - m(0, 0) * m(2, 1)) * invdet;
    m_inv(2, 2) = (m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1)) * invdet;
    return m_inv;
}

template <typename T0, typename T1>
DEVICE
inline auto operator+(const TMatrix4x4<T0> &m0, const TMatrix4x4<T1> &m1)
        -> TMatrix4x4<decltype(m0(0, 0) + m1(0, 0))> {
    TMatrix4x4<decltype(m0(0, 0) + m1(0, 0))> m;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            m(i, j) = m0(i, j) + m1(i, j);
        }
    }
    return m;
}

template <typename T>
DEVICE
TMatrix4x4<T> transpose(const TMatrix4x4<T> &m) {
    return TMatrix4x4<T>(m(0, 0), m(1, 0), m(2, 0), m(3, 0),
                         m(0, 1), m(1, 1), m(2, 1), m(3, 1),
                         m(0, 2), m(1, 2), m(2, 2), m(3, 2),
                         m(0, 3), m(1, 3), m(2, 3), m(3, 3));
}

template <typename T>
DEVICE
inline TMatrix4x4<T> operator-(const TMatrix4x4<T> &m0) {
    TMatrix4x4<T> m;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            m(i, j) = -m0(i, j);
        }
    }
    return m;
}

template <typename T>
DEVICE
inline TMatrix4x4<T> operator-(const TMatrix4x4<T> &m0, const TMatrix4x4<T> &m1) {
    TMatrix4x4<T> m;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            m(i, j) = m0(i, j) - m1(i, j);
        }
    }
    return m;
}

template <typename T>
DEVICE
inline TMatrix4x4<T>& operator+=(TMatrix4x4<T> &m0, const TMatrix4x4<T> &m1) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            m0(i, j) += m1(i, j);
        }
    }
    return m0;
}

template <typename T>
DEVICE
inline TMatrix4x4<T>& operator-=(TMatrix4x4<T> &m0, const TMatrix4x4<T> &m1) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            m0(i, j) -= m1(i, j);
        }
    }
    return m0;
}

template <typename T>
DEVICE
inline TMatrix4x4<T> operator*(const TMatrix4x4<T> &m0, const TMatrix4x4<T> &m1) {
    TMatrix4x4<T> m;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                m(i, j) += m0(i, k) * m1(k, j);
            }
        }
    }
    return m;
}

template <typename T>
DEVICE
TMatrix4x4<T> inverse(const TMatrix4x4<T> &m) {
    // https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
    TMatrix4x4<T> inv;

    inv(0, 0) = m(1, 1) * m(2, 2) * m(3, 3) -
                m(1, 1) * m(2, 3) * m(3, 2) -
                m(2, 1) * m(1, 2) * m(3, 3) +
                m(2, 1) * m(1, 3) * m(3, 2) +
                m(3, 1) * m(1, 2) * m(2, 3) -
                m(3, 1) * m(1, 3) * m(2, 2);

    inv(1, 0) = -m(1, 0) * m(2, 2) * m(3, 3) +
                 m(1, 0) * m(2, 3) * m(3, 2) +
                 m(2, 0) * m(1, 2) * m(3, 3) -
                 m(2, 0) * m(1, 3) * m(3, 2) -
                 m(3, 0) * m(1, 2) * m(2, 3) +
                 m(3, 0) * m(1, 3) * m(2, 2);

    inv(2, 0) = m(1, 0) * m(2, 1) * m(3, 3) -
                m(1, 0) * m(2, 3) * m(3, 1) -
                m(2, 0) * m(1, 1) * m(3, 3) +
                m(2, 0) * m(1, 3) * m(3, 1) +
                m(3, 0) * m(1, 1) * m(2, 3) -
                m(3, 0) * m(1, 3) * m(2, 1);

    inv(3, 0) = -m(1, 0) * m(2, 1) * m(3, 2) +
                 m(1, 0) * m(2, 2) * m(3, 1) +
                 m(2, 0) * m(1, 1) * m(3, 2) -
                 m(2, 0) * m(1, 2) * m(3, 1) -
                 m(3, 0) * m(1, 1) * m(2, 2) +
                 m(3, 0) * m(1, 2) * m(2, 1);

    inv(0, 1) = -m(0, 1) * m(2, 2) * m(3, 3) +
                 m(0, 1) * m(2, 3) * m(3, 2) +
                 m(2, 1) * m(0, 2) * m(3, 3) -
                 m(2, 1) * m(0, 3) * m(3, 2) -
                 m(3, 1) * m(0, 2) * m(2, 3) +
                 m(3, 1) * m(0, 3) * m(2, 2);

    inv(1, 1) = m(0, 0) * m(2, 2) * m(3, 3) -
                m(0, 0) * m(2, 3) * m(3, 2) -
                m(2, 0) * m(0, 2) * m(3, 3) +
                m(2, 0) * m(0, 3) * m(3, 2) +
                m(3, 0) * m(0, 2) * m(2, 3) -
                m(3, 0) * m(0, 3) * m(2, 2);

    inv(2, 1) = -m(0, 0) * m(2, 1) * m(3, 3) +
                 m(0, 0) * m(2, 3) * m(3, 1) +
                 m(2, 0) * m(0, 1) * m(3, 3) -
                 m(2, 0) * m(0, 3) * m(3, 1) -
                 m(3, 0) * m(0, 1) * m(2, 3) +
                 m(3, 0) * m(0, 3) * m(2, 1);

    inv(3, 1) = m(0, 0) * m(2, 1) * m(3, 2) -
                m(0, 0) * m(2, 2) * m(3, 1) -
                m(2, 0) * m(0, 1) * m(3, 2) +
                m(2, 0) * m(0, 2) * m(3, 1) +
                m(3, 0) * m(0, 1) * m(2, 2) -
                m(3, 0) * m(0, 2) * m(2, 1);

    inv(0, 2) = m(0, 1) * m(1, 2) * m(3, 3) -
                m(0, 1) * m(1, 3) * m(3, 2) -
                m(1, 1) * m(0, 2) * m(3, 3) +
                m(1, 1) * m(0, 3) * m(3, 2) +
                m(3, 1) * m(0, 2) * m(1, 3) -
                m(3, 1) * m(0, 3) * m(1, 2);

    inv(1, 2) = -m(0, 0) * m(1, 2) * m(3, 3) +
                 m(0, 0) * m(1, 3) * m(3, 2) +
                 m(1, 0) * m(0, 2) * m(3, 3) -
                 m(1, 0) * m(0, 3) * m(3, 2) -
                 m(3, 0) * m(0, 2) * m(1, 3) +
                 m(3, 0) * m(0, 3) * m(1, 2);

    inv(2, 2) = m(0, 0) * m(1, 1) * m(3, 3) -
                m(0, 0) * m(1, 3) * m(3, 1) -
                m(1, 0) * m(0, 1) * m(3, 3) +
                m(1, 0) * m(0, 3) * m(3, 1) +
                m(3, 0) * m(0, 1) * m(1, 3) -
                m(3, 0) * m(0, 3) * m(1, 1);

    inv(3, 2) = -m(0, 0) * m(1, 1) * m(3, 2) +
                 m(0, 0) * m(1, 2) * m(3, 1) +
                 m(1, 0) * m(0, 1) * m(3, 2) -
                 m(1, 0) * m(0, 2) * m(3, 1) -
                 m(3, 0) * m(0, 1) * m(1, 2) +
                 m(3, 0) * m(0, 2) * m(1, 1);

    inv(0, 3) = -m(0, 1) * m(1, 2) * m(2, 3) +
                 m(0, 1) * m(1, 3) * m(2, 2) +
                 m(1, 1) * m(0, 2) * m(2, 3) -
                 m(1, 1) * m(0, 3) * m(2, 2) -
                 m(2, 1) * m(0, 2) * m(1, 3) +
                 m(2, 1) * m(0, 3) * m(1, 2);

    inv(1, 3) = m(0, 0) * m(1, 2) * m(2, 3) -
                m(0, 0) * m(1, 3) * m(2, 2) -
                m(1, 0) * m(0, 2) * m(2, 3) +
                m(1, 0) * m(0, 3) * m(2, 2) +
                m(2, 0) * m(0, 2) * m(1, 3) -
                m(2, 0) * m(0, 3) * m(1, 2);

    inv(2, 3) = -m(0, 0) * m(1, 1) * m(2, 3) +
                 m(0, 0) * m(1, 3) * m(2, 1) +
                 m(1, 0) * m(0, 1) * m(2, 3) -
                 m(1, 0) * m(0, 3) * m(2, 1) -
                 m(2, 0) * m(0, 1) * m(1, 3) +
                 m(2, 0) * m(0, 3) * m(1, 1);

    inv(3, 3) = m(0, 0) * m(1, 1) * m(2, 2) -
                m(0, 0) * m(1, 2) * m(2, 1) -
                m(1, 0) * m(0, 1) * m(2, 2) +
                m(1, 0) * m(0, 2) * m(2, 1) +
                m(2, 0) * m(0, 1) * m(1, 2) -
                m(2, 0) * m(0, 2) * m(1, 1);

    auto det = m(0, 0) * inv(0, 0) +
               m(0, 1) * inv(1, 0) +
               m(0, 2) * inv(2, 0) +
               m(0, 3) * inv(3, 0);

    if (det == 0) {
        return TMatrix4x4<T>{};
    }

    auto inv_det = 1.0 / det;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            inv(i, j) *= inv_det;
        }
    }

    return inv;
}

template <typename T>
inline std::ostream& operator<<(std::ostream &os, const TMatrix3x3<T> &m) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            os << m(i, j) << " ";
        }
        os << std::endl;
    }
    return os;
}

template <typename T>
inline std::ostream& operator<<(std::ostream &os, const TMatrix4x4<T> &m) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            os << m(i, j) << " ";
        }
        os << std::endl;
    }
    return os;
}
