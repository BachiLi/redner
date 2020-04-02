#pragma once

#include "vector.h"
#include <cstdio>

inline void equal_or_error(const std::string &file, int line,
                           int expected, int output) {
    if (expected != output) {
        fprintf(stderr, "Test failed at %s, line %d.\n", file.c_str(), line);
        fprintf(stderr, "Expected %d, got %d.\n", expected, output);
        exit(1);
    }
}

template <typename T>
void equal_or_error(const std::string &file, int line,
                    T expected, T output, T tolerance = 1e-3f) {
    if (fabs(expected - output) > tolerance) {
        fprintf(stderr, "Test failed at %s, line %d.\n", file.c_str(), line);
        std::cerr << "Expected " << expected << ", got " << output << std::endl;
        exit(1);
    }
}

template <typename T>
void equal_or_error(const std::string &file, int line,
                    const TVector2<T> &expected,
                    const TVector2<T> &output, T tolerance = 1e-3f) {
    if (fabs(expected[0] - output[0]) > tolerance ||
            fabs(expected[1] - output[1]) > tolerance) {
        fprintf(stderr, "Test failed at %s, line %d.\n", file.c_str(), line);
        std::cerr << "Expected " << expected << ", got " << output << std::endl;
        exit(1);
    }
}

template <typename T>
void equal_or_error(const std::string &file, int line,
                    const TVector3<T> &expected,
                    const TVector3<T> &output, T tolerance = 1e-3f) {
    if (fabs(expected[0] - output[0]) > tolerance ||
            fabs(expected[1] - output[1]) > tolerance ||
            fabs(expected[2] - output[2]) > tolerance) {
        fprintf(stderr, "Test failed at %s, line %d.\n", file.c_str(), line);
        std::cerr << "Expected " << expected << ", got " << output << std::endl;
        exit(1);
    }
}
