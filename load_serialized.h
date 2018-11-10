#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

/**
 *  Mitsuba's serialized mesh loader.
 *  Code adapted from https://github.com/mitsuba-renderer/mitsuba/blob/master/src/librender/trimesh.cpp#L175
 *
 */

struct MitsubaTriMesh {
    pybind11::array_t<float> vertices;
    pybind11::array_t<int> indices;
    pybind11::array_t<float> uvs;
    pybind11::array_t<float> normals;
};

MitsubaTriMesh load_serialized(const std::string &filename, int idx);
