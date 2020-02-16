#pragma once

#include "redner.h"
#include "ptr.h"

#include <algorithm>

/// Merge duplicated vertices unless the dihedral angle is larger than max_smooth_angle
/// Implementation taken from Mitsuba https://github.com/mitsuba-renderer/mitsuba/blob/master/src/librender/trimesh.cpp#L468
/// Assume the pointers are in cpu memory, since the algorithm is serial at the moment
/// Returns the new number of vertices
int rebuild_topology(
    ptr<float> vertices,
    ptr<int> indices,
    ptr<float> uvs,
    ptr<float> normals,
    ptr<int> uv_indices,
    int num_vertices,
    int num_triangles,
    float max_smooth_angle);

