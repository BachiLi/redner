#pragma once

#include "redner.h"
#include "shape.h"

#include <map>

/*
 * Shape adjacency data structure. Used for warp field rendering to compute
 * a more useful warp field
 */
struct ShapeAdjacency {
    ShapeAdjacency() {}
    ShapeAdjacency(ptr<Shape> shape) :
        shape(shape.get()) {}

    std::vector<int> adjacency; // triangle-edge -> triangle
    std::vector<std::vector<int>> vertex_adjacency; // vertex-id -> face_list
    Shape* shape;
};

void compute_adjacency(const Shape* shape, ShapeAdjacency* adjacency);