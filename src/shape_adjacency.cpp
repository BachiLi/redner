#include "shape_adjacency.h"

/* 
 * Computes vertex and edge adjacency information.
 * This is simply a list of faces adjacent to each vertex and
 * half-edges adjacent to each other (represented using an ordered pair of vertices).
 * The half-edge adjacency is used to determine if the edge is interior or boundary (for open meshes)
 * 
 * This method is pretty slow, but is only called once per run.
 */ 
void compute_adjacency(const Shape* shape, ShapeAdjacency* adjacency) {
    // TODO: This is single threaded (but hasn't been a bottleneck so far). Make parallel.
    adjacency->adjacency.resize(shape->num_triangles * 3, -1);

    // Just a simple (fairly naive) algorithm for now..
    std::map<std::pair<int, int>, int> seenlist;
    adjacency->vertex_adjacency.resize(shape->num_vertices, std::vector<int>());
    for(int i = 0; i < shape->num_triangles; i++) {
        adjacency->vertex_adjacency[shape->indices[i * 3 + 0]].push_back(i);
        adjacency->vertex_adjacency[shape->indices[i * 3 + 1]].push_back(i);
        adjacency->vertex_adjacency[shape->indices[i * 3 + 2]].push_back(i);

        for(int j = 0;j < 3; j++) {
            auto idx0 = shape->indices[i * 3 + j];
            auto idx1 = shape->indices[i * 3 + (j + 1) % 3];

            // Can't have two of the same pairs.
            assert(seenlist.count(std::make_pair(idx0, idx1)) == 0);
            if (seenlist.count(std::make_pair(idx1, idx0))) {
                auto adjacent_edge = seenlist[std::make_pair(idx1, idx0)];
                adjacency->adjacency[adjacent_edge] = i * 3 + j;
                adjacency->adjacency[i * 3 + j] = adjacent_edge;
            }

            seenlist[std::make_pair(idx0, idx1)] = i * 3 + j;
        }

    }
}