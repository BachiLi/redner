#include "shape_adjacency.h"

// Computes the adjacency.
// This is pretty slow, call only once per run.
void compute_adjacency(const Shape* shape, ShapeAdjacency* adjacency) {
    // TODO: Replace with parallel version.
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
                //std::cout << "Found adjacency " << adjacent_edge << "<->"  << i * 3 + j << std::endl;
            }

            seenlist[std::make_pair(idx0, idx1)] = i * 3 + j;
        }

    }
    /*
    std::cout << "" << shape->indices[ * 3 + _j]
    for(int i = 0; i < shape->num_triangles; i++) {
        for(int j = 0; j < 3; j++)
            if(adjacency->adjacency[i * 3 + j] == -1){
                auto idx0 = shape->indices[i * 3 + j];
                auto idx1 = shape->indices[i * 3 + (j + 1) % 3];
                int _j = 3;
                int _i = 0;
                for(_i = 0; _i < shape->num_triangles && _j == 3; _i++)
                    for(_j = 0; _j < 3; _j++)
                    {
                        if(shape->indices[_i * 3 + _j] == idx1 &&
                            shape->indices[_i * 3 + (_j + 1) % 3] == idx0)
                            {
                                break;
                            }
                    }
                std::cout << "Couldn't find pair for triangle " 
                          << i << ", indices: " << idx0 << ", " << idx1;

                if(_j != 3)
                    std::cout << ", brute force: " << _i << ", " << _j << std::endl;
                else
                    std::cout << ", brute force found nothing." << std::endl;
            }
    }*/

    // Debuggin..
    /*std::cout << "Adjacency" << std::endl;
    for(int i = 0; i < shape->num_triangles; i++) {
        std::cout << adjacency->adjacency[i * 3 + 0] << ", "
                  << adjacency->adjacency[i * 3 + 1] << ", "
                  << adjacency->adjacency[i * 3 + 2] << ", "
                  << std::endl;
    }*/
}