#pragma once

#include "redner.h"
#include "ptr.h"
#include "../xatlas/xatlas.h"

#include <vector>

struct UVTriMesh {
    ptr<float> vertices;
    ptr<int> indices;
    ptr<float> uvs;
    ptr<int> uv_indices;
    int num_vertices;
    int num_uv_vertices;
    int num_triangles;
};

struct TextureAtlas {
    TextureAtlas() {
        atlas = xatlas::Create();
    }
    ~TextureAtlas() {
        xatlas::Destroy(atlas);
    }

    xatlas::Atlas *atlas;
};

// Return number of uv vertices required.
std::vector<int> automatic_uv_map(const std::vector<UVTriMesh> &meshes, TextureAtlas &atlas, bool print_progress);
void copy_texture_atlas(const TextureAtlas &atlas, std::vector<UVTriMesh> &meshes);
