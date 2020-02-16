#include "automatic_uv_map.h"
#include <ctime>
#include <cstdio>
#include <cstdarg>
#include <stdexcept>

class Stopwatch {
public:
    Stopwatch() { reset(); }
    void reset() { m_start = clock(); }
    double elapsed() const { return (clock() - m_start) * 1000.0 / CLOCKS_PER_SEC; }
private:
    clock_t m_start;
};

static int Print(const char *format, ...) {
    va_list arg;
    va_start(arg, format);
    printf("\r"); // Clear progress text (PrintProgress).
    const int result = vprintf(format, arg);
    va_end(arg);
    return result;
}

static void PrintProgress(const char *name, const char *indent1, const char *indent2, int progress, Stopwatch *stopwatch) {
    if (progress == 0)
        stopwatch->reset();
    printf("\r%s%s [", indent1, name);
    for (int i = 0; i < 10; i++)
        printf(progress / ((i + 1) * 10) ? "*" : " ");
    printf("] %d%%", progress);
    fflush(stdout);
    if (progress == 100)
        printf("\n%s%.2f seconds (%g ms) elapsed\n", indent2, stopwatch->elapsed() / 1000.0, stopwatch->elapsed());
}

static bool ProgressCallback(xatlas::ProgressCategory::Enum category, int progress, void *userData) {
    Stopwatch *stopwatch = (Stopwatch *)userData;
    PrintProgress(xatlas::StringForEnum(category), "   ", "      ", progress, stopwatch);
    return true;
}

std::vector<int> automatic_uv_map(const std::vector<UVTriMesh> &meshes, TextureAtlas &atlas, bool print_progress) {
    xatlas::SetPrint(Print, print_progress);
    Stopwatch stopwatch;
    if (print_progress) {
        xatlas::SetProgressCallback(atlas.atlas, ProgressCallback, &stopwatch);
    }
    // Add meshes to atlas.
    for (int i = 0; i < (int)meshes.size(); i++) {
        const UVTriMesh &mesh = meshes[i];
        xatlas::MeshDecl meshDecl;
        meshDecl.vertexCount = mesh.num_vertices;
        meshDecl.vertexPositionData = mesh.vertices.get();
        meshDecl.vertexPositionStride = sizeof(float) * 3;
        if (mesh.uvs.get()) {
            meshDecl.vertexUvData = mesh.uvs.get();
            meshDecl.vertexUvStride = sizeof(float) * 2;
        }
        meshDecl.indexCount = mesh.num_triangles * 3;
        meshDecl.indexData = mesh.indices.get();
        meshDecl.indexFormat = xatlas::IndexFormat::UInt32;
        xatlas::AddMeshError::Enum error = xatlas::AddMesh(atlas.atlas, meshDecl, meshes.size());
        if (error != xatlas::AddMeshError::Success) {
            char buf[256];
            sprintf(buf, "\rError adding mesh %d: %s\n", i, xatlas::StringForEnum(error));
            throw std::runtime_error(buf);
        }
    }
    xatlas::AddMeshJoin(atlas.atlas); // Not necessary. Only called here so geometry totals are printed after the AddMesh progress indicator.
    if (print_progress) {
        printf("Generating atlas\n");
    }
    xatlas::Generate(atlas.atlas);
    if (print_progress) {
        printf("Atlas generation done\n");
    }
    std::vector<int> num_uv_vertices(meshes.size());
    for (uint32_t i = 0; i < atlas.atlas->meshCount; i++) {
        const xatlas::Mesh &mesh = atlas.atlas->meshes[i];
        num_uv_vertices[i] = mesh.vertexCount;
    }
    return num_uv_vertices;
}

void copy_texture_atlas(const TextureAtlas &atlas, std::vector<UVTriMesh> &meshes) {
    for (uint32_t i = 0; i < atlas.atlas->meshCount; i++) {
        const xatlas::Mesh &mesh = atlas.atlas->meshes[i];
        float *uv_target = meshes[i].uvs.get();
        for (uint32_t v = 0; v < mesh.vertexCount; v++) {
            const xatlas::Vertex &vertex = mesh.vertexArray[v];
            uv_target[2 * v + 0] = vertex.uv[0] / atlas.atlas->width;
            uv_target[2 * v + 1] = vertex.uv[1] / atlas.atlas->height;
        }
        int *ind_target = meshes[i].uv_indices.get();
        for (uint32_t f = 0; f < mesh.indexCount; f++) {
            ind_target[f] = mesh.indexArray[f];
        }
    }
}
