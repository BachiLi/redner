import pyredner

class Shape:
    def __init__(self, vertices, indices, uvs, normals, mat_id):
        assert(vertices.is_contiguous())
        assert(indices.is_contiguous())
        assert(uvs is None or uvs.is_contiguous())
        assert(normals is None or normals.is_contiguous())
        if pyredner.get_use_gpu():
            assert(vertices.is_cuda)
            assert(indices.is_cuda)        
            assert(uvs is None or uvs.is_cuda)
            assert(normals is None or normals.is_cuda)
        else:
            assert(not vertices.is_cuda)
            assert(not indices.is_cuda)        
            assert(uvs is None or not uvs.is_cuda)
            assert(normals is None or not normals.is_cuda)

        self.vertices = vertices
        self.indices = indices
        self.uvs = uvs
        self.normals = normals
        self.mat_id = mat_id
        self.light_id = -1
