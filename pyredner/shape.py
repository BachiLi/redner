import pyredner
import torch
import math

def compute_vertex_normal(vertices, indices):
    def dot(v1, v2):
        return torch.sum(v1 * v2, dim = 1)
    def squared_length(v):
        return torch.sum(v * v, dim = 1)
    def length(v):
        return torch.sqrt(squared_length(v))
    # Nelson Max, "Weights for Computing Vertex Normals from Facet Vectors", 1999
    normals = torch.zeros(vertices.shape, dtype = torch.float32, device = vertices.device)
    v = [vertices[indices[:, 0].long(), :],
         vertices[indices[:, 1].long(), :],
         vertices[indices[:, 2].long(), :]]
    for i in range(3):
        v0 = v[i]
        v1 = v[(i + 1) % 3]
        v2 = v[(i + 2) % 3]
        e1 = v1 - v0
        e2 = v2 - v0
        e1_len = length(e1)
        e2_len = length(e2)
        side_a = e1 / torch.reshape(e1_len, [-1, 1])
        side_b = e2 / torch.reshape(e2_len, [-1, 1])
        if i == 0:
            n = torch.cross(side_a, side_b)
            n = n / torch.reshape(length(n), [-1, 1])
        angle = torch.where(dot(side_a, side_b) < 0, 
            math.pi - 2.0 * torch.asin(0.5 * length(side_a + side_b)),
            2.0 * torch.asin(0.5 * length(side_b - side_a)))
        sin_angle = torch.sin(angle)
        
        # XXX: Inefficient but it's PyTorch's limitation
        contrib = n * (sin_angle / (e1_len * e2_len)).reshape(-1, 1).expand(-1, 3)
        index = indices[:, i].long().reshape(-1, 1).expand([-1, 3])
        normals.scatter_add_(0, index, contrib)

    normals = normals / torch.reshape(length(normals), [-1, 1])
    return normals.contiguous()

class Shape:
    def __init__(self, vertices, indices, uvs, normals, material_id):
        assert(vertices.dtype == torch.float32)
        assert(indices.dtype == torch.int32)
        assert(vertices.is_contiguous())
        assert(indices.is_contiguous())
        if (uvs is not None):
            assert(uvs.dtype == torch.float32)
            assert(uvs.is_contiguous())
        if (normals is not None):
            assert(normals.dtype == torch.float32)
            assert(normals.is_contiguous())
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
        self.material_id = material_id
        self.light_id = -1

    def state_dict(self):
        return {
            'vertices': self.vertices,
            'indices': self.indices,
            'uvs': self.uvs,
            'normals': self.normals,
            'material_id': self.material_id,
            'light_id': self.light_id,
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        out = cls(
            state_dict['vertices'],
            state_dict['indices'],
            state_dict['uvs'],
            state_dict['normals'],
            state_dict['material_id'])
        return out
