import pyredner
import torch
import math
import redner

def compute_vertex_normal(vertices, indices):
    def dot(v1, v2):
        return torch.sum(v1 * v2, dim = 1)
    def squared_length(v):
        return torch.sum(v * v, dim = 1)
    def length(v):
        return torch.sqrt(squared_length(v))
    def safe_asin(v):
        # Hack: asin(1)' is infinite, so we want to clamp the contribution
        return torch.asin(v.clamp(0, 1-1e-6))
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
            n = torch.where(length(n).reshape(-1, 1).expand(-1, 3) > 0,
                n / torch.reshape(length(n), [-1, 1]),
                torch.zeros(n.shape, dtype=n.dtype, device=n.device))
        angle = torch.where(dot(side_a, side_b) < 0, 
            math.pi - 2.0 * safe_asin(0.5 * length(side_a + side_b)),
            2.0 * safe_asin(0.5 * length(side_b - side_a)))
        sin_angle = torch.sin(angle)
        
        # XXX: Inefficient but it's PyTorch's limitation
        e1e2 = e1_len * e2_len
        # contrib is 0 when e1e2 is 0
        contrib = torch.where(e1e2.reshape(-1, 1).expand(-1, 3) > 0,
            n * (sin_angle / e1e2).reshape(-1, 1).expand(-1, 3),
            torch.zeros(n.shape, dtype = torch.float32, device = vertices.device))
        index = indices[:, i].long().reshape(-1, 1).expand(-1, 3)
        normals.scatter_add_(0, index, contrib)

    # Assign 0, 0, 1 to degenerate faces
    degenerate_normals = torch.zeros(normals.shape, dtype = torch.float32, device = vertices.device)
    degenerate_normals[:, 2] = 1.0
    normals = torch.where(length(normals).reshape(-1, 1).expand(-1, 3) > 0,
        normals / torch.reshape(length(normals), [-1, 1]),
        degenerate_normals)
    assert(torch.isfinite(normals).all())
    return normals.contiguous()

def compute_uvs(vertices, indices, print_progress = True):
    """
        Args: vertices -- N x 3 float tensor
              indices -- M x 3 int tensor
        Return: uvs & uvs_indices
    """
    vertices = vertices.cpu()
    indices = indices.cpu()

    uv_trimesh = redner.UVTriMesh(redner.float_ptr(vertices.data_ptr()),
                                  redner.int_ptr(indices.data_ptr()),
                                  redner.float_ptr(0),
                                  redner.int_ptr(0),
                                  int(vertices.shape[0]),
                                  0,
                                  int(indices.shape[0]))

    atlas = redner.TextureAtlas()
    num_uv_vertices = redner.automatic_uv_map([uv_trimesh], atlas, print_progress)[0]

    uvs = torch.zeros(num_uv_vertices, 2, dtype=torch.float32)
    uv_indices = torch.zeros_like(indices)
    uv_trimesh.uvs = redner.float_ptr(uvs.data_ptr())
    uv_trimesh.uv_indices = redner.int_ptr(uv_indices.data_ptr())
    uv_trimesh.num_uv_vertices = num_uv_vertices

    redner.copy_texture_atlas(atlas, [uv_trimesh])

    if pyredner.get_use_gpu():
        vertices = vertices.cuda(device = pyredner.get_device())
        indices = indices.cuda(device = pyredner.get_device())
        uvs = uvs.cuda(device = pyredner.get_device())
        uv_indices = uv_indices.cuda(device = pyredner.get_device())
    return uvs, uv_indices

class Shape:
    """
        redner supports only triangle meshes for now. It stores a pool of
        vertices and access the pool using integer index. Some times the
        two vertices can have the same 3D position but different texture
        coordinates, because UV mapping creates seams and need to duplicate
        vertices. In this can we canta an additional "uv_indices" array
        to access the uv pool.

        Args:
            vertices (float tensor with size N x 3): 3D position of vertices.
            indices (int tensor with size M x 3): vertex indices of triangle faces.
            material_id (integer): index of the assigned material.
            uvs (optional, float tensor with size N' x 2): optional texture coordinates.
            normals (optional, float tensor with size N'' x 3): shading normal.
            uv_indices (optional, int tensor with size M x 3): overrides indices when accessing uv coordinates.
            normal_indices (optional, int tensor with size M x 3): overrides indices when accessing shading normals.
    """
    def __init__(self,
                 vertices,
                 indices,
                 material_id,
                 uvs = None,
                 normals = None,
                 uv_indices = None,
                 normal_indices = None):
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
        if (uv_indices is not None):
            assert(uv_indices.dtype == torch.int32)
            assert(uv_indices.is_contiguous())
        if (normal_indices is not None):
            assert(normal_indices.dtype == torch.int32)
            assert(normal_indices.is_contiguous())

        if pyredner.get_use_gpu():
            assert(vertices.is_cuda)
            assert(indices.is_cuda)
            assert(uvs is None or uvs.is_cuda)
            assert(normals is None or normals.is_cuda)
            assert(uv_indices is None or uv_indices.is_cuda)
            assert(normal_indices is None or normal_indices.is_cuda)
        else:
            assert(not vertices.is_cuda)
            assert(not indices.is_cuda)        
            assert(uvs is None or not uvs.is_cuda)
            assert(normals is None or not normals.is_cuda)
            assert(uv_indices is None or not uv_indices.is_cuda)
            assert(normal_indices is None or not normal_indices.is_cuda)

        self.vertices = vertices
        self.indices = indices
        self.material_id = material_id
        self.uvs = uvs
        self.normals = normals
        self.uv_indices = uv_indices
        self.normal_indices = normal_indices
        self.light_id = -1

    def state_dict(self):
        return {
            'vertices': self.vertices,
            'indices': self.indices,
            'material_id': self.material_id,
            'light_id': self.light_id,
            'uvs': self.uvs,
            'normals': self.normals,
            'uv_indices': self.uv_indices,
            'normal_indices': self.normal_indices
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        out = cls(
            state_dict['vertices'],
            state_dict['indices'],
            state_dict['material_id'],
            state_dict['uvs'],
            state_dict['normals'],
            state_dict['uv_indices'],
            state_dict['normal_indices'])
        out.light_id = state_dict['light_id']
        return out
