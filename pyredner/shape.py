import pyredner
import torch
import math
import redner
from typing import Optional

def compute_vertex_normal(vertices: torch.Tensor,
                          indices: torch.Tensor,
                          weighting_scheme: str = 'max'):
    """
        Compute vertex normal by weighted average of nearby face normals.
        Args
        ====
        vertices: torch.Tensor
            3D position of vertices.
            float32 tensor with size num_vertices x 3
        indices: torch.Tensor
            Vertex indices of triangle faces.
            int32 tensor with size num_triangles x 3
        weighting_scheme: str
            How do we compute the weighting. Currently we support two weighting methods:
            'max' and 'cotangent'.
            'max' corresponds to Nelson Max's algorithm that uses the inverse length and sine of the angle as the weight
            (see `Weights for Computing Vertex Normals from Facet Vectors <https://escholarship.org/content/qt7657d8h3/qt7657d8h3.pdf?t=ptt283>`_),
            'cotangent' corresponds to weights derived through a discretization of the gradient of triangle area
            (see, e.g., "Implicit Fairing of Irregular Meshes using Diffusion and Curvature Flow" from Desbrun et al.)

        Returns
        =======
        torch.Tensor
            float32 Tensor with size num_vertices x 3 representing vertex normal
    """

    def dot(v1, v2):
        return torch.sum(v1 * v2, dim = 1)
    def squared_length(v):
        return torch.sum(v * v, dim = 1)
    def length(v):
        return torch.sqrt(squared_length(v))
    def safe_asin(v):
        # Hack: asin(1)' is infinite, so we want to clamp the contribution
        return torch.asin(v.clamp(0, 1-1e-6))

    # XXX: This whole thing is inefficient but it's PyTorch's limitation

    normals = torch.zeros(vertices.shape, dtype = torch.float32, device = vertices.device)
    v = [vertices[indices[:, 0].long(), :],
         vertices[indices[:, 1].long(), :],
         vertices[indices[:, 2].long(), :]]
    if weighting_scheme == 'max':
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
            # numerically stable angle between two unit direction vectors
            # http://www.plunk.org/~hatch/rightway.php
            angle = torch.where(dot(side_a, side_b) < 0,
                torch.tensor(math.pi) - 2.0 * safe_asin(0.5 * length(side_a + side_b)),
                2.0 * safe_asin(0.5 * length(side_b - side_a)))
            sin_angle = torch.sin(angle)
            e1e2 = e1_len * e2_len
            # contrib is 0 when e1e2 is 0
            contrib = torch.where(e1e2.reshape(-1, 1).expand(-1, 3) > 0,
                n * (sin_angle / e1e2).reshape(-1, 1).expand(-1, 3),
                torch.zeros(n.shape, dtype=torch.float32, device=vertices.device))
            index = indices[:, i].long().reshape(-1, 1).expand(-1, 3)
            normals.scatter_add_(0, index, contrib)
        # Assign 0, 0, 1 to degenerate faces
        degenerate_normals = torch.zeros(normals.shape, dtype = torch.float32, device = vertices.device)
        degenerate_normals[:, 2] = 1.0
        normals = torch.where(length(normals).reshape(-1, 1).expand(-1, 3) > 0,
            normals / torch.reshape(length(normals), [-1, 1]),
            degenerate_normals)
    elif weighting_scheme == 'cotangent':
        # Cotangent weighting generates 0-length normal when
        # the local surface is planar. Prepare weighted average normal
        # computed using Nelson Max's algorithm for those cases.
        max_normal = compute_vertex_normal(vertices, indices, 'max')
        for i in range(3):
            # Loop over each pair of edges sharing the same vertex,
            # compute the cotangent and contribute to the third edge.
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
            # numerically stable angle between two unit direction vectors
            # http://www.plunk.org/~hatch/rightway.php
            angle = torch.where(dot(side_a, side_b) < 0,
                torch.tensor(math.pi) - 2.0 * safe_asin(0.5 * length(side_a + side_b)),
                2.0 * safe_asin(0.5 * length(side_b - side_a)))
            cotangent = torch.tensor(1.0) / torch.tan(angle)
            v1_index = indices[:, (i + 1) % 3].long().reshape(-1, 1).expand(-1, 3)
            v2_index = indices[:, (i + 2) % 3].long().reshape(-1, 1).expand(-1, 3)
            contrib = (v2 - v1) * cotangent.reshape([-1, 1])
            normals.scatter_add_(0, v1_index, contrib)
            normals.scatter_add_(0, v2_index, -contrib)
        # Make sure the normals are pointing at the right direction
        normals = torch.where(dot(normals, max_normal).reshape(-1, 1).expand(-1, 3) > 0, normals, -normals)
        normals = torch.where(length(normals).reshape(-1, 1).expand(-1, 3) > 0.05,
            normals / torch.reshape(length(normals), [-1, 1]),
            max_normal)
    else:
        assert False, 'Unknown weighting scheme: {}'.format(weighting_scheme)

    assert(torch.isfinite(normals).all())
    return normals.contiguous()


def bound_vertices(vertices: torch.Tensor, indices: torch.Tensor):
    """
        Calculate the indices of boundary vertices of a mesh
        and express it in Tensor form.

        Args
        ====
        vertices: torch.Tensor
            3D position of vertices.
            float32 tensor with size num_vertices x 3
        indices: torch.Tensor
            Vertex indices of triangle faces.
            int32 tensor with size num_triangles x 3

        Returns
        =======
        bound: torch.Tensor
            float32 Tensor with size num_vertices representing vertex normal
            bound[i] = 0. if i-th vertices is on boundary of mesh; else 1.
    """
    neighbor_sum = torch.zeros(vertices.size(0), device=vertices.device)
    for i in range(3):
        contrib = indices[:, (i + 2) % 3] - indices[:, (i + 1) % 3]
        index = indices[:, i].long()
        neighbor_sum.scatter_add_(0, index, contrib.float())
        # neighbor_sum[index[i]] += contrib[i]
    return torch.where(neighbor_sum == 0,
                       torch.ones(vertices.size(0), device=vertices.device),
                       torch.zeros(vertices.size(0), device=vertices.device))

def smooth(vertices: torch.Tensor,
                     indices: torch.Tensor,
                     lmd: torch.float32,
                     weighting_scheme: str = 'reciprocal',
                     control: torch.Tensor = None):
    """
        Update positions of vertices in a mesh. The shift amount of a vertex equals
        to lmd times weight sum of all edges to neighbors.

        $v_i += lmd * \frac {\sum_{j \in neighbors(i)} w_{ij}(v_j - v_i)} {\sum_{j \in neighbors(i)} w_{ij}}$

        Args
        ====
        vertices: torch.Tensor
            3D position of vertices.
            float32 tensor with size num_vertices x 3
        indices: torch.Tensor
            Vertex indices of triangle faces.
            int32 tensor with size num_triangles x 3
        lmd: torch.float32
            step length coefficient
        weighting_scheme: str = 'reciprocal'
            Different weighting schemes:
                'reciprocal': (default)
                    w[i][j] = 1 / len(v[j] - v[i])
                'uniform':
                    w[i][j] = 1
                'cotangent':
                    w[i][j] = cot(angle(i-m-j)) + cot(angle(i-n-j))
                    m and n are vertices that form triangles with i and j
        control: torch.Tensor
            extra coefficient deciding which vertices to be update.
            In default case, do not update boundary vertices of the mesh
                control (default) = bound_vertices(vertices, indices)
            type help(pyredner.bound_vertices)
    """
    if control is None:
        control = bound_vertices(vertices, indices)
    else:
        assert control.numel() == vertices.size(0), 'Size of control tensor inconsistent with number of vertices'

    def dot(v1, v2):
        return torch.sum(v1 * v2, dim=1)

    def squared_length(v):
        return torch.sum(v * v, dim=1)

    def length(v):
        return torch.sqrt(squared_length(v))

    def safe_asin(v):
        # Hack: asin(1)' is infinite, so we want to clamp the contribution
        return torch.asin(v.clamp(0, 1 - 1e-6))


    total_contrib = torch.zeros(vertices.shape, dtype=torch.float32, device=vertices.device)
    total_weight_contrib = torch.zeros(vertices.shape, dtype=torch.float32, device=vertices.device)

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

        # XXX: Inefficient but it's PyTorch's limitation
        e1e2 = e1_len * e2_len
        # contrib is 0 when e1e2 is 0

        if weighting_scheme == 'reciprocal':
            contrib = torch.where(e1e2.reshape(-1, 1).expand(-1, 3) > 0,
                                  e1 / e1_len.reshape(-1, 1).expand(-1, 3) +
                                  e2 / e2_len.reshape(-1, 1).expand(-1, 3),
                                  torch.zeros(v0.shape, dtype=torch.float32, device=vertices.device))
            weight_contrib = torch.where(e1e2.reshape(-1, 1).expand(-1, 3) > 0,
                                         torch.tensor(1.) / e1_len.reshape(-1, 1).expand(-1, 3) +
                                         torch.tensor(1.) / e2_len.reshape(-1, 1).expand(-1, 3),
                                         torch.zeros(v0.shape, dtype=torch.float32, device=vertices.device))
            index = indices[:, i].long().reshape(-1, 1).expand(-1, 3)
            total_contrib.scatter_add_(0, index, contrib)
            total_weight_contrib.scatter_add_(0, index, weight_contrib)
        elif weighting_scheme == 'uniform':
            contrib = torch.where(e1e2.reshape(-1, 1).expand(-1, 3) > 0,
                                  e1 + e2,
                                  torch.zeros(v0.shape, dtype=torch.float32, device=vertices.device))
            weight_contrib = torch.where(e1e2.reshape(-1, 1).expand(-1, 3) > 0,
                                         2 * torch.ones(v0.shape, dtype=torch.float32, device=vertices.device),
                                         torch.zeros(v0.shape, dtype=torch.float32, device=vertices.device))
            index = indices[:, i].long().reshape(-1, 1).expand(-1, 3)
            total_contrib.scatter_add_(0, index, contrib)
            total_weight_contrib.scatter_add_(0, index, weight_contrib)
        elif weighting_scheme == 'cotangent':
            pass
            side_a = e1 / torch.reshape(e1_len, [-1, 1])
            side_b = e2 / torch.reshape(e2_len, [-1, 1])
            angle = torch.where(dot(side_a, side_b) < 0,
                                torch.tensor(math.pi) - 2.0 * safe_asin(0.5 * length(side_a + side_b)),
                                2.0 * safe_asin(0.5 * length(side_b - side_a)))
            cotangent = torch.tensor(1.0) / torch.tan(angle)
            v1_index = indices[:, (i + 1) % 3].long().reshape(-1, 1).expand(-1, 3)
            v2_index = indices[:, (i + 2) % 3].long().reshape(-1, 1).expand(-1, 3)
            contrib = (v2 - v1) * cotangent.reshape([-1, 1])
            weight_contrib = cotangent.reshape([-1, 1]).expand(-1, 3)
            total_contrib.scatter_add_(0, v1_index, contrib)
            total_contrib.scatter_add_(0, v2_index, -contrib)
            total_weight_contrib.scatter_add_(0, v1_index, weight_contrib)
            total_weight_contrib.scatter_add_(0, v2_index, weight_contrib)
        else:
            assert False, 'Unknown weighting_scheme: {}'.format(weighting_scheme)

    shift = total_contrib / total_weight_contrib * control.reshape(-1, 1)
    vertices.data += shift * lmd
    return

def compute_uvs(vertices, indices, print_progress = True):
    """
        Compute UV coordinates of a given mesh using a charting algorithm
        with least square conformal mapping. This calls the `xatlas <https://github.com/jpcy/xatlas>`_ library.
        Args
        ====
        vertices: torch.Tensor
            3D position of vertices
            float32 tensor with size num_vertices x 3
        indices: torch.Tensor
            vertex indices of triangle faces.
            int32 tensor with size num_triangles x 3
        Returns
        =======
        torch.Tensor
            uv vertices pool, float32 Tensor with size num_uv_vertices x 3
        torch.Tensor
            uv indices, int32 Tensor with size num_triangles x 3
    """
    device = vertices.device
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

    vertices = vertices.to(device)
    indices = indices.to(device)
    uvs = uvs.to(device)
    uv_indices = uv_indices.to(device)
    return uvs, uv_indices

class Shape:
    """
        redner supports only triangle meshes for now. It stores a pool of
        vertices and access the pool using integer index. Some times the
        two vertices can have the same 3D position but different texture
        coordinates, because UV mapping creates seams and need to duplicate
        vertices. In this can we can use an additional "uv_indices" array
        to access the uv pool.
        Args
        ====
        vertices: torch.Tensor
            3D position of vertices
            float32 tensor with size num_vertices x 3
        indices: torch.Tensor
            vertex indices of triangle faces.
            int32 tensor with size num_triangles x 3
        uvs: Optional[torch.Tensor]:
            optional texture coordinates.
            float32 tensor with size num_uvs x 2
            doesn't need to be the same size with vertices if uv_indices is not None
        normals: Optional[torch.Tensor]
            shading normal
            float32 tensor with size num_normals x 3
            doesn't need to be the same size with vertices if normal_indices is not None
        uv_indices: Optional[torch.Tensor]
            overrides indices when accessing uv coordinates
            int32 tensor with size num_uvs x 3
        normal_indices: Optional[torch.Tensor]
            overrides indices when accessing shading normals
            int32 tensor with size num_normals x 3
    """
    def __init__(self,
                 vertices: torch.Tensor,
                 indices: torch.Tensor,
                 material_id: int,
                 uvs: Optional[torch.Tensor] = None,
                 normals: Optional[torch.Tensor] = None,
                 uv_indices: Optional[torch.Tensor] = None,
                 normal_indices: Optional[torch.Tensor] = None,
                 colors: Optional[torch.Tensor] = None):
        assert(vertices.dtype == torch.float32)
        assert(vertices.is_contiguous())
        assert(len(vertices.shape) == 2 and vertices.shape[1] == 3)
        assert(indices.dtype == torch.int32)
        assert(indices.is_contiguous())
        assert(len(indices.shape) == 2 and indices.shape[1] == 3)
        if uvs is not None:
            assert(uvs.dtype == torch.float32)
            assert(uvs.is_contiguous())
            assert(len(uvs.shape) == 2 and uvs.shape[1] == 2)
        if normals is not None:
            assert(normals.dtype == torch.float32)
            assert(normals.is_contiguous())
            assert(len(normals.shape) == 2 and normals.shape[1] == 3)
        if uv_indices is not None:
            assert(uv_indices.dtype == torch.int32)
            assert(uv_indices.is_contiguous())
            assert(len(uv_indices.shape) == 2 and uv_indices.shape[1] == 3)
        if normal_indices is not None:
            assert(normal_indices.dtype == torch.int32)
            assert(normal_indices.is_contiguous())
            assert(len(normal_indices.shape) == 2 and normal_indices.shape[1] == 3)
        if colors is not None:
            assert(colors.dtype == torch.float32)
            assert(colors.is_contiguous())
            assert(len(colors.shape) == 2 and colors.shape[1] == 3)

        self.vertices = vertices
        self.indices = indices
        self.material_id = material_id
        self.uvs = uvs
        self.normals = normals
        self.uv_indices = uv_indices
        self.normal_indices = normal_indices
        self.colors = colors
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
            'normal_indices': self.normal_indices,
            'colors': self.colors
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
            state_dict['normal_indices'],
            state_dict['colors'])
        out.light_id = state_dict['light_id']
        return out
