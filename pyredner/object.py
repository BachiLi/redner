import pyredner
import torch
from typing import Optional

class Object:
    """
        Object combines geometry, material, and lighting information
        and aggregate them in a single class. This is a convinent class
        for constructing redner scenes.

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
        material: pyredner.Material

        light_intensity: Optional[torch.Tensor]
            make this object an area light
            float32 tensor with size 3
        light_two_sided: boolean
            Does the light emit from two sides of the shape?
        uvs: Optional[torch.Tensor]:
            optional texture coordinates.
            float32 tensor with size num_uvs x 2
            doesn't need to be the same size with vertices if uv_indices is None
        normals: Optional[torch.Tensor]
            shading normal
            float32 tensor with size num_normals x 3
            doesn't need to be the same size with vertices if normal_indices is None
        uv_indices: Optional[torch.Tensor]
            overrides indices when accessing uv coordinates
            int32 tensor with size num_uvs x 3
        normal_indices: Optional[torch.Tensor]
            overrides indices when accessing shading normals
            int32 tensor with size num_normals x 3
        colors: Optional[torch.Tensor]
            optional per-vertex color
            float32 tensor with size num_vertices x 3
        directly_visible: boolean
            optional setting to see if object is visible to camera
            during rendering.
    """
    def __init__(self,
                 vertices: torch.Tensor,
                 indices: torch.Tensor,
                 material: pyredner.Material,
                 light_intensity: Optional[torch.Tensor] = None,
                 light_two_sided: bool = False,
                 uvs: Optional[torch.Tensor] = None,
                 normals: Optional[torch.Tensor] = None,
                 uv_indices: Optional[torch.Tensor] = None,
                 normal_indices: Optional[torch.Tensor] = None,
                 colors: Optional[torch.Tensor] = None,
                 directly_visible: bool = True):
        self.vertices = vertices
        self.indices = indices
        self.uvs = uvs
        self.normals = normals
        self.uv_indices = uv_indices
        self.normal_indices = normal_indices
        self.colors = colors
        self.material = material
        self.light_intensity = light_intensity
        self.light_two_sided = light_two_sided
        self.directly_visible = directly_visible
