import pyredner
import torch
from typing import Optional

class Object:
    """
        Object combines geometry, material, and lighting information
        and aggregate them in a single class. This is a convinent class
        for constructing redner scenes.
        Args:
            vertices (float tensor with size N x 3): 3D position of vertices.
            indices (int tensor with size M x 3): vertex indices of triangle faces.
            material (Material): the material
            light_intensity (optional, float tensor with size 3): make this object an area light
            light_two_sided (boolean): does the light emit from two sides of the shape
            uvs (optional, float tensor with size N' x 2): optional texture coordinates.
            normals (optional, float tensor with size N'' x 3): shading normal.
            uv_indices (optional, int tensor with size M x 3): overrides indices when accessing uv coordinates.
            normal_indices (optional, int tensor with size M x 3): overrides indices when accessing shading normals.
            colors (optional, float tensor with size N x 3): optional vertex color.
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
                 colors: Optional[torch.Tensor] = None):
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
