import torch
import numpy as np
import pyredner

class Material:
    def __init__(self,
                 diffuse_reflectance,
                 specular_reflectance = None,
                 roughness = None,
                 diffuse_uv_scale = torch.tensor([1.0, 1.0]),
                 specular_uv_scale = torch.tensor([1.0, 1.0]),
                 roughness_uv_scale = torch.tensor([1.0, 1.0]),
                 two_sided = False):
        assert(diffuse_reflectance.is_contiguous())
        if specular_reflectance is None:
            specular_reflectance = torch.tensor([0.0,0.0,0.0],
                device = pyredner.get_device())
        else:
            assert(specular_reflectance.is_contiguous())
        if roughness is None:
            roughness = torch.tensor([1.0], device = pyredner.get_device())
        else:
            assert(roughness.is_contiguous())
        if pyredner.get_use_gpu():
            assert(diffuse_reflectance.is_cuda)
            assert(specular_reflectance.is_cuda)
            assert(roughness.is_cuda)
        else:
            assert(not diffuse_reflectance.is_cuda)
            assert(not specular_reflectance.is_cuda)
            assert(not roughness.is_cuda)

        self.diffuse_reflectance = diffuse_reflectance
        self.specular_reflectance = specular_reflectance
        self.roughness = roughness
        self.diffuse_uv_scale = diffuse_uv_scale
        self.specular_uv_scale = diffuse_uv_scale
        self.roughness_uv_scale = diffuse_uv_scale
        self.two_sided = two_sided
