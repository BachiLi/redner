import pyredner
import torch

class Material:
    def __init__(self,
                 diffuse_reflectance,
                 specular_reflectance = None,
                 roughness = None,
                 normal_map = None,
                 two_sided = False):
        if specular_reflectance is None:
            specular_reflectance = pyredner.Texture(\
                torch.tensor([0.0,0.0,0.0], device = pyredner.get_device()))
        if roughness is None:
            roughness = pyredner.Texture(\
                torch.tensor([1.0], device = pyredner.get_device()))

        # Convert to constant texture if necessary
        if isinstance(diffuse_reflectance, torch.Tensor):
            diffuse_reflectance = pyredner.Texture(diffuse_reflectance)
        if isinstance(specular_reflectance, torch.Tensor):
            specular_reflectance = pyredner.Texture(specular_reflectance)
        if isinstance(roughness, torch.Tensor):
            roughness = pyredner.Texture(roughness)
        if normal_map is not None and isinstance(normal_map, torch.Tensor):
            normal_map = pyredner.Texture(normal_map)

        self.diffuse_reflectance = diffuse_reflectance
        self.specular_reflectance = specular_reflectance
        self.roughness = roughness
        self.normal_map = normal_map
        self.two_sided = two_sided

    def state_dict(self):
        return {
            'diffuse_reflectance': self.diffuse_reflectance.state_dict(),
            'specular_reflectance': self.specular_reflectance.state_dict(),
            'roughness': self.roughness.state_dict(),
            'normal_map': self.normal_map.state_dict() if self.normal_map is not None else None,
            'two_sided': self.two_sided,
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        normal_map = state_dict['normal_map']
        out = cls(
            pyredner.Texture.load_state_dict(state_dict['diffuse_reflectance']),
            pyredner.Texture.load_state_dict(state_dict['specular_reflectance']),
            pyredner.Texture.load_state_dict(state_dict['roughness']),
            pyredner.Texture.load_state_dict(normal_map) if normal_map is not None else None,
            state_dict['two_sided'])
        return out
