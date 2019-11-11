import pyredner
import torch

class Material:
    """
        redner currently employs a two-layer diffuse-specular material model.
        More specifically, it is a linear blend between a Lambertian and
        a microfacet model with Phong distribution.
        It takes either constant color or 2D textures for the reflectances
        and roughness, and an optional normal map texture.
        It can also use vertex color stored in the Shape. In this case
        the model fallback to a diffuse model.

        Args:
            diffuse_reflectance (pyredner.Texture or torch.tensor, optional if use_vertex_color is True)
            specular_reflectance (pyredner.Texture or torch.tensor, optional)
            roughness (pyredner.Texture or torch.tensor, 1 channel, optional)
            generic_texture (pyredner.Texture or torch.tensor, arbitrary number of channels, optional)
            normal_map (pyredner.Texture, 3 channels, optional)
            two_sided (bool) -- By default, the material only reflect lights
                                on the side the normal is pointing to.
                                Set this to True to make the material reflects
                                from both sides.
            use_vertex_color (bool) -- Ignores the reflectances and use the vertex color as diffuse color.
    """
    def __init__(self,
                 diffuse_reflectance = None,
                 specular_reflectance = None,
                 roughness = None,
                 generic_texture = None,
                 normal_map = None,
                 two_sided = False,
                 use_vertex_color = False):
        if diffuse_reflectance is None:
            diffuse_reflectance = pyredner.Texture(\
                torch.tensor([0.0,0.0,0.0], device = pyredner.get_device()))
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
        if generic_texture is not None and isinstance(generic_texture, torch.Tensor):
            generic_texture = pyredner.Texture(generic_texture)
        if normal_map is not None and isinstance(normal_map, torch.Tensor):
            normal_map = pyredner.Texture(normal_map)

        self.diffuse_reflectance = diffuse_reflectance
        self.specular_reflectance = specular_reflectance
        self.roughness = roughness
        self.generic_texture = generic_texture
        self.normal_map = normal_map
        self.two_sided = two_sided
        self.use_vertex_color = use_vertex_color

    def state_dict(self):
        return {
            'diffuse_reflectance': self.diffuse_reflectance.state_dict(),
            'specular_reflectance': self.specular_reflectance.state_dict(),
            'roughness': self.roughness.state_dict(),
            'generic_texture': self.generic_texture.state_dict(),
            'normal_map': self.normal_map.state_dict() if self.normal_map is not None else None,
            'two_sided': self.two_sided,
            'use_vertex_color': self.use_vertex_color
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        normal_map = state_dict['normal_map']
        out = cls(
            pyredner.Texture.load_state_dict(state_dict['diffuse_reflectance']),
            pyredner.Texture.load_state_dict(state_dict['specular_reflectance']),
            pyredner.Texture.load_state_dict(state_dict['roughness']),
            pyredner.Texture.load_state_dict(state_dict['generic_texture']),
            pyredner.Texture.load_state_dict(normal_map) if normal_map is not None else None,
            state_dict['two_sided'],
            state_dict['use_vertex_color'])
        return out
