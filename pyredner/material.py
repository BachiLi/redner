import pyredner
import torch
from typing import Union, Optional

class Material:
    """
        redner currently employs a two-layer diffuse-specular material model.
        More specifically, it is a linear blend between a Lambertian model and
        a microfacet model with Phong distribution, with Schilick's Fresnel approximation.
        It takes either constant color or 2D textures for the reflectances
        and roughness, and an optional normal map texture.
        It can also use vertex color stored in the Shape. In this case
        the model fallback to a diffuse model.

        Args
        ====
        diffuse_reflectance: Optional[Union[torch.Tensor, pyredner.Texture]]
            a float32 tensor with size 3 or [height, width, 3] or a Texture
            optional if use_vertex_color is True
        specular_reflectance: Optional[Union[torch.Tensor, pyredner.Texture]]
            a float32 tensor with size 3 or [height, width, 3] or a Texture
        roughness: Optional[Union[torch.Tensor, pyredner.Texture]]
            a float32 tensor with size 1 or [height, width, 1] or a Texture
        generic_texture: Optional[Union[torch.Tensor, pyredner.Texture]]
            a float32 tensor with dimension 1 or 3, arbitrary number of channels
            use render_g_buffer to visualize this texture
        normal_map: Optional[Union[torch.Tensor, pyredner.Texture]]
            a float32 tensor with size 3 or [height, width, 3] or a Texture
        two_sided: bool
            By default, the material only reflect lights on the side the
            normal is pointing to.
            Set this to True to make the material reflects from both sides.
        use_vertex_color: bool
            ignores the reflectances and use the vertex color as diffuse color
    """
    def __init__(self,
                 diffuse_reflectance: Optional[Union[torch.Tensor, pyredner.Texture]] = None,
                 specular_reflectance: Optional[Union[torch.Tensor, pyredner.Texture]] = None,
                 roughness: Optional[Union[torch.Tensor, pyredner.Texture]] = None,
                 generic_texture: Optional[Union[torch.Tensor, pyredner.Texture]] = None,
                 normal_map: Optional[Union[torch.Tensor, pyredner.Texture]] = None,
                 two_sided: bool = False,
                 use_vertex_color: bool = False):
        if diffuse_reflectance is None:
            diffuse_reflectance = pyredner.Texture(\
                torch.zeros(3, device = pyredner.get_device()))
        if specular_reflectance is None:
            specular_reflectance = pyredner.Texture(\
                torch.zeros(3, device = pyredner.get_device()))
            compute_specular_lighting = False
        else:
            compute_specular_lighting = True
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

        assert((len(diffuse_reflectance.texels.shape) == 1 and diffuse_reflectance.texels.shape[0] == 3) or \
               (len(diffuse_reflectance.texels.shape) == 3 and diffuse_reflectance.texels.shape[2] == 3))
        assert((len(specular_reflectance.texels.shape) == 1 and specular_reflectance.texels.shape[0] == 3) or \
               (len(specular_reflectance.texels.shape) == 3 and specular_reflectance.texels.shape[2] == 3))
        assert((len(roughness.texels.shape) == 1 and roughness.texels.shape[0] == 1) or \
               (len(roughness.texels.shape) == 3 and roughness.texels.shape[2] == 1))
        if normal_map is not None:
            assert((len(normal_map.texels.shape) == 1 and normal_map.texels.shape[0] == 3) or \
                   (len(normal_map.texels.shape) == 3 and normal_map.texels.shape[2] == 3))

        self.diffuse_reflectance = diffuse_reflectance
        self._specular_reflectance = specular_reflectance
        self.compute_specular_lighting = compute_specular_lighting
        self.roughness = roughness
        self.generic_texture = generic_texture
        self.normal_map = normal_map
        self.two_sided = two_sided
        self.use_vertex_color = use_vertex_color

    @property
    def specular_reflectance(self):
        return self._specular_reflectance

    @specular_reflectance.setter
    def specular_reflectance(self, value):
        self._specular_reflectance = value
        if value is not None:
            self.compute_specular_lighting = True
        else:
            self._specular_reflectance = pyredner.Texture(\
                torch.zeros(3, device = pyredner.get_device()))
            self.compute_specular_lighting = False

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
