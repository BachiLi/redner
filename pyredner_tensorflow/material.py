import pyredner_tensorflow as pyredner
import tensorflow as tf

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
            diffuse_reflectance (pyredner.Texture, optional if use_vertex_color is True)
            specular_reflectance (pyredner.Texture, optional)
            roughness (pyredner.Texture, 1 channel, optional)
            generic_texture (pyredner.Texture or torch.tensor, optional)
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
        assert(tf.executing_eagerly())
        if diffuse_reflectance is None:
            diffuse_reflectance = pyredner.Texture(tf.zeros([3], dtype=tf.float32))
        if specular_reflectance is None:
            specular_reflectance = pyredner.Texture(tf.zeros([3], dtype=tf.float32))
        if roughness is None:
            roughness = pyredner.Texture(tf.ones([1], dtype=tf.float32))

        # Convert to constant texture if necessary
        if tf.is_tensor(diffuse_reflectance):
            diffuse_reflectance = pyredner.Texture(diffuse_reflectance)
        if tf.is_tensor(specular_reflectance):
            specular_reflectance = pyredner.Texture(specular_reflectance)
        if tf.is_tensor(roughness):
            roughness = pyredner.Texture(roughness)
        if generic_texture is not None and tf.is_tensor(generic_texture):
            generic_texture = pyredner.Texture(generic_texture)
        if normal_map is not None and tf.is_tensor(normal_map):
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
            'normal_map': self.normal_map.state_dict(),
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
            pyredner.Texture.load_state_dict(generic_texture) if generic_texture is not None else None,
            pyredner.Texture.load_state_dict(normal_map) if normal_map is not None else None,
            state_dict['two_sided'],
            state_dict['use_vertex_color'])
        return out
