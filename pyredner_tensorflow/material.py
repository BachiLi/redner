import pyredner_tensorflow as pyredner
import tensorflow as tf

class Material:
    def __init__(self,
                 diffuse_reflectance,
                 specular_reflectance = None,
                 roughness = None,
                 normal_map = None,
                 two_sided = False):
        assert(tf.executing_eagerly())
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
        if normal_map is not None and tf.is_tensor(normal_map):
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
            'normal_map': self.normal_map.state_dict(),
            'two_sided': self.two_sided,
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        out = cls(
            pyredner.Texture.load_state_dict(state_dict['diffuse_reflectance']),
            pyredner.Texture.load_state_dict(state_dict['specular_reflectance']),
            pyredner.Texture.load_state_dict(state_dict['roughness']),
            pyredner.Texture.load_state_dict(normal_map) if normal_map is not None else None,
            state_dict['two_sided'])
        return out
