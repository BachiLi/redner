import pyrednertensorflow as pyredner
import tensorflow as tf
tfe = tf.contrib.eager

class Material:
    def __init__(self,
                 diffuse_reflectance,
                 specular_reflectance = None,
                 roughness = None,
                 two_sided = False):
        if specular_reflectance is None:
            specular_reflectance = pyredner.Texture(
                tfe.Variable([0.0,0.0,0.0]))
        if roughness is None:
            roughness = pyredner.Texture(
                tfe.Variable([1.0]))
        # import pdb; pdb.set_trace()
        # Convert to constant texture if necessary
        if pyredner.is_tensor(diffuse_reflectance):
            diffuse_reflectance = pyredner.Texture(diffuse_reflectance)
        if pyredner.is_tensor(specular_reflectance):
            specular_reflectance = pyredner.Texture(specular_reflectance)
        if pyredner.is_tensor(roughness):
            roughness = pyredner.Texture(roughness)

        # assert(diffuse_reflectance.texels.dtype == tf.float32)
        # assert(specular_reflectance.texels.dtype == tf.float32)
        # assert(roughness.texels.dtype == tf.float32)

        self.diffuse_reflectance = diffuse_reflectance
        self.specular_reflectance = specular_reflectance
        self.roughness = roughness
        self.two_sided = two_sided

    def state_dict(self):
        return {
            'diffuse_reflectance': self.diffuse_reflectance.state_dict(),
            'specular_reflectance': self.specular_reflectance.state_dict(),
            'roughness': self.roughness.state_dict(),
            'two_sided': self.two_sided,
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        out = cls(
            pyredner.Texture.load_state_dict(state_dict['diffuse_reflectance']),
            pyredner.Texture.load_state_dict(state_dict['specular_reflectance']),
            pyredner.Texture.load_state_dict(state_dict['roughness']),
            state_dict['two_sided'])
        return out
