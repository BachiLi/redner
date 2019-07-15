import pyrednertensorflow as pyredner
import numpy as np
import tensorflow as tf
import math
import pdb

class EnvironmentMap:
    def __init__(self, values, env_to_world = tf.eye(4, 4)):
        # Convert to constant texture if necessary
        if pyredner.is_tensor(values):
            values = pyredner.Texture(values)

        # assert(values.texels.is_contiguous())
        assert(values.texels.dtype == tf.float32)
        
        if pyredner.get_use_gpu():
            assert(tf.test.is_gpu_available(
                cuda_only=False,
                min_cuda_compute_capability=None))
        

        # Build sampling table (32 x 64)
        luminance = 0.212671 * values.texels[:, :, 0] + \
                    0.715160 * values.texels[:, :, 1] + \
                    0.072169 * values.texels[:, :, 2]
        # For each y, compute CDF over x (32 x 64)
        sample_cdf_xs_ = tf.cumsum(luminance, axis=1)

        y_weight = tf.sin(
            math.pi * (tf.cast(
                tf.range(luminance.shape[0].value),
                tf.float32) + 0.5) / float(luminance.shape[0].value))
        
        # Compute CDF for x
        # pdb.set_trace()
        sample_cdf_ys_ = tf.cumsum(sample_cdf_xs_[:, -1] * y_weight, axis=0)
        # print("sample_cdf_ys_[-1]", sample_cdf_ys_[-1].eval())
        pdf_norm = (luminance.shape[0].value * luminance.shape[1].value) / \
        	(sample_cdf_ys_[-1] * (2 * math.pi * math.pi))
        # Normalize to [0, 1)
        sample_cdf_xs = (sample_cdf_xs_ - sample_cdf_xs_[:, 0:1]) / \
            tf.math.maximum(
                sample_cdf_xs_[
                    :, 
                    (luminance.shape[1].value - 1):luminance.shape[1].value],
                    1e-8 * tf.convert_to_tensor(np.ones((sample_cdf_xs_.shape[0], 1)), dtype=tf.float32)
                )
        sample_cdf_ys = (sample_cdf_ys_ - sample_cdf_ys_[0]) / \
            tf.math.maximum(sample_cdf_ys_[-1], tf.constant([1e-8]))

        self.values = values
        self.env_to_world = env_to_world
        self.world_to_env = tf.linalg.inv(env_to_world)
        self.sample_cdf_ys = tf.identity(sample_cdf_ys)
        self.sample_cdf_xs = tf.identity(sample_cdf_xs)
        self.pdf_norm = pdf_norm

    def state_dict(self):
        return {
            'values': self.values.state_dict(),
            'env_to_world': self.env_to_world,
            'world_to_env': self.world_to_env,
            'sample_cdf_ys': self.sample_cdf_ys,
            'sample_cdf_xs': self.sample_cdf_xs,
            'pdf_norm': self.pdf_norm,
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        out = cls.__new__(EnvironmentMap)
        out.values = pyredner.Texture.load_state_dict(state_dict['values'])
        out.env_to_world = state_dict['env_to_world']
        out.world_to_env = state_dict['world_to_env']
        out.sample_cdf_ys = state_dict['sample_cdf_ys']
        out.sample_cdf_xs = state_dict['sample_cdf_xs']
        out.pdf_norm = state_dict['pdf_norm']

        return out
