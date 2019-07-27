import pyredner_tensorflow as pyredner
import numpy as np
import tensorflow as tf
import math
import pdb

class EnvironmentMap:
    def __init__(self, values, env_to_world = tf.eye(4, 4)):
        assert(tf.executing_eagerly())
        # Convert to constant texture if necessary
        if tf.is_tensor(values):
            values = pyredner.Texture(values)

        # assert(values.texels.is_contiguous())
        assert(values.texels.dtype == tf.float32)
        
        with tf.device(pyredner.get_device_name()):
            # Build sampling table
            luminance = 0.212671 * values.texels[:, :, 0] + \
                        0.715160 * values.texels[:, :, 1] + \
                        0.072169 * values.texels[:, :, 2]
            # For each y, compute CDF over x
            sample_cdf_xs_ = tf.cumsum(luminance, axis=1)

            y_weight = tf.sin(
                math.pi * (tf.cast(
                    tf.range(luminance.shape[0].value),
                    tf.float32) + 0.5) / float(luminance.shape[0].value))

            # Compute CDF for x
            sample_cdf_ys_ = tf.cumsum(sample_cdf_xs_[:, -1] * y_weight, axis=0)
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
            self.sample_cdf_ys = sample_cdf_ys
            self.sample_cdf_xs = sample_cdf_xs
        with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
            self.pdf_norm = pdf_norm.cpu()
            env_to_world = tf.identity(env_to_world).cpu()
            self.env_to_world = env_to_world
            self.world_to_env = tf.linalg.inv(env_to_world)

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
