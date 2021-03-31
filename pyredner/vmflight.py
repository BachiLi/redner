import pyredner
import torch
import math

class VonMisesFisherLight:
    def __init__(self, kappa, intensity, env_to_world = torch.eye(4, 4)):
        # Convert to constant texture if necessary

        assert(env_to_world.device.type == 'cpu')
        assert(env_to_world.dtype == torch.float32)
        assert(env_to_world.is_contiguous())

        pdf_norm = 1.0

        self.kappa = kappa
        self.intensity = intensity
        self.env_to_world = env_to_world
        self.world_to_env = torch.inverse(env_to_world).contiguous()
        self.pdf_norm = pdf_norm

    def state_dict(self):
        return {
            'kappa': self.kappa,
            'intensity': self.intensity,
            'env_to_world': self.env_to_world,
            'world_to_env': self.world_to_env,
            'pdf_norm': self.pdf_norm,
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        out = cls.__new__(VonMisesFisherLight)
        out.kappa = state_dict['kappa']
        out.intensity = state_dict['intensity']
        out.env_to_world = state_dict['env_to_world']
        out.world_to_env = state_dict['world_to_env']
        out.pdf_norm = state_dict['pdf_norm']

        return out
