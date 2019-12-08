import torch

class AreaLight:
    def __init__(self,
                 shape_id: int,
                 intensity: torch.Tensor,
                 two_sided: bool = False):
        self.shape_id = shape_id
        self.intensity = intensity
        self.two_sided = two_sided

    def state_dict(self):
        return {
            'shape_id': self.shape_id,
            'intensity': self.intensity,
            'two_sided': self.two_sided
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        return cls(
            state_dict['shape_id'],
            state_dict['intensity'],
            state_dict['two_sided'])
