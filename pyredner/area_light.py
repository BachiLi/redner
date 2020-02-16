import torch

class AreaLight:
    """
        A mesh-based area light that points to a shape and assigns intensity.

        Args
        ----------
        shape_id: int

        intensity: torch.Tensor
            1-d tensor with size 3 and type float32
        two_sided: bool
            Is the light emitting light from the two sides of the faces?
        directly_visible: bool
            Can the camera sees the light source directly?
    """

    def __init__(self,
                 shape_id: int,
                 intensity: torch.Tensor,
                 two_sided: bool = False,
                 directly_visible: bool = True):
        self.shape_id = shape_id
        self.intensity = intensity
        self.two_sided = two_sided
        self.directly_visible = directly_visible

    def state_dict(self):
        return {
            'shape_id': self.shape_id,
            'intensity': self.intensity,
            'two_sided': self.two_sided,
            'directly_visible': self.directly_visible
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        return cls(
            state_dict['shape_id'],
            state_dict['intensity'],
            state_dict['two_sided'],
            state_dict['directly_visible'])
