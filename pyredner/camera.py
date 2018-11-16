import numpy as np
import torch
import pyredner.transform as transform

class Camera:
    def __init__(self,
                 position,
                 look_at,
                 up,
                 fov,
                 clip_near,
                 resolution,
                 fisheye = False):
        assert(position.dtype == torch.float32)
        assert(len(position.shape) == 1 and position.shape[0] == 3)
        assert(look_at.dtype == torch.float32)
        assert(len(look_at.shape) == 1 and look_at.shape[0] == 3)
        assert(up.dtype == torch.float32)
        assert(len(up.shape) == 1 and up.shape[0] == 3)
        assert(fov.dtype == torch.float32)
        assert(len(fov.shape) == 1 and fov.shape[0] == 1)
        assert(isinstance(clip_near, float))

        self.position = position
        self.look_at = look_at
        self.up = up
        self.fov = fov
        self.cam_to_world = transform.gen_look_at_matrix(position, look_at, up)
        self.world_to_cam = torch.inverse(self.cam_to_world)
        self.fov_factor = torch.tan(transform.radians(0.5 * fov))
        self.clip_near = clip_near
        self.resolution = resolution
        self.fisheye = fisheye
