import numpy as np
import torch
import pyredner.transform as transform

class Camera:
    """
        redner supports a perspective camera and a fisheye camera.
        Both of them employ a look at transform.

        Note:
            Currently we assume all the camera variables are stored in CPU,
            no matter whether redner is operating under CPU or GPU mode.

        Args:
            position (length 3 float tensor): the origin of the camera
            look_at (length 3 float tensor): the point camera is looking at
            up (length 3 float tensor): the up vector of the camera
            fov (length 1 float tensor): the field of view of the camera in angle, 
                                         no effect if the camera is a fisheye camera
            clip_near (float): the near clipping plane of the camera, need to > 0
            resolution (length 2 tuple): the size of the output image in (width, height)
            fisheye (bool): whether the camera is a fisheye camera.
    """
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
        self.world_to_cam = torch.inverse(self.cam_to_world).contiguous()
        self.fov_factor = torch.tan(transform.radians(0.5 * fov))
        self.clip_near = clip_near
        self.resolution = resolution
        self.fisheye = fisheye
