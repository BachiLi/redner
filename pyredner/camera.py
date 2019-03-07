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
            resolution (length 2 tuple): the size of the output image in (height, width)
            cam_to_ndc (3x3 matrix): a matrix that transforms
                [-1, 1/aspect_ratio] x [1, -1/aspect_ratio] to [0, 1] x [0, 1]
                where aspect_ratio = width / height
            fisheye (bool): whether the camera is a fisheye camera.
    """
    def __init__(self,
                 position,
                 look_at,
                 up,
                 fov,
                 clip_near,
                 resolution,
                 cam_to_ndc = None,
                 fisheye = False):
        assert(position.dtype == torch.float32)
        assert(len(position.shape) == 1 and position.shape[0] == 3)
        assert(look_at.dtype == torch.float32)
        assert(len(look_at.shape) == 1 and look_at.shape[0] == 3)
        assert(up.dtype == torch.float32)
        assert(len(up.shape) == 1 and up.shape[0] == 3)
        if fov is not None:
            assert(fov.dtype == torch.float32)
            assert(len(fov.shape) == 1 and fov.shape[0] == 1)
        assert(isinstance(clip_near, float))

        self._position = position
        self._look_at = look_at
        self._up = up
        self._fov = fov
        self.cam_to_world = transform.gen_look_at_matrix(position, look_at, up)
        self.world_to_cam = torch.inverse(self.cam_to_world).contiguous()
        if cam_to_ndc is None:
            fov_factor = 1.0 / torch.tan(transform.radians(0.5 * fov))
            o = torch.ones([1], dtype=torch.float32)
            diag = torch.cat([fov_factor, fov_factor, o], 0)
            self._cam_to_ndc = torch.diag(diag)
        else:
            self._cam_to_ndc = cam_to_ndc
        self.ndc_to_cam = torch.inverse(self.cam_to_ndc)
        self.clip_near = clip_near
        self.resolution = resolution
        self.fisheye = fisheye

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value
        self.cam_to_world = \
            transform.gen_look_at_matrix(self._position, self._look_at, self._up)
        self.world_to_cam = torch.inverse(self.cam_to_world).contiguous()

    @property
    def look_at(self):
        return self._look_at
    
    @look_at.setter
    def look_at(self, value):
        self._look_at = value
        self.cam_to_world = \
            transform.gen_look_at_matrix(self._position, self._look_at, self._up)
        self.world_to_cam = torch.inverse(self.cam_to_world).contiguous()

    @property
    def up(self):
        return self._up

    @up.setter
    def up(self, value):
        self._up = value
        self.cam_to_world = \
            transform.gen_look_at_matrix(self._position, self._look_at, self._up)
        self.world_to_cam = torch.inverse(self.cam_to_world).contiguous()

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, value):
        self._fov = value
        fov_factor = 1.0 / torch.tan(transform.radians(0.5 * self._fov))
        o = torch.ones([1], dtype=torch.float32)
        diag = torch.cat([fov_factor, fov_factor, o], 0)
        self._cam_to_ndc = torch.diag(diag)
        self.ndc_to_cam = torch.inverse(self._cam_to_ndc)

    @property
    def cam_to_ndc(self):
        return self._cam_to_ndc

    @cam_to_ndc.setter
    def cam_to_ndc(self, value):
        self._cam_to_ndc = value
        self.ndc_to_cam = torch.inverse(self._cam_to_ndc)

