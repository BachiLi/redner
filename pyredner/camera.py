import numpy as np
import torch
import pyredner.transform as transform
import redner

class Camera:
    """
        redner supports four types of cameras:
            perspective, orthographic, fisheye, and panorama.
        The camera takes a look at transform or a cam_to_world matrix to
        transform from camera local space to world space. It also can optionally
        take an intrinsic matrix that models field of view and camera skew.

        Note:
            Currently we assume all the camera variables are stored in CPU,
            no matter whether redner is operating under CPU or GPU mode.

        Args:
            position (length 3 float tensor): the origin of the camera
            look_at (length 3 float tensor): the point camera is looking at
            up (length 3 float tensor): the up vector of the camera
            fov (length 1 float tensor): the field of view of the camera in angle, 
                                         no effect if the camera is a fisheye or panorama camera.
            clip_near (float): the near clipping plane of the camera, need to > 0
            resolution (length 2 tuple): the size of the output image in (height, width)
            cam_to_world (4x4 matrix, optional): overrides position, look_at, up vectors.
            intrinsic_mat (3x3 matrix, optional):
                A matrix that transforms a point in camera space before the point
                is projected to 2D screen space. Used for modelling field of view and
                camera skewing. After the multiplication the point should be in
                [-1, 1/aspect_ratio] x [1, -1/aspect_ratio] in homogeneous coordinates.
                The projection is then carried by the specific camera types.
                Perspective camera normalizes the homogeneous coordinates, while
                orthogonal camera drop the Z coordinate.
                Ignored in fisheye or panorama cameras.
                This matrix overrides fov.
            camera_type (render.camera_type): the type of the camera (perspective, orthographic, or fisheye)
            fisheye (bool): whether the camera is a fisheye camera (legacy parameter just to ensure compatibility).
    """
    def __init__(self,
                 position = None,
                 look_at = None,
                 up = None,
                 fov = None,
                 clip_near = 1e-4,
                 resolution = (256, 256),
                 cam_to_world = None,
                 intrinsic_mat = None,
                 camera_type = redner.CameraType.perspective,
                 fisheye = False):
        if position is not None:
            assert(position.dtype == torch.float32)
            assert(len(position.shape) == 1 and position.shape[0] == 3)
            assert(position.device.type == 'cpu')
        if look_at is not None:
            assert(look_at.dtype == torch.float32)
            assert(len(look_at.shape) == 1 and look_at.shape[0] == 3)
            assert(look_at.device.type == 'cpu')
        if up is not None:
            assert(up.dtype == torch.float32)
            assert(len(up.shape) == 1 and up.shape[0] == 3)
            assert(up.device.type == 'cpu')
        if fov is not None:
            assert(fov.dtype == torch.float32)
            assert(len(fov.shape) == 1 and fov.shape[0] == 1)
            assert(fov.device.type == 'cpu')
        assert(isinstance(clip_near, float))
        if position is None and look_at is None and up is None:
            assert(cam_to_world is  not None)

        self.position = position
        self.look_at = look_at
        self.up = up
        self._fov = fov
        self._cam_to_world = cam_to_world
        if cam_to_world is not None:
            self.world_to_cam = torch.inverse(self.cam_to_world).contiguous()
        else:
            self.world_to_cam = None
        if intrinsic_mat is None:
            if camera_type == redner.CameraType.perspective:
                fov_factor = 1.0 / torch.tan(transform.radians(0.5 * fov))
                o = torch.ones([1], dtype=torch.float32)
                diag = torch.cat([fov_factor, fov_factor, o], 0)
                self._intrinsic_mat = torch.diag(diag).contiguous()
            else:
                self._intrinsic_mat = torch.eye(3, dtype=torch.float32)
        else:
            self._intrinsic_mat = intrinsic_mat
        self.intrinsic_mat_inv = torch.inverse(self.intrinsic_mat).contiguous()
        self.clip_near = clip_near
        self.resolution = resolution
        self.camera_type = camera_type
        if fisheye:
            self.camera_type = redner.camera_type.fisheye

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, value):
        self._fov = value
        fov_factor = 1.0 / torch.tan(transform.radians(0.5 * self._fov))
        o = torch.ones([1], dtype=torch.float32)
        diag = torch.cat([fov_factor, fov_factor, o], 0)
        self._intrinsic_mat = torch.diag(diag).contiguous()
        self.intrinsic_mat_inv = torch.inverse(self._intrinsic_mat).contiguous()

    @property
    def intrinsic_mat(self):
        return self._intrinsic_mat

    @intrinsic_mat.setter
    def intrinsic_mat(self, value):
        self._intrinsic_mat = value
        self.intrinsic_mat_inv = torch.inverse(self._intrinsic_mat).contiguous()

    @property
    def cam_to_world(self):
        return self._cam_to_world

    @cam_to_world.setter
    def cam_to_world(self, value):
        self._cam_to_world = value
        self.world_to_cam = torch.inverse(self.cam_to_world).contiguous()

    def state_dict(self):
        return {
            'position': self._position,
            'look_at': self._look_at,
            'up': self._up,
            'fov': self._fov,
            'cam_to_world': self._cam_to_world,
            'intrinsic_mat': self._intrinsic_mat,
            'clip_near': self.clip_near,
            'resolution': self.resolution,
            'camera_type': self.camera_type
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        out = cls.__new__(Camera)
        out._position = state_dict['position']
        out._look_at = state_dict['look_at']
        out._up = state_dict['up']
        out._fov = state_dict['fov']
        out.cam_to_world = state_dict['cam_to_world']
        out.intrinsic_mat = state_dict['intrinsic_mat']
        out.clip_near = state_dict['clip_near']
        out.resolution = state_dict['resolution']
        out.camera_type = state_dict['camera_type']
        return out
