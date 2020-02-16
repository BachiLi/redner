import torch
import pyredner.transform as transform
import redner
import math
import pyredner
from typing import Tuple, Optional, List

class Camera:
    """
        Redner supports four types of cameras\: perspective, orthographic, fisheye, and panorama.
        The camera takes a look at transform or a cam_to_world matrix to
        transform from camera local space to world space. It also can optionally
        take an intrinsic matrix that models field of view and camera skew.

        Args
        ====
        position: Optional[torch.Tensor]
            the origin of the camera, 1-d tensor with size 3 and type float32
        look_at: Optional[torch.Tensor]
            the point camera is looking at, 1-d tensor with size 3 and type float32
        up: Optional[torch.Tensor]
            the up vector of the camera, 1-d tensor with size 3 and type float32
        fov: Optional[torch.Tensor]
            the field of view of the camera in angle
            no effect if the camera is a fisheye or panorama camera
            1-d tensor with size 1 and type float32
        clip_near: float
            the near clipping plane of the camera, need to > 0
        resolution: Tuple[int, int]
            the size of the output image in (height, width)
        cam_to_world: Optional[torch.Tensor]
            overrides position, look_at, up vectors
            4x4 matrix, optional
        intrinsic_mat: Optional[torch.Tensor]
            a matrix that transforms a point in camera space before the point
            is projected to 2D screen space
            used for modelling field of view and camera skewing
            after the multiplication the point should be in
            [-1, 1/aspect_ratio] x [1, -1/aspect_ratio] in homogeneous coordinates
            the projection is then carried by the specific camera types
            perspective camera normalizes the homogeneous coordinates
            while orthogonal camera drop the Z coordinate.
            ignored by fisheye or panorama cameras
            overrides fov
            3x3 matrix, optional
        camera_type: render.camera_type
            the type of the camera (perspective, orthographic, fisheye, or panorama)
        fisheye: bool
            whether the camera is a fisheye camera
            (legacy parameter just to ensure compatibility).
    """
    def __init__(self,
                 position: Optional[torch.Tensor] = None,
                 look_at: Optional[torch.Tensor] = None,
                 up: Optional[torch.Tensor] = None,
                 fov: Optional[torch.Tensor] = None,
                 clip_near: float = 1e-4,
                 resolution: Tuple[int, int] = (256, 256),
                 cam_to_world: Optional[torch.Tensor] = None,
                 intrinsic_mat: Optional[torch.Tensor] = None,
                 camera_type = pyredner.camera_type.perspective,
                 fisheye: bool = False):
        if position is not None:
            assert(position.dtype == torch.float32)
            assert(len(position.shape) == 1 and position.shape[0] == 3)
        if look_at is not None:
            assert(look_at.dtype == torch.float32)
            assert(len(look_at.shape) == 1 and look_at.shape[0] == 3)
        if up is not None:
            assert(up.dtype == torch.float32)
            assert(len(up.shape) == 1 and up.shape[0] == 3)
        if fov is not None:
            assert(fov.dtype == torch.float32)
            assert(len(fov.shape) == 1 and fov.shape[0] == 1)
        assert(isinstance(clip_near, float))
        if position is None and look_at is None and up is None:
            assert(cam_to_world is not None)

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
            self.camera_type = pyredner.camera_type.fisheye

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
        if value is not None:
            self._intrinsic_mat = value
            self.intrinsic_mat_inv = torch.inverse(self._intrinsic_mat).contiguous()
        else:
            assert(self.fov is not None)
            self.fov = self._fov

    @property
    def cam_to_world(self):
        return self._cam_to_world

    @cam_to_world.setter
    def cam_to_world(self, value):
        if value is not None:
            self._cam_to_world = value
            self.world_to_cam = torch.inverse(self.cam_to_world).contiguous()
        else:
            self._cam_to_world = None
            self.world_to_cam = None

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

def automatic_camera_placement(shapes: List,
                               resolution: Tuple[int, int]):
    """
        Given a list of objects or shapes, generates camera parameters automatically
        using the bounding boxes of the shapes. Place the camera at
        some distances from the shapes, so that it can see all of them.
        Inspired by https://github.com/mitsuba-renderer/mitsuba/blob/master/src/librender/scene.cpp#L286

        Parameters
        ==========
        shapes: List
            a list of redner Shape or Object
        resolution: Tuple[int, int]
            the size of the output image in (height, width)

        Returns
        =======
        pyredner.Camera
            a camera that can see all the objects.
    """
    aabb_min = torch.tensor((float('inf'), float('inf'), float('inf')))
    aabb_max = -torch.tensor((float('inf'), float('inf'), float('inf')))
    for shape in shapes:
        v = shape.vertices
        v_min = torch.min(v, 0)[0].cpu()
        v_max = torch.max(v, 0)[0].cpu()
        aabb_min = torch.min(aabb_min, v_min)
        aabb_max = torch.max(aabb_max, v_max)
    assert(torch.isfinite(aabb_min).all() and torch.isfinite(aabb_max).all())
    center = (aabb_max + aabb_min) * 0.5
    extents = aabb_max - aabb_min
    max_extents_xy = torch.max(extents[0], extents[1])
    distance = max_extents_xy / (2 * math.tan(45 * 0.5 * math.pi / 180.0))
    max_extents_xyz = torch.max(extents[2], max_extents_xy)    
    return Camera(position = torch.tensor((center[0], center[1], aabb_min[2] - distance)),
                  look_at = center,
                  up = torch.tensor((0.0, 1.0, 0.0)),
                  fov = torch.tensor([45.0]),
                  clip_near = 0.001 * float(distance),
                  resolution = resolution)

def generate_intrinsic_mat(fx: torch.Tensor,
                           fy: torch.Tensor,
                           skew: torch.Tensor,
                           x0: torch.Tensor,
                           y0: torch.Tensor):
    """
        | Generate the following 3x3 intrinsic matrix given the parameters.
        | fx, skew, x0
        |  0,   fy, y0
        |  0,    0,  1

        Parameters
        ==========
        fx: torch.Tensor
            Focal length at x dimension. 1D tensor with size 1.
        fy: torch.Tensor
            Focal length at y dimension. 1D tensor with size 1.
        skew: torch.Tensor
            Axis skew parameter describing shearing transform. 1D tensor with size 1.
        x0: torch.Tensor
            Principle point offset at x dimension. 1D tensor with size 1.
        y0: torch.Tensor
            Principle point offset at y dimension. 1D tensor with size 1.

        Returns
        =======
        torch.Tensor
            3x3 intrinsic matrix
    """
    z = torch.zeros_like(fx)
    o = torch.ones_like(fx)
    row0 = torch.cat([fx, skew, x0])
    row1 = torch.cat([ z,   fy, y0])
    row2 = torch.cat([ z,    z,  o])
    return torch.stack([row0, row1, row2]).contiguous()
