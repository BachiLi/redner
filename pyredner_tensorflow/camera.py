from typing import Tuple
import numpy as np
import tensorflow as tf
import pyredner_tensorflow.transform as transform
import redner
import pyredner_tensorflow as pyredner

class Camera:
    """
        redner supports four types of cameras:
            perspective, orthographic, fisheye, and panorama.
        The camera takes a look at transform or a cam_to_world matrix to
        transform from camera local space to world space. It also can optionally
        take an intrinsic matrix that models field of view and camera skew.

        Note:
            The Camera constructor converts all variables into a CPU device,
            no matter where they are originally.

        Args:
            position (length 3 float tensor): the origin of the camera
            look_at (length 3 float tensor): the point camera is looking at
            up (length 3 float tensor): the up vector of the camera
            fov (length 1 float tensor): the field of view of the camera in angle, 
                                         no effect if the camera is a fisheye camera
            clip_near (float): the near clipping plane of the camera, need to > 0
            resolution (length 2 tuple): the size of the output image in (height, width)
            cam_to_world (4x4 matrix): overrides position, look_at, up vectors.
            intrinsic_mat (3x3 matrix):
                A matrix that transforms a point in camera space before the point
                is projected to 2D screen space. Used for modelling field of view and
                camera skewing. After the multiplication the point should be in
                [-1, 1/aspect_ratio] x [1, -1/aspect_ratio] in homogeneous coordinates.
                The projection is then carried by the specific camera types.
                Perspective camera normalizes the homogeneous coordinates, while
                orthogonal camera drop the Z coordinate.
                This matrix overrides fov.
            camera_type (render.camera_type): the type of the camera (perspective, orthographic, or fisheye)
            fisheye (bool): whether the camera is a fisheye camera (legacy parameter just to ensure compatibility).
    """
    def __init__(self,
                 position: tf.Tensor = None,
                 look_at: tf.Tensor = None,
                 up: tf.Tensor = None,
                 fov: tf.Tensor = None,
                 clip_near: float = 1e-4,
                 resolution: Tuple[int] = (256, 256),
                 cam_to_world: tf.Tensor = None,
                 intrinsic_mat: tf.Tensor = None,
                 camera_type = redner.CameraType.perspective,
                 fisheye: bool = False):
        assert(tf.executing_eagerly())
        assert(position.dtype == tf.float32)
        assert(len(position.shape) == 1 and position.shape[0] == 3)
        assert(look_at.dtype == tf.float32)
        assert(len(look_at.shape) == 1 and look_at.shape[0] == 3)
        assert(up.dtype == tf.float32)
        assert(len(up.shape) == 1 and up.shape[0] == 3)
        if fov is not None:
            assert(fov.dtype == tf.float32)
            assert(len(fov.shape) == 1 and fov.shape[0] == 1)
        assert(isinstance(clip_near, float))
        if position is None and look_at is None and up is None:
            assert(cam_to_world is  not None)

        with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
            self.position = tf.identity(position).cpu()
            self.look_at = tf.identity(look_at).cpu()
            self.up = tf.identity(up).cpu()
            self.fov = tf.identity(fov).cpu()
            self._cam_to_world = cam_to_world
            if cam_to_world is not None:
                self.world_to_cam = tf.linalg.inv(self.cam_to_world)
            else:
                self.world_to_cam = None
            if intrinsic_mat is None:
                if camera_type == redner.CameraType.perspective:
                    fov_factor = 1.0 / tf.tan(transform.radians(0.5 * fov))
                    o = tf.convert_to_tensor(np.ones([1], dtype=np.float32), dtype=tf.float32)
                    diag = tf.concat([fov_factor, fov_factor, o], 0)
                    self._intrinsic_mat = tf.linalg.tensor_diag(diag)
                else:
                    self._intrinsic_mat = tf.eye(3, dtype=tf.float32)   
            else:
                self._intrinsic_mat = tf.identity(intrinsic_mat).cpu()
            self.intrinsic_mat_inv = tf.linalg.inv(self._intrinsic_mat)
            self.clip_near = clip_near
            self.resolution = resolution
            self.camera_type = camera_type
            if fisheye:
                self.camera_type = redner.CameraType.fisheye

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, value):
        with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
            self._fov = tf.identity(value).cpu()
            fov_factor = 1.0 / tf.tan(transform.radians(0.5 * self._fov))
            o = tf.convert_to_tensor(np.ones([1], dtype=np.float32), dtype=tf.float32)
            diag = tf.concat([fov_factor, fov_factor, o], 0)
            self._intrinsic_mat = tf.linalg.tensor_diag(diag)
            self.intrinsic_mat_inv = tf.linalg.inv(self._intrinsic_mat)

    @property
    def intrinsic_mat(self):
        return self._intrinsic_mat

    @intrinsic_mat.setter
    def intrinsic_mat(self, value):
        with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
            self._intrinsic_mat = tf.identity(value).cpu()
            self.intrinsic_mat_inv = tf.linalg.inv(self._intrinsic_mat)

    @property
    def cam_to_world(self):
        return self._cam_to_world

    @cam_to_world.setter
    def cam_to_world(self, value):
        self._cam_to_world = value
        self.world_to_cam = tf.linalg.inv(self.cam_to_world)

    def state_dict(self):
        return {
            'position': self.position,
            'look_at': self.look_at,
            'up': self.up,
            'fov': self.fov,
            'cam_to_world': self._cam_to_world,
            'world_to_cam': self.world_to_cam,
            'intrinsic_mat': self._intrinsic_mat,
            'clip_near': self.clip_near,
            'resolution': self.resolution,
            'camera_type': self.camera_type
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        out = cls.__new__(Camera)
        out.position = state_dict['position']
        out.look_at = state_dict['look_at']
        out.up = state_dict['up']
        out.fov = state_dict['fov']
        out.cam_to_world = state_dict['cam_to_world']
        out.intrinsic_mat = state_dict['intrinsic_mat']
        out.clip_near = state_dict['clip_near']
        out.resolution = state_dict['resolution']
        out.camera_type = state_dict['camera_type']
        return out
