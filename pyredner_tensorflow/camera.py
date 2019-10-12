from typing import Tuple
import numpy as np
import tensorflow as tf
import pyredner_tensorflow.transform as transform
import redner
import pyredner_tensorflow as pyredner

class Camera:
    """
        redner supports a perspective camera and a fisheye camera.
        Both of them employ a look at transform.

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
            cam_to_ndc (3x3 matrix): a matrix that transforms
                [-1, 1/aspect_ratio] x [1, -1/aspect_ratio] to [0, 1] x [0, 1]
                where aspect_ratio = width / height
            camera_type (render.camera_type): the type of the camera (perspective, orthographic, or fisheye)
            fisheye (bool): whether the camera is a fisheye camera (legacy parameter just to ensure compatibility).
    """
    def __init__(self,
                 position: tf.Tensor,
                 look_at: tf.Tensor,
                 up: tf.Tensor,
                 fov: tf.Tensor,
                 clip_near: float,
                 resolution: Tuple[int],
                 cam_to_world: tf.Tensor = None,
                 cam_to_ndc: tf.Tensor = None,
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
            if cam_to_ndc is None:
                if camera_type == redner.CameraType.perspective:
                    fov_factor = 1.0 / tf.tan(transform.radians(0.5 * fov))
                    o = tf.convert_to_tensor(np.ones([1], dtype=np.float32), dtype=tf.float32)
                    diag = tf.concat([fov_factor, fov_factor, o], 0)
                    self._cam_to_ndc = tf.linalg.tensor_diag(diag)
                else:
                    self._cam_to_ndc = tf.eye(3, dtype=tf.float32)   
            else:
                self._cam_to_ndc = tf.identity(cam_to_ndc).cpu()
            self.ndc_to_cam = tf.linalg.inv(self.cam_to_ndc)
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
            self._cam_to_ndc = tf.linalg.tensor_diag(diag)
            self.ndc_to_cam = tf.linalg.inv(self._cam_to_ndc)

    @property
    def cam_to_ndc(self):
        return self._cam_to_ndc

    @cam_to_ndc.setter
    def cam_to_ndc(self, value):
        with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
            self._cam_to_ndc = tf.identity(value).cpu()
            self.ndc_to_cam = tf.linalg.inv(self._cam_to_ndc)

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
            'cam_to_ndc': self._cam_to_ndc,
            'ndc_to_cam': self.ndc_to_cam,
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
        out._cam_to_world = state_dict['cam_to_world']
        out.world_to_cam = state_dict['world_to_cam']
        out._cam_to_ndc = state_dict['cam_to_ndc']
        out.ndc_to_cam = state_dict['ndc_to_cam']
        out.clip_near = state_dict['clip_near']
        out.resolution = state_dict['resolution']
        out.camera_type = state_dict['camera_type']
        return out
