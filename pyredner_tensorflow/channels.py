import redner
import tensorflow as tf

class Channel:
    def __init__(self):
        self.radiance = redner.channels.radiance
        self.alpha = redner.channels.alpha
        self.depth = redner.channels.depth
        self.position = redner.channels.position
        self.geometry_normal = redner.channels.geometry_normal
        self.shading_normal = redner.channels.shading_normal
        self.uv = redner.channels.uv
        self.diffuse_reflectance = redner.channels.diffuse_reflectance
        self.specular_reflectance = redner.channels.specular_reflectance
        self.roughness = redner.channels.roughness
        self.generic_texture = redner.channels.generic_texture
        self.vertex_color = redner.channels.vertex_color
        self.shape_id = redner.channels.shape_id
        self.material_id = redner.channels.material_id

channels = Channel()

class RednerChannels:
    __channels = [
        redner.channels.radiance,
        redner.channels.alpha,
        redner.channels.depth,
        redner.channels.position,
        redner.channels.geometry_normal,
        redner.channels.shading_normal,
        redner.channels.uv,
        redner.channels.diffuse_reflectance,
        redner.channels.specular_reflectance,
        redner.channels.roughness,
        redner.channels.generic_texture,
        redner.channels.vertex_color,
        redner.channels.shape_id,
        redner.channels.material_id
    ]

    @staticmethod
    def asTensor(channel: redner.channels) -> tf.Tensor:
        assert isinstance(channel, redner.channels)

        for i in range(len(RednerChannels.__channels)):
            if RednerChannels.__channels[i] == channel:
                return tf.constant(i)

    @staticmethod
    def asChannel(index: tf.Tensor) -> redner.channels:
        try:
            channel = RednerChannels.__channels[index]
        except IndexError:
            print(f'{index} is out of range: [0, {len(RednerChannels.__channels)})')
            import sys
            sys.exit()
        else:
            return channel

class RednerCameraType:
    __cameratypes = [
        redner.CameraType.perspective,
        redner.CameraType.orthographic,
        redner.CameraType.fisheye,
    ]

    @staticmethod
    def asTensor(cameratype: redner.CameraType) -> tf.Tensor:
        assert isinstance(cameratype, redner.CameraType)

        for i in range(len(RednerCameraType.__cameratypes)):
            if RednerCameraType.__cameratypes[i] == cameratype:
                return tf.constant(i)


    @staticmethod
    def asCameraType(index: tf.Tensor) -> redner.CameraType:
        try:
            cameratype = RednerCameraType.__cameratypes[index]
        except IndexError:
            print(f'{index} is out of range: [0, {len(RednerCameraType.__cameratypes)})')
            import sys
            sys.exit()
        else:
            return cameratype
