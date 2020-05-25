import pyredner_tensorflow as pyredner
import random
import redner
import tensorflow as tf
import math
from typing import Union, Tuple, Optional, List

class DeferredLight:
    pass

class AmbientLight(DeferredLight):
    """
        Ambient light for deferred rendering.
    """
    def __init__(self,
                 intensity: tf.Tensor):
        self.intensity = intensity

    def render(self,
               position: tf.Tensor,
               normal: tf.Tensor,
               albedo: tf.Tensor):
        return self.intensity * albedo

class PointLight(DeferredLight):
    """
        Point light with squared distance falloff for deferred rendering.
    """
    def __init__(self,
                 position: tf.Tensor,
                 intensity: tf.Tensor):
        self.position = position
        self.intensity = intensity

    def render(self,
               position: tf.Tensor,
               normal: tf.Tensor,
               albedo: tf.Tensor):
        light_dir = self.position - position
        # the d^2 term:
        light_dist_sq = tf.reduce_sum(light_dir * light_dir, axis = -1, keepdims = True)
        light_dist = tf.sqrt(light_dist_sq)
        # Normalize light direction
        light_dir = light_dir / light_dist
        dot_l_n = tf.reduce_sum(light_dir * normal, axis = -1, keepdims = True)
        dot_l_n = tf.maximum(dot_l_n, tf.zeros_like(dot_l_n))
        return self.intensity * dot_l_n * (albedo / math.pi) / light_dist_sq 

class DirectionalLight(DeferredLight):
    """
        Directional light for deferred rendering.
    """
    def __init__(self,
                 direction: tf.Tensor,
                 intensity: tf.Tensor):
        self.direction = direction
        self.intensity = intensity

    def render(self,
               position: tf.Tensor,
               normal: tf.Tensor,
               albedo: tf.Tensor):
        # Normalize light direction
        light_dir = -self.direction / tf.norm(self.direction)
        light_dir = tf.reshape(light_dir, (1, 1, 3))
        dot_l_n = tf.reduce_sum(light_dir * normal, axis = -1, keepdims = True)
        dot_l_n = tf.maximum(dot_l_n, tf.zeros_like(dot_l_n))
        return self.intensity * dot_l_n * (albedo / math.pi)

class SpotLight(DeferredLight):
    """
        Spot light with cosine falloff for deferred rendering.
        Note that we do not provide the cosine cutoff here since it is not
        differentiable.
    """
    def __init__(self,
                 position: tf.Tensor,
                 spot_direction: tf.Tensor,
                 spot_exponent: tf.Tensor,
                 intensity: tf.Tensor):
        self.position = position
        self.spot_direction = spot_direction
        self.spot_exponent = spot_exponent
        self.intensity = intensity

    def render(self,
               position: tf.Tensor,
               normal: tf.Tensor,
               albedo: tf.Tensor):
        light_dir = self.position - position
        # Normalize light direction
        light_dir = light_dir / tf.norm(light_dir, axis = -1, keepdims = True)
        # Normalize spot direction
        spot_direction = -self.spot_direction / tf.norm(self.spot_direction)
        spot_cosine = tf.reduce_sum(light_dir * spot_direction, axis = -1, keepdims = True)
        spot_cosine = tf.maximum(spot_cosine, tf.zeros_like(spot_cosine))
        spot_factor = tf.pow(spot_cosine, self.spot_exponent)
        dot_l_n = tf.reduce_sum(light_dir * normal, axis = -1, keepdims = True)
        dot_l_n = tf.maximum(dot_l_n, tf.zeros_like(dot_l_n))
        return self.intensity * spot_factor * dot_l_n * (albedo / math.pi)

def render_deferred(scene: Union[pyredner.Scene, List[pyredner.Scene]],
                    lights: Union[List[DeferredLight], List[List[DeferredLight]]],
                    alpha: bool = False,
                    aa_samples: int = 2,
                    seed: Optional[Union[int, List[int]]] = None,
                    sample_pixel_center: bool = False,
                    device_name: Optional[str] = None):
    """
        Render the scenes using `deferred rendering <https://en.wikipedia.org/wiki/Deferred_shading>`_.
        We generate G-buffer images containing world-space position,
        normal, and albedo using redner, then shade the G-buffer
        using TensorFlow code. Assuming Lambertian shading and does not
        compute shadow.

        Args
        ====
        scene: Union[pyredner.Scene, List[pyredner.Scene]]
            pyredner Scene containing camera, geometry and material.
            Can be a single scene or a list for batch render.
            For batch rendering all scenes need to have the same resolution.
        lights: Union[List[DeferredLight], List[List[DeferredLight]]]
            Lights for deferred rendering. If the scene is a list, and only
            a single list of lights is provided, the same lights are applied
            to all scenes. If a list of lists of lights is provided, each scene
            is lit by the corresponding lights.
        alpha: bool
            If set to False, generates a 3-channel image,
            otherwise generates a 4-channel image where the
            fourth channel is alpha.
        aa_samples: int
            Number of samples used for anti-aliasing at both x, y dimensions
            (e.g. if aa_samples=2, 4 samples are used).
        seed: Optional[Union[int, List[int]]]
            Random seed used for sampling. Randomly assigned if set to None.
            For batch render, if seed it not None, need to provide a list
            of seeds.
        sample_pixel_center: bool
            Always sample at the pixel center when rendering.
            This trades noise with aliasing.
            If this option is activated, the rendering becomes non-differentiable
            (since there is no antialiasing integral),
            and redner's edge sampling becomes an approximation to the gradients of the aliased rendering.
        device_name: Optional[str]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device_name().

        Returns
        =======
        tf.Tensor or List[tf.Tensor]
            | if input scene is a list: a tensor with size [N, H, W, C], N is the list size
            | else: a tensor with size [H, W, C]
            | if alpha == True, C = 4.
            | else, C = 3.
    """
    channels = [redner.channels.position,
                redner.channels.shading_normal,
                redner.channels.diffuse_reflectance]
    if alpha:
        channels.append(redner.channels.alpha)
    if isinstance(scene, pyredner.Scene):
        if seed == None:
            seed = random.randint(0, 16777216)
        # We do full-screen anti-aliasing: increase the rendering resolution
        # and downsample it after lighting
        org_res = scene.camera.resolution
        org_viewport = scene.camera.viewport
        scene.camera.resolution = (org_res[0] * aa_samples,
                                   org_res[1] * aa_samples)
        if org_viewport is not None:
            scene.camera.viewport = [i * aa_samples for i in org_viewport]
        scene_args = pyredner.serialize_scene(\
            scene = scene,
            num_samples = (1, 1),
            max_bounces = 0,
            sampler_type = redner.SamplerType.sobol,
            channels = channels,
            use_secondary_edge_sampling = False,
            sample_pixel_center = sample_pixel_center,
            device_name = device_name)
        # Need to revert the resolution back
        scene.camera.resolution = org_res
        scene.camera.viewport = org_viewport
        g_buffer = pyredner.render(seed, *scene_args)
        pos = g_buffer[:, :, :3]
        normal = g_buffer[:, :, 3:6]
        albedo = g_buffer[:, :, 6:9]
        img = tf.zeros((g_buffer.shape[0], g_buffer.shape[1], 3))
        for light in lights:
            img = img + light.render(pos, normal, albedo)
        if alpha:
            # alpha is in the last channel
            img = tf.concat((img, g_buffer[:, :, 9:10]), axis = 2)
        if aa_samples > 1:
            # Downsample
            img = tf.expand_dims(img, 0) # HWC -> NHWC
            if org_viewport is not None:
                org_size = org_viewport[2] - org_viewport[0], org_viewport[3] - org_viewport[1]
            else:
                org_size = org_res
            # TODO: switch to method = 'area' when tensorflow implements the gradients...
            img = tf.image.resize(img, size = org_size, method = 'bilinear', antialias = True)
            img = tf.squeeze(img, axis = 0) # NHWC -> HWC
        return img
    else:
        assert(isinstance(scene, list))
        if seed == None:
            # Randomly generate a list of seed
            seed = []
            for i in range(len(scene)):
                seed.append(random.randint(0, 16777216))
        assert(len(seed) == len(scene))
        if len(lights) > 0 and not isinstance(lights[0], list):
            # Specialize version: stack g buffers and light all images in parallel
            g_buffers = []
            # Render each scene in the batch and stack them together
            for sc, se in zip(scene, seed):
                # We do full-screen anti-aliasing: increase the rendering resolution
                # and downsample it after lighting
                org_res = sc.camera.resolution
                org_viewport = sc.camera.viewport
                sc.camera.resolution = (org_res[0] * aa_samples,
                                        org_res[1] * aa_samples)
                if org_viewport is not None:
                    sc.camera.viewport = [i * aa_samples for i in org_viewport]
                scene_args = pyredner.serialize_scene(\
                    scene = sc,
                    num_samples = (1, 1),
                    max_bounces = 0,
                    sampler_type = redner.SamplerType.sobol,
                    channels = channels,
                    use_secondary_edge_sampling = False,
                    sample_pixel_center = sample_pixel_center,
                    device_name = device_name)
                # Need to revert the resolution back
                sc.camera.resolution = org_res
                sc.camera.viewport = org_viewport
                g_buffers.append(pyredner.render(se, *scene_args))
            g_buffers = tf.stack(g_buffers)
            pos = g_buffers[:, :, :, :3]
            normal = g_buffers[:, :, :, 3:6]
            albedo = g_buffers[:, :, :, 6:9]
            imgs = tf.zeros((g_buffers.shape[0],
                             g_buffers.shape[1],
                             g_buffers.shape[2],
                             3))
            for light in lights:
                imgs = imgs + light.render(pos, normal, albedo)
            if alpha:
                imgs = tf.concat((imgs, g_buffers[:, :, :, 9:10]), axis = -1)
        else:
            # If each scene has a different lighting: light them in the loop
            imgs = []
            # Render each scene in the batch and stack them together
            for sc, se, lgts in zip(scene, seed, lights):
                # We do full-screen anti-aliasing: increase the rendering resolution
                # and downsample it after lighting
                org_res = sc.camera.resolution
                org_viewport = sc.camera.viewport
                sc.camera.resolution = (org_res[0] * aa_samples,
                                        org_res[1] * aa_samples)
                if org_viewport is not None:
                    sc.camera.viewport = [i * aa_samples for i in org_viewport]
                scene_args = pyredner.serialize_scene(\
                    scene = sc,
                    num_samples = (1, 1),
                    max_bounces = 0,
                    sampler_type = redner.SamplerType.sobol,
                    channels = channels,
                    use_secondary_edge_sampling = False,
                    sample_pixel_center = sample_pixel_center,
                    device_name = device_name)
                # Need to revert the resolution back
                sc.camera.resolution = org_res
                sc.camera.viewport = org_viewport
                g_buffer = pyredner.render(se, *scene_args)
                pos = g_buffer[:, :, :3]
                normal = g_buffer[:, :, 3:6]
                albedo = g_buffer[:, :, 6:9]
                img = tf.zeros(g_buffer.shape[0],
                               g_buffer.shape[1],
                               3)
                for light in lgts:
                    img = img + light.render(pos, normal, albedo)
                if alpha:
                    # alpha is in the last channel
                    img = tf.concat((img, g_buffer[:, :, 9:10]), axis = -1)
                imgs.append(img)
            imgs = tf.stack(imgs)
        if aa_samples > 1:
            if org_viewport is not None:
                org_size = org_viewport[2] - org_viewport[0], org_viewport[3] - org_viewport[1]
            else:
                org_size = org_res
            # Downsample
            # TODO: switch to method = 'area' when tensorflow implements the gradients...
            imgs = tf.image.resize(imgs, size = org_size, method = 'bilinear', antialias = True)
        return imgs

def render_generic(scene: pyredner.Scene,
                   channels: List,
                   max_bounces: int = 1,
                   sampler_type = pyredner.sampler_type.sobol,
                   num_samples: Union[int, Tuple[int, int]] = (4, 4),
                   seed: Optional[int] = None,
                   sample_pixel_center: bool = False,
                   device_name: Optional[str] = None):
    """
        A generic rendering function that can be either pathtracing or
        g-buffer rendering or both.

        Args
        ====
        scene: Union[pyredner.Scene, List[pyredner.Scene]]
            pyredner Scene containing camera, geometry and material.
            Can be a single scene or a list for batch render.
            For batch rendering all scenes need to have the same resolution.
        channels: List[pyredner.channels]
            | A list of the following channels\:
            | pyredner.channels.alpha
            | pyredner.channels.depth
            | pyredner.channels.position
            | pyredner.channels.geometry_normal
            | pyredner.channels.shading_normal
            | pyredner.channels.uv
            | pyredner.channels.barycentric_coordinates
            | pyredner.channels.diffuse_reflectance
            | pyredner.channels.specular_reflectance
            | pyredner.channels.roughness
            | pyredner.channels.generic_texture
            | pyredner.channels.vertex_color
            | pyredner.channels.shape_id
            | pyredner.channels.triangle_id
            | pyredner.channels.material_id
        max_bounces: int
            Number of bounces for global illumination, 1 means direct lighting only.
        sampler_type: pyredner.sampler_type
            | Which sampling pattern to use? See 
              `Chapter 7 of the PBRT book <http://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction.html>`_
              for an explanation of the difference between different samplers.
            | Following samplers are supported\:
            | pyredner.sampler_type.independent
            | pyredner.sampler_type.sobol
        num_samples: int
            Number of samples per pixel for forward and backward passes.
            Can be an integer or a tuple of 2 integers.
        seed: Optional[Union[int, List[int]]]
            Random seed used for sampling. Randomly assigned if set to None.
            For batch render, if seed it not None, need to provide a list
            of seeds.
        sample_pixel_center: bool
            Always sample at the pixel center when rendering.
            This trades noise with aliasing.
            If this option is activated, the rendering becomes non-differentiable
            (since there is no antialiasing integral),
            and redner's edge sampling becomes an approximation to the gradients of the aliased rendering.
        device_name: Optional[str]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device_name().

        Returns
        =======
        tf.Tensor or List[tf.Tensor]
            | if input scene is a list: a tensor with size [N, H, W, C], N is the list size
            | else: a tensor with size [H, W, C]
    """
    if isinstance(scene, pyredner.Scene):
        if seed==None:
            seed = random.randint(0, 16777216)
        scene_args = pyredner.serialize_scene(\
            scene = scene,
            num_samples = num_samples,
            max_bounces = max_bounces,
            sampler_type = sampler_type,
            channels = channels,
            sample_pixel_center = sample_pixel_center,
            device_name = device_name)
        return pyredner.render(seed, *scene_args)
    else:
        assert(isinstance(scene, list))
        if seed == None:
            # Randomly generate a list of seed
            seed = []
            for i in range(len(scene)):
                seed.append(random.randint(0, 16777216))
        assert(len(seed) == len(scene))
        # Render each scene in the batch and stack them together
        imgs = []
        for sc, se in zip(scene, seed):
            scene_args = pyredner.serialize_scene(\
                scene = sc,
                num_samples = num_samples,
                max_bounces = max_bounces,
                sampler_type = sampler_type,
                channels = channels,
                sample_pixel_center = sample_pixel_center,
                device_name = device_name)
            imgs.append(pyredner.render(se, *scene_args))
        imgs = tf.stack(imgs)
        return imgs

def render_g_buffer(scene: pyredner.Scene,
                    channels: List[redner.channels],
                    num_samples: Union[int, Tuple[int, int]] = (1, 1),
                    seed: Optional[int] = None,
                    sample_pixel_center: bool = False,
                    device_name: Optional[str] = None):
    """
        Render a G buffer from the scene.

        Args
        ====
        scene: pyredner.Scene
            pyredner Scene containing camera, geometry, material, and lighting
        channels: List[pyredner.channels]
            | A list of the following channels\:
            | pyredner.channels.alpha
            | pyredner.channels.depth
            | pyredner.channels.position
            | pyredner.channels.geometry_normal
            | pyredner.channels.shading_normal
            | pyredner.channels.uv
            | pyredner.channels.barycentric_coordinates
            | pyredner.channels.diffuse_reflectance
            | pyredner.channels.specular_reflectance
            | pyredner.channels.roughness
            | pyredner.channels.generic_texture
            | pyredner.channels.vertex_color
            | pyredner.channels.shape_id
            | pyredner.channels.triangle_id
            | pyredner.channels.material_id
        num_samples: Union[int, Tuple[int, int]]
            Number of samples for forward and backward passes, respectively.
            If a single integer is provided, use the same number of samples
            for both.
        seed: Optional[int]
            Random seed used for sampling. Randomly assigned if set to None.
        sample_pixel_center: bool
            Always sample at the pixel center when rendering.
            This trades noise with aliasing.
            If this option is activated, the rendering becomes non-differentiable
            (since there is no antialiasing integral),
            and redner's edge sampling becomes an approximation to the gradients of the aliased rendering.
        device_name: Optional[str]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device_name().

        Returns
        =======
        tf.Tensor
            a tensor with size [H, W, C]
    """
    return render_generic(scene = scene,
                          channels = channels,
                          max_bounces = 0,
                          sampler_type = redner.SamplerType.sobol,
                          num_samples = num_samples,
                          seed = seed,
                          sample_pixel_center = sample_pixel_center,
                          device_name = device_name)

def render_pathtracing(scene: Union[pyredner.Scene, List[pyredner.Scene]],
                       alpha: bool = False,
                       max_bounces: int = 1,
                       sampler_type = pyredner.sampler_type.sobol,
                       num_samples: Union[int, Tuple[int, int]] = (4, 4),
                       seed: Optional[Union[int, List[int]]] = None,
                       sample_pixel_center: bool = False,
                       device_name: Optional[str] = None):
    """
        Render a pyredner scene using pathtracing.

        Args
        ====
        scene: Union[pyredner.Scene, List[pyredner.Scene]]
            pyredner Scene containing camera, geometry and material.
            Can be a single scene or a list for batch render.
            For batch rendering all scenes need to have the same resolution.
        max_bounces: int
            Number of bounces for global illumination, 1 means direct lighting only.
        sampler_type: pyredner.sampler_type
            | Which sampling pattern to use? See 
              `Chapter 7 of the PBRT book <http://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction.html>`_
              for an explanation of the difference between different samplers.
            | Following samplers are supported\:
            | pyredner.sampler_type.independent
            | pyredner.sampler_type.sobol
        num_samples: int
            Number of samples per pixel for forward and backward passes.
            Can be an integer or a tuple of 2 integers.
        seed: Optional[Union[int, List[int]]]
            Random seed used for sampling. Randomly assigned if set to None.
            For batch render, if seed it not None, need to provide a list
            of seeds.
        sample_pixel_center: bool
            Always sample at the pixel center when rendering.
            This trades noise with aliasing.
            If this option is activated, the rendering becomes non-differentiable
            (since there is no antialiasing integral),
            and redner's edge sampling becomes an approximation to the gradients of the aliased rendering.
        device_name: Optional[str]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device_name().

        Returns
        =======
        tf.Tensor or List[tf.Tensor]
            | if input scene is a list: a tensor with size [N, H, W, C], N is the list size
            | else: a tensor with size [H, W, C]
            | if alpha == True, C = 4.
            | else, C = 3.
    """
    channels = [redner.channels.radiance]
    if alpha:
        channels.append(redner.channels.alpha)
    return render_generic(scene = scene,
                          channels = channels,
                          max_bounces = max_bounces,
                          sampler_type = sampler_type,
                          num_samples = num_samples,
                          seed = seed,
                          sample_pixel_center = sample_pixel_center,
                          device_name = device_name)

def render_albedo(scene: Union[pyredner.Scene, List[pyredner.Scene]],
                  alpha: bool = False,
                  num_samples: Union[int, Tuple[int, int]] = (16, 4),
                  seed: Optional[Union[int, List[int]]] = None,
                  sample_pixel_center: bool = False,
                  device_name: Optional[str] = None):
    """
        Render the diffuse albedo colors of the scenes.

        Args
        ====
        scene: Union[pyredner.Scene, List[pyredner.Scene]]
            pyredner Scene containing camera, geometry and material.
            Can be a single scene or a list for batch render.
            For batch rendering all scenes need to have the same resolution.
        alpha: bool
            If set to False, generates a 3-channel image,
            otherwise generates a 4-channel image where the
            fourth channel is alpha.
        num_samples: Union[int, Tuple[int, int]]
            number of samples for forward and backward passes, respectively
            if a single integer is provided, use the same number of samples
            for both
        seed: Optional[Union[int, List[int]]]
            Random seed used for sampling. Randomly assigned if set to None.
            For batch render, if seed it not None, need to provide a list
            of seeds.
        sample_pixel_center: bool
            Always sample at the pixel center when rendering.
            This trades noise with aliasing.
            If this option is activated, the rendering becomes non-differentiable
            (since there is no antialiasing integral),
            and redner's edge sampling becomes an approximation to the gradients of the aliased rendering.
        device_name: Optional[str]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device_name().

        Returns
        =======
        tf.Tensor or List[tf.Tensor]
            | if input scene is a list: a tensor with size [N, H, W, C], N is the list size
            | else: a tensor with size [H, W, C]
            | if alpha == True, C = 4.
            | else, C = 3.
    """
    channels = [redner.channels.diffuse_reflectance]
    if alpha:
        channels.append(redner.channels.alpha)
    return render_g_buffer(scene = scene,
                           channels = channels,
                           num_samples = num_samples,
                           seed = seed,
                           sample_pixel_center = sample_pixel_center,
                           device_name = device_name)
