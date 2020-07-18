import pyredner
import random
import redner
import torch
import math
from typing import Union, Tuple, Optional, List

class DeferredLight:
    pass

class AmbientLight(DeferredLight):
    """
        Ambient light for deferred rendering.
    """
    def __init__(self,
                 intensity: torch.Tensor):
        self.intensity = intensity

    def render(self,
               position: torch.Tensor,
               normal: torch.Tensor,
               albedo: torch.Tensor):
        return self.intensity.to(albedo.device) * albedo

class PointLight(DeferredLight):
    """
        Point light with squared distance falloff for deferred rendering.
    """
    def __init__(self,
                 position: torch.Tensor,
                 intensity: torch.Tensor):
        self.position = position
        self.intensity = intensity

    def render(self,
               position: torch.Tensor,
               normal: torch.Tensor,
               albedo: torch.Tensor):
        light_dir = self.position.to(position.device) - position
        # the d^2 term:
        light_dist_sq = torch.sum(light_dir * light_dir, dim = -1, keepdim = True)
        light_dist = torch.sqrt(light_dist_sq)
        # Normalize light direction
        light_dir = light_dir / light_dist
        dot_l_n = torch.sum(light_dir * normal, dim = -1, keepdim = True)
        dot_l_n = torch.max(dot_l_n, torch.zeros_like(dot_l_n))
        return self.intensity.to(dot_l_n.device) * dot_l_n * (albedo / math.pi) / light_dist_sq 

class DirectionalLight(DeferredLight):
    """
        Directional light for deferred rendering.
    """
    def __init__(self,
                 direction: torch.Tensor,
                 intensity: torch.Tensor):
        self.direction = direction
        self.intensity = intensity

    def render(self,
               position: torch.Tensor,
               normal: torch.Tensor,
               albedo: torch.Tensor):
        # Normalize light direction
        light_dir = -self.direction / torch.norm(self.direction)
        light_dir = light_dir.view(1, 1, 3)
        light_dir = light_dir.to(normal.device)
        dot_l_n = torch.sum(light_dir * normal, dim = -1, keepdim = True)
        dot_l_n = torch.max(dot_l_n, torch.zeros_like(dot_l_n))
        return self.intensity.to(dot_l_n.device) * dot_l_n * (albedo / math.pi)

class SpotLight(DeferredLight):
    """
        Spot light with cosine falloff for deferred rendering.
        Note that we do not provide the cosine cutoff here since it is not
        differentiable.
    """
    def __init__(self,
                 position: torch.Tensor,
                 spot_direction: torch.Tensor,
                 spot_exponent: torch.Tensor,
                 intensity: torch.Tensor):
        self.position = position
        self.spot_direction = spot_direction
        self.spot_exponent = spot_exponent
        self.intensity = intensity

    def render(self,
               position: torch.Tensor,
               normal: torch.Tensor,
               albedo: torch.Tensor):
        light_dir = self.position.to(position.device) - position
        # Normalize light direction
        light_dir = light_dir / torch.norm(light_dir, dim = -1, keepdim = True)
        # Normalize spot direction
        spot_direction = -self.spot_direction / torch.norm(self.spot_direction)
        spot_direction = spot_direction.to(light_dir.device)
        spot_cosine = torch.sum(light_dir * spot_direction, dim = -1, keepdim = True)
        spot_cosine = torch.max(spot_cosine, torch.zeros_like(spot_cosine))
        spot_factor = torch.pow(spot_cosine, self.spot_exponent.to(spot_cosine.device))
        dot_l_n = torch.sum(light_dir * normal, dim = -1, keepdim = True)
        dot_l_n = torch.max(dot_l_n, torch.zeros_like(dot_l_n))
        return self.intensity.to(spot_factor.device) * spot_factor * dot_l_n * (albedo / math.pi)

def render_deferred(scene: Union[pyredner.Scene, List[pyredner.Scene]],
                    lights: Union[List[DeferredLight], List[List[DeferredLight]]],
                    alpha: bool = False,
                    aa_samples: int = 2,
                    seed: Optional[Union[int, List[int], Tuple[int, int], List[Tuple[int, int]]]] = None,
                    sample_pixel_center: bool = False,
                    use_primary_edge_sampling: bool = True,
                    device: Optional[torch.device] = None):
    """
        Render the scenes using `deferred rendering <https://en.wikipedia.org/wiki/Deferred_shading>`_.
        We generate G-buffer images containing world-space position,
        normal, and albedo using redner, then shade the G-buffer
        using PyTorch code. Assuming Lambertian shading and does not
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
        seed: Optional[Union[int, List[int], Tuple[int, int], List[Tuple[int, int]]]]
            Random seed used for sampling. Randomly assigned if set to None.
            For batch render, if seed it not None, need to provide a list
            of seeds.
        sample_pixel_center: bool
            Always sample at the pixel center when rendering.
            This trades noise with aliasing.
            If this option is activated, the rendering becomes non-differentiable
            (since there is no antialiasing integral),
            and redner's edge sampling becomes an approximation to the gradients of the aliased rendering.
        use_primary_edge_sampling: bool
            debug option
        device: Optional[torch.device]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device().

        Returns
        =======
        torch.Tensor or List[torch.Tensor]
            | if input scene is a list: a tensor with size [N, H, W, C], N is the list size
            | else: a tensor with size [H, W, C]
            | if alpha == True, C = 4.
            | else, C = 3.
    """
    if device is None:
        device = pyredner.get_device()

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
        scene_args = pyredner.RenderFunction.serialize_scene(\
            scene = scene,
            num_samples = (1, 1),
            max_bounces = 0,
            sampler_type = redner.SamplerType.sobol,
            channels = channels,
            use_primary_edge_sampling = use_primary_edge_sampling,
            use_secondary_edge_sampling = False,
            sample_pixel_center = sample_pixel_center,
            device = device)
        # Need to revert the resolution back
        scene.camera.resolution = org_res
        scene.camera.viewport = org_viewport
        g_buffer = pyredner.RenderFunction.apply(seed, *scene_args)
        pos = g_buffer[:, :, :3]
        normal = g_buffer[:, :, 3:6]
        albedo = g_buffer[:, :, 6:9]
        img = torch.zeros(g_buffer.shape[0], g_buffer.shape[1], 3, device = device)
        for light in lights:
            img = img + light.render(pos, normal, albedo)
        if alpha:
            # alpha is in the last channel
            img = torch.cat((img, g_buffer[:, :, 9:10]), dim = -1)
        if aa_samples > 1:
            # Downsample
            img = img.permute(2, 0, 1) # HWC -> CHW
            img = img.unsqueeze(0) # CHW -> NCHW
            if org_viewport is not None:
                org_size = org_viewport[2] - org_viewport[0], org_viewport[3] - org_viewport[1]
            else:
                org_size = org_res
            img = torch.nn.functional.interpolate(img, size = org_size, mode = 'area')
            img = img.squeeze(dim = 0) # NCHW -> CHW
            img = img.permute(1, 2, 0)
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
                scene_args = pyredner.RenderFunction.serialize_scene(\
                    scene = sc,
                    num_samples = (1, 1),
                    max_bounces = 0,
                    sampler_type = redner.SamplerType.sobol,
                    channels = channels,
                    use_primary_edge_sampling = use_primary_edge_sampling,
                    use_secondary_edge_sampling = False,
                    sample_pixel_center = sample_pixel_center,
                    device = device)
                # Need to revert the resolution back
                sc.camera.resolution = org_res
                sc.camera.viewport = org_viewport
                g_buffers.append(pyredner.RenderFunction.apply(se, *scene_args))
            g_buffers = torch.stack(g_buffers)
            pos = g_buffers[:, :, :, :3]
            normal = g_buffers[:, :, :, 3:6]
            albedo = g_buffers[:, :, :, 6:9]
            imgs = torch.zeros(g_buffers.shape[0],
                               g_buffers.shape[1],
                               g_buffers.shape[2],
                               3,
                               device = device)
            for light in lights:
                imgs = imgs + light.render(pos, normal, albedo)
            if alpha:
                imgs = torch.cat((imgs, g_buffers[:, :, :, 9:10]), dim = -1)
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
                scene_args = pyredner.RenderFunction.serialize_scene(\
                    scene = sc,
                    num_samples = (1, 1),
                    max_bounces = 0,
                    sampler_type = redner.SamplerType.sobol,
                    channels = channels,
                    use_primary_edge_sampling = use_primary_edge_sampling,
                    use_secondary_edge_sampling = False,
                    sample_pixel_center = sample_pixel_center,
                    device = device)
                # Need to revert the resolution back
                sc.camera.resolution = org_res
                sc.camera.viewport = org_viewport
                g_buffer = pyredner.RenderFunction.apply(se, *scene_args)
                pos = g_buffer[:, :, :3]
                normal = g_buffer[:, :, 3:6]
                albedo = g_buffer[:, :, 6:9]
                img = torch.zeros(g_buffer.shape[0],
                                  g_buffer.shape[1],
                                  3,
                                  device = device)
                for light in lgts:
                    img = img + light.render(pos, normal, albedo)
                if alpha:
                    # alpha is in the last channel
                    img = torch.cat((img, g_buffer[:, :, 9:10]), dim = -1)
                imgs.append(img)
            imgs = torch.stack(imgs)
        if aa_samples > 1:
            # Downsample
            imgs = imgs.permute(0, 3, 1, 2) # NHWC -> NCHW
            if org_viewport is not None:
                org_size = org_viewport[2] - org_viewport[0], org_viewport[3] - org_viewport[1]
            else:
                org_size = org_res
            imgs = torch.nn.functional.interpolate(imgs, size = org_size, mode = 'area')
            imgs = imgs.permute(0, 2, 3, 1) # NCHW -> NHWC
        return imgs

def render_generic(scene: pyredner.Scene,
                   channels: List,
                   max_bounces: int = 1,
                   sampler_type = pyredner.sampler_type.sobol,
                   num_samples: Union[int, Tuple[int, int]] = (4, 4),
                   seed: Optional[Union[int, List[int], Tuple[int, int], List[Tuple[int, int]]]] = None,
                   sample_pixel_center: bool = False,
                   use_primary_edge_sampling: bool = True,
                   use_secondary_edge_sampling: bool = True,
                   device: Optional[torch.device] = None):
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
            | pyredner.channels.radiance,
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
        seed: Optional[Union[int, List[int], Tuple[int, int], List[Tuple[int, int]]]]
            Random seed used for sampling. Randomly assigned if set to None.
            For batch render, if seed it not None, need to provide a list
            of seeds.
        sample_pixel_center: bool
            Always sample at the pixel center when rendering.
            This trades noise with aliasing.
            If this option is activated, the rendering becomes non-differentiable
            (since there is no antialiasing integral),
            and redner's edge sampling becomes an approximation to the gradients of the aliased rendering.
        use_primary_edge_sampling: bool
            debug option
        use_secondary_edge_sampling: bool
            debug option
        device: Optional[torch.device]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device().

        Returns
        =======
        torch.Tensor or List[torch.Tensor]
            | if input scene is a list: a tensor with size [N, H, W, C], N is the list size
            | else: a tensor with size [H, W, C]
    """
    if device is None:
        device = pyredner.get_device()

    if isinstance(scene, pyredner.Scene):
        if seed==None:
            seed = random.randint(0, 16777216)
        scene_args = pyredner.RenderFunction.serialize_scene(\
            scene = scene,
            num_samples = num_samples,
            max_bounces = max_bounces,
            sampler_type = sampler_type,
            channels = channels,
            sample_pixel_center = sample_pixel_center,
            use_primary_edge_sampling = use_primary_edge_sampling,
            use_secondary_edge_sampling = use_secondary_edge_sampling,
            device = device)
        return pyredner.RenderFunction.apply(seed, *scene_args)
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
            scene_args = pyredner.RenderFunction.serialize_scene(\
                scene = sc,
                num_samples = num_samples,
                max_bounces = max_bounces,
                sampler_type = sampler_type,
                channels = channels,
                sample_pixel_center = sample_pixel_center,
                use_primary_edge_sampling = use_primary_edge_sampling,
                use_secondary_edge_sampling = use_secondary_edge_sampling,
                device = device)
            imgs.append(pyredner.RenderFunction.apply(se, *scene_args))
        imgs = torch.stack(imgs)
        return imgs

def render_g_buffer(scene: Union[pyredner.Scene, List[pyredner.Scene]],
                    channels: List,
                    num_samples: Union[int, Tuple[int, int]] = (1, 1),
                    seed: Optional[Union[int, List[int], Tuple[int, int], List[Tuple[int, int]]]] = None,
                    sample_pixel_center: bool = False,
                    use_primary_edge_sampling: bool = True,
                    use_secondary_edge_sampling: bool = True,
                    device: Optional[torch.device] = None):
    """
        Render G buffers from the scene.

        Args
        ====
        scene: Union[pyredner.Scene, List[pyredner.Scene]]
            pyredner Scene containing camera, geometry and material.
            Can be a single scene or a list for batch render.
            For batch rendering all scenes need to have the same resolution.
        channels: List[pyredner.channels]
            | A list of the following channels\:
            | pyredner.channels.radiance,
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
        seed: Optional[Union[int, List[int], Tuple[int, int], List[Tuple[int, int]]]]
            Random seed used for sampling. Randomly assigned if set to None.
            For batch render, if seed it not None, need to provide a list
            of seeds.
        sample_pixel_center: bool
            Always sample at the pixel center when rendering.
            This trades noise with aliasing.
            If this option is activated, the rendering becomes non-differentiable
            (since there is no antialiasing integral),
            and redner's edge sampling becomes an approximation to the gradients of the aliased rendering.
        use_primary_edge_sampling: bool
            debug option
        use_secondary_edge_sampling: bool
            debug option
        device: Optional[torch.device]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device().

        Returns
        =======
        torch.Tensor or List[torch.Tensor]
            | if input scene is a list: a tensor with size [N, H, W, C], N is the list size
            | else: a tensor with size [H, W, C]
    """
    return render_generic(scene = scene,
                          channels = channels,
                          max_bounces = 0,
                          sampler_type = redner.SamplerType.sobol,
                          num_samples = num_samples,
                          seed = seed,
                          sample_pixel_center = sample_pixel_center,
                          use_primary_edge_sampling = use_primary_edge_sampling,
                          use_secondary_edge_sampling = use_secondary_edge_sampling,
                          device = device)

def render_pathtracing(scene: Union[pyredner.Scene, List[pyredner.Scene]],
                       alpha: bool = False,
                       max_bounces: int = 1,
                       sampler_type = pyredner.sampler_type.sobol,
                       num_samples: Union[int, Tuple[int, int]] = (4, 4),
                       seed: Optional[Union[int, List[int], Tuple[int, int], List[Tuple[int, int]]]] = None,
                       sample_pixel_center: bool = False,
                       use_primary_edge_sampling: bool = True,
                       use_secondary_edge_sampling: bool = True,
                       device: Optional[torch.device] = None):
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
        seed: Optional[Union[int, List[int], Tuple[int, int], List[Tuple[int, int]]]]
            Random seed used for sampling. Randomly assigned if set to None.
            For batch render, if seed it not None, need to provide a list
            of seeds.
        sample_pixel_center: bool
            Always sample at the pixel center when rendering.
            This trades noise with aliasing.
            If this option is activated, the rendering becomes non-differentiable
            (since there is no antialiasing integral),
            and redner's edge sampling becomes an approximation to the gradients of the aliased rendering.
        use_primary_edge_sampling: bool
            debug option
        use_secondary_edge_sampling: bool
            debug option
        device: Optional[torch.device]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device().

        Returns
        =======
        torch.Tensor or List[torch.Tensor]
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
                          use_primary_edge_sampling = use_primary_edge_sampling,
                          use_secondary_edge_sampling = use_secondary_edge_sampling,
                          device = device)

def render_albedo(scene: Union[pyredner.Scene, List[pyredner.Scene]],
                  alpha: bool = False,
                  num_samples: Union[int, Tuple[int, int]] = (16, 4),
                  seed: Optional[Union[int, List[int], Tuple[int, int], List[Tuple[int, int]]]] = None,
                  sample_pixel_center: bool = False,
                  use_primary_edge_sampling: bool = True,
                  device: Optional[torch.device] = None):
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
        device: Optional[torch.device]
            Which device should we store the data in.
            If set to None, use the device from pyredner.get_device().

        Returns
        =======
        torch.Tensor or List[torch.Tensor]
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
                           use_primary_edge_sampling = use_primary_edge_sampling,
                           device = device)
