import torch
import numpy as np
import redner
import pyredner
import time
import skimage.io
from typing import List, Union, Tuple

use_correlated_random_number = False
def set_use_correlated_random_number(v: bool):
    """
        | There is a bias-variance trade off in the backward pass.
        | If the forward pass and the backward pass are correlated
        | the gradients are biased for L2 loss.
        | E[d/dx(f(x) - y)^2] = E[(f(x) - y) d/dx f(x)]
        |                     = E[f(x) - y] E[d/dx f(x)]
        | The last equation only holds when f(x) and d/dx f(x) are independent.
        | It is usually better to use the unbiased one, but we left it as an option here
    """
    global use_correlated_random_number
    use_correlated_random_number = v

def get_use_correlated_random_number():
    """
        See set_use_correlated_random_number
    """
    global use_correlated_random_number
    return use_correlated_random_number

print_timing = True

def serialize_texture(texture, args):
    if texture is None:
        args.append(0)
        return
    args.append(len(texture.mipmap))
    for mipmap in texture.mipmap:
        assert(torch.isfinite(mipmap).all())
        assert(mipmap.is_contiguous())
        args.append(mipmap.to(pyredner.get_device()))
    assert(torch.isfinite(texture.uv_scale).all())
    args.append(texture.uv_scale.to(pyredner.get_device()))

class RenderFunction(torch.autograd.Function):
    """
        The PyTorch interface of C++ redner.
    """

    @staticmethod
    def serialize_scene(scene: pyredner.Scene,
                        num_samples: Union[int, Tuple[int, int]],
                        max_bounces: int,
                        channels: List = [redner.channels.radiance],
                        sampler_type = redner.SamplerType.independent,
                        use_primary_edge_sampling: bool = True,
                        use_secondary_edge_sampling: bool = True):
        """
            Given a pyredner scene & rendering options, convert them to a linear list of argument,
            so that we can use it in PyTorch.

            Args
            ====
            scene: pyredner.Scene
            num_samples: int
                Number of samples per pixel for forward and backward passes.
                Can be an integer or a tuple of 2 integers.
                If a single integer is provided, use the same number of samples
                for both.
            max_bounces: int
                Number of bounces for global illumination,
                1 means direct lighting only.
            channels: List[redner.channels]
                | A list of channels that should present in the output image
                | following channels are supported\:
                | redner.channels.radiance,
                | redner.channels.alpha,
                | redner.channels.depth,
                | redner.channels.position,
                | redner.channels.geometry_normal,
                | redner.channels.shading_normal,
                | redner.channels.uv,
                | redner.channels.diffuse_reflectance,
                | redner.channels.specular_reflectance,
                | redner.channels.vertex_color,
                | redner.channels.roughness,
                | redner.channels.generic_texture,
                | redner.channels.shape_id,
                | redner.channels.material_id
                | all channels, except for shape id and material id, are differentiable
            sampler_type: redner.SamplerType
                | Which sampling pattern to use?
                | see `Chapter 7 of the PBRT book <http://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction.html>`
                  for an explanation of the difference between different samplers.
                | Following samplers are supported:
                | redner.SamplerType.independent
                | redner.SamplerType.sobol
            use_primary_edge_sampling: bool

            use_secondary_edge_sampling: bool

        """
        # Record if there is any parameter that requires gradient need discontinuity sampling.
        # For skipping edge sampling when it is not necessary.
        requires_visibility_grad = False
        cam = scene.camera
        num_shapes = len(scene.shapes)
        num_materials = len(scene.materials)
        num_lights = len(scene.area_lights)
        for light_id, light in enumerate(scene.area_lights):
            scene.shapes[light.shape_id].light_id = light_id

        args = []
        args.append(num_shapes)
        args.append(num_materials)
        args.append(num_lights)
        assert(cam.position is None or torch.isfinite(cam.position).all())
        assert(cam.look_at is None or torch.isfinite(cam.look_at).all())
        assert(cam.up is None or torch.isfinite(cam.up).all())
        assert(torch.isfinite(cam.intrinsic_mat_inv).all())
        assert(torch.isfinite(cam.intrinsic_mat).all())
        if cam.position is not None and cam.position.requires_grad:
            requires_visibility_grad = True
        if cam.look_at is not None and cam.look_at.requires_grad:
            requires_visibility_grad = True
        if cam.up is not None and cam.up.requires_grad:
            requires_visibility_grad = True
        args.append(cam.position.cpu() if cam.position is not None else None)
        args.append(cam.look_at.cpu() if cam.look_at is not None else None)
        args.append(cam.up.cpu() if cam.up is not None else None)
        if cam.cam_to_world is not None:
            if cam.cam_to_world.requires_grad:
                requires_visibility_grad = True
            args.append(cam.cam_to_world.cpu().contiguous())
        else:
            args.append(None)
        if cam.world_to_cam is not None:
            if cam.world_to_cam.requires_grad:
                requires_visibility_grad = True
            args.append(cam.world_to_cam.cpu().contiguous())
        else:
            args.append(None)
        if cam.intrinsic_mat.requires_grad or cam.intrinsic_mat_inv.requires_grad:
            requires_visibility_grad = True
        args.append(cam.intrinsic_mat_inv.cpu().contiguous())
        args.append(cam.intrinsic_mat.cpu().contiguous())
        args.append(cam.clip_near)
        args.append(cam.resolution)
        args.append(cam.camera_type)
        for shape in scene.shapes:
            assert(torch.isfinite(shape.vertices).all())
            if (shape.uvs is not None):
                assert(torch.isfinite(shape.uvs).all())
            if (shape.normals is not None):
                assert(torch.isfinite(shape.normals).all())
            if (shape.vertices.requires_grad):
                requires_visibility_grad = True
            args.append(shape.vertices.to(pyredner.get_device()))
            args.append(shape.indices.to(pyredner.get_device()))
            args.append(shape.uvs.to(pyredner.get_device()) if shape.uvs is not None else None)
            args.append(shape.normals.to(pyredner.get_device()) if shape.normals is not None else None)
            args.append(shape.uv_indices.to(pyredner.get_device()) if shape.uv_indices is not None else None)
            args.append(shape.normal_indices.to(pyredner.get_device()) if shape.normal_indices is not None else None)
            args.append(shape.colors.to(pyredner.get_device()) if shape.colors is not None else None)
            args.append(shape.material_id)
            args.append(shape.light_id)
        for material in scene.materials:
            serialize_texture(material.diffuse_reflectance, args)
            serialize_texture(material.specular_reflectance, args)
            serialize_texture(material.roughness, args)
            serialize_texture(material.generic_texture, args)
            serialize_texture(material.normal_map, args)
            args.append(material.compute_specular_lighting)
            args.append(material.two_sided)
            args.append(material.use_vertex_color)
        for light in scene.area_lights:
            args.append(light.shape_id)
            args.append(light.intensity)
            args.append(light.two_sided)
        if scene.envmap is not None:
            assert(torch.isfinite(scene.envmap.env_to_world).all())
            assert(torch.isfinite(scene.envmap.world_to_env).all())
            assert(torch.isfinite(scene.envmap.sample_cdf_ys).all())
            assert(torch.isfinite(scene.envmap.sample_cdf_xs).all())
            serialize_texture(scene.envmap.values, args)
            args.append(scene.envmap.env_to_world.cpu())
            args.append(scene.envmap.world_to_env.cpu())
            args.append(scene.envmap.sample_cdf_ys.to(pyredner.get_device()))
            args.append(scene.envmap.sample_cdf_xs.to(pyredner.get_device()))
            args.append(scene.envmap.pdf_norm)
        else:
            args.append(None)

        args.append(num_samples)
        args.append(max_bounces)
        args.append(channels)
        args.append(sampler_type)
        if requires_visibility_grad:
            args.append(use_primary_edge_sampling)
            args.append(use_secondary_edge_sampling)
        else:
            # Don't need to do edge sampling if we don't require spatial derivatives
            args.append(False)
            args.append(False)

        return args

    @staticmethod
    def forward(ctx,
                seed,
                *args):
        """
            Forward rendering pass: given a serialized scene and output an image.
        """
        # Unpack arguments
        current_index = 0
        num_shapes = args[current_index]
        current_index += 1
        num_materials = args[current_index]
        current_index += 1
        num_lights = args[current_index]
        current_index += 1

        cam_position = args[current_index]
        current_index += 1
        cam_look_at = args[current_index]
        current_index += 1
        cam_up = args[current_index]
        current_index += 1
        cam_to_world = args[current_index]
        current_index += 1
        world_to_cam = args[current_index]
        current_index += 1
        intrinsic_mat_inv = args[current_index]
        current_index += 1
        intrinsic_mat = args[current_index]
        current_index += 1
        clip_near = args[current_index]
        current_index += 1
        resolution = args[current_index]
        current_index += 1
        camera_type = args[current_index]
        current_index += 1
        if cam_to_world is None:
            camera = redner.Camera(resolution[1],
                                   resolution[0],
                                   redner.float_ptr(cam_position.data_ptr()),
                                   redner.float_ptr(cam_look_at.data_ptr()),
                                   redner.float_ptr(cam_up.data_ptr()),
                                   redner.float_ptr(0), # cam_to_world
                                   redner.float_ptr(0), # world_to_cam
                                   redner.float_ptr(intrinsic_mat_inv.data_ptr()),
                                   redner.float_ptr(intrinsic_mat.data_ptr()),
                                   clip_near,
                                   camera_type)
        else:
            camera = redner.Camera(resolution[1],
                                   resolution[0],
                                   redner.float_ptr(0), # cam_position
                                   redner.float_ptr(0), # cam_look_at
                                   redner.float_ptr(0), # cam_up
                                   redner.float_ptr(cam_to_world.data_ptr()),
                                   redner.float_ptr(world_to_cam.data_ptr()),
                                   redner.float_ptr(intrinsic_mat_inv.data_ptr()),
                                   redner.float_ptr(intrinsic_mat.data_ptr()),
                                   clip_near,
                                   camera_type)
        shapes = []
        for i in range(num_shapes):
            vertices = args[current_index]
            current_index += 1
            indices = args[current_index]
            current_index += 1
            uvs = args[current_index]
            current_index += 1
            normals = args[current_index]
            current_index += 1
            uv_indices = args[current_index]
            current_index += 1
            normal_indices = args[current_index]
            current_index += 1
            colors = args[current_index]
            current_index += 1
            material_id = args[current_index]
            current_index += 1
            light_id = args[current_index]
            current_index += 1
            assert(vertices.is_contiguous())
            assert(indices.is_contiguous())
            if uvs is not None:
                assert(uvs.is_contiguous())
            if normals is not None:
                assert(normals.is_contiguous())
            if uv_indices is not None:
                assert(uv_indices.is_contiguous())
            if normal_indices is not None:
                assert(normal_indices.is_contiguous())
            shapes.append(redner.Shape(\
                redner.float_ptr(vertices.data_ptr()),
                redner.int_ptr(indices.data_ptr()),
                redner.float_ptr(uvs.data_ptr() if uvs is not None else 0),
                redner.float_ptr(normals.data_ptr() if normals is not None else 0),
                redner.int_ptr(uv_indices.data_ptr() if uv_indices is not None else 0),
                redner.int_ptr(normal_indices.data_ptr() if normal_indices is not None else 0),
                redner.float_ptr(colors.data_ptr() if colors is not None else 0),
                int(vertices.shape[0]),
                int(uvs.shape[0]) if uvs is not None else 0,
                int(normals.shape[0]) if normals is not None else 0,
                int(indices.shape[0]),
                material_id,
                light_id))

        materials = []
        for i in range(num_materials):
            num_levels = args[current_index]
            current_index += 1
            diffuse_reflectance = []
            for j in range(num_levels):
                diffuse_reflectance.append(args[current_index])
                current_index += 1
            diffuse_uv_scale = args[current_index]
            current_index += 1
            
            num_levels = args[current_index]
            current_index += 1
            specular_reflectance = []
            for j in range(num_levels):
                specular_reflectance.append(args[current_index])
                current_index += 1
            specular_uv_scale = args[current_index]
            current_index += 1
            
            num_levels = args[current_index]
            current_index += 1
            roughness = []
            for j in range(num_levels):
                roughness.append(args[current_index])
                current_index += 1
            roughness_uv_scale = args[current_index]
            current_index += 1

            num_levels = args[current_index]
            current_index += 1
            generic_texture = []
            if num_levels > 0:
                for j in range(num_levels):
                    generic_texture.append(args[current_index])
                    current_index += 1
                generic_uv_scale = args[current_index]
                current_index += 1
            else:
                generic_uv_scale = None

            num_levels = args[current_index]
            current_index += 1
            normal_map = []
            if num_levels > 0:
                for j in range(num_levels):
                    normal_map.append(args[current_index])
                    current_index += 1
                normal_map_uv_scale = args[current_index]
                current_index += 1
            else:
                normal_map_uv_scale = None

            compute_specular_lighting = args[current_index]
            current_index += 1
            two_sided = args[current_index]
            current_index += 1
            use_vertex_color = args[current_index]
            current_index += 1

            if diffuse_reflectance[0].dim() == 1:
                # Constant texture
                diffuse_reflectance = redner.Texture3(\
                    [redner.float_ptr(diffuse_reflectance[0].data_ptr())],
                    [0], [0], 3,
                    redner.float_ptr(diffuse_uv_scale.data_ptr()))
            else:
                assert(diffuse_reflectance[0].dim() == 3)
                diffuse_reflectance = redner.Texture3(\
                    [redner.float_ptr(x.data_ptr()) for x in diffuse_reflectance],
                    [x.shape[1] for x in diffuse_reflectance],
                    [x.shape[0] for x in diffuse_reflectance],
                    3,
                    redner.float_ptr(diffuse_uv_scale.data_ptr()))

            if specular_reflectance[0].dim() == 1:
                # Constant texture
                specular_reflectance = redner.Texture3(\
                    [redner.float_ptr(specular_reflectance[0].data_ptr())],
                    [0], [0], 3,
                    redner.float_ptr(specular_uv_scale.data_ptr()))
            else:
                assert(specular_reflectance[0].dim() == 3)
                specular_reflectance = redner.Texture3(\
                    [redner.float_ptr(x.data_ptr()) for x in specular_reflectance],
                    [x.shape[1] for x in specular_reflectance],
                    [x.shape[0] for x in specular_reflectance],
                    3,
                    redner.float_ptr(specular_uv_scale.data_ptr()))

            if roughness[0].dim() == 1:
                # Constant texture
                roughness = redner.Texture1(\
                    [redner.float_ptr(roughness[0].data_ptr())],
                    [0], [0], 1,
                    redner.float_ptr(roughness_uv_scale.data_ptr()))
            else:
                assert(roughness[0].dim() == 3)
                roughness = redner.Texture1(\
                    [redner.float_ptr(x.data_ptr()) for x in roughness],
                    [x.shape[1] for x in roughness],
                    [x.shape[0] for x in roughness],
                    1,
                    redner.float_ptr(roughness_uv_scale.data_ptr()))

            if len(generic_texture) > 0:
                assert(generic_texture[0].dim() == 3)
                generic_texture = redner.TextureN(\
                    [redner.float_ptr(x.data_ptr()) for x in generic_texture],
                    [x.shape[1] for x in generic_texture],
                    [x.shape[0] for x in generic_texture],
                    generic_texture[0].shape[2],
                    redner.float_ptr(generic_uv_scale.data_ptr()))
            else:
                generic_texture = redner.TextureN(\
                    [], [], [], 0, redner.float_ptr(0))

            if len(normal_map) > 0:
                assert(normal_map[0].dim() == 3)
                normal_map = redner.Texture3(\
                    [redner.float_ptr(x.data_ptr()) for x in normal_map],
                    [x.shape[1] for x in normal_map],
                    [x.shape[0] for x in normal_map],
                    3,
                    redner.float_ptr(normal_map_uv_scale.data_ptr()))
            else:
                normal_map = redner.Texture3(\
                    [], [], [], 0, redner.float_ptr(0))
            materials.append(redner.Material(\
                diffuse_reflectance,
                specular_reflectance,
                roughness,
                generic_texture,
                normal_map,
                compute_specular_lighting,
                two_sided,
                use_vertex_color))

        area_lights = []
        for i in range(num_lights):
            shape_id = args[current_index]
            current_index += 1
            intensity = args[current_index]
            current_index += 1
            two_sided = args[current_index]
            current_index += 1

            area_lights.append(redner.AreaLight(\
                shape_id,
                redner.float_ptr(intensity.data_ptr()),
                two_sided))

        envmap = None
        if args[current_index] is not None:
            num_levels = args[current_index]
            current_index += 1
            values = []
            for j in range(num_levels):
                values.append(args[current_index])
                current_index += 1
            envmap_uv_scale = args[current_index]
            current_index += 1
            env_to_world = args[current_index]
            current_index += 1
            world_to_env = args[current_index]
            current_index += 1
            sample_cdf_ys = args[current_index]
            current_index += 1
            sample_cdf_xs = args[current_index]
            current_index += 1
            pdf_norm = args[current_index]
            current_index += 1
            values = redner.Texture3(\
                [redner.float_ptr(x.data_ptr()) for x in values],
                [x.shape[1] for x in values], # width
                [x.shape[0] for x in values], # height
                3, # channels
                redner.float_ptr(envmap_uv_scale.data_ptr()))
            envmap = redner.EnvironmentMap(\
                values,
                redner.float_ptr(env_to_world.data_ptr()),
                redner.float_ptr(world_to_env.data_ptr()),
                redner.float_ptr(sample_cdf_ys.data_ptr()),
                redner.float_ptr(sample_cdf_xs.data_ptr()),
                pdf_norm)
        else:
            current_index += 1

        # Options
        num_samples = args[current_index]
        current_index += 1
        max_bounces = args[current_index]
        current_index += 1
        channels = args[current_index]
        current_index += 1
        sampler_type = args[current_index]
        current_index += 1
        use_primary_edge_sampling = args[current_index]
        current_index += 1
        use_secondary_edge_sampling = args[current_index]
        current_index += 1

        start = time.time()
        scene = redner.Scene(camera,
                             shapes,
                             materials,
                             area_lights,
                             envmap,
                             pyredner.get_use_gpu(),
                             pyredner.get_device().index if pyredner.get_device().index is not None else -1,
                             use_primary_edge_sampling,
                             use_secondary_edge_sampling)
        time_elapsed = time.time() - start
        if print_timing:
            print('Scene construction, time: %.5f s' % time_elapsed)

        # check that num_samples is a tuple
        if isinstance(num_samples, int):
            num_samples = (num_samples, num_samples)

        options = redner.RenderOptions(seed, num_samples[0], max_bounces, channels, sampler_type)
        num_channels = redner.compute_num_channels(channels,
                                                   scene.max_generic_texture_dimension)
        rendered_image = torch.zeros(resolution[0], resolution[1], num_channels,
            device = pyredner.get_device())
        start = time.time()
        redner.render(scene,
                      options,
                      redner.float_ptr(rendered_image.data_ptr()),
                      redner.float_ptr(0),
                      None,
                      redner.float_ptr(0))
        time_elapsed = time.time() - start
        if print_timing:
            print('Forward pass, time: %.5f s' % time_elapsed)

        # # For debugging
        # debug_img = torch.zeros(256, 256, 3)
        # redner.render(scene,
        #               options,
        #               redner.float_ptr(rendered_image.data_ptr()),
        #               redner.float_ptr(0),
        #               None,
        #               redner.float_ptr(debug_img.data_ptr()))
        # pyredner.imwrite(debug_img, 'debug.exr')
        # exit()

        ctx.camera = camera
        ctx.shapes = shapes
        ctx.materials = materials
        ctx.area_lights = area_lights
        ctx.envmap = envmap
        ctx.scene = scene
        ctx.options = options
        ctx.num_samples = num_samples
        ctx.args = args # Important to prevent GC from deallocating the tensors
        return rendered_image

    @staticmethod
    def backward(ctx,
                 grad_img):
        if not grad_img.is_contiguous():
            grad_img = grad_img.contiguous()
        scene = ctx.scene
        options = ctx.options
        camera = ctx.camera

        if camera.use_look_at:
            d_cam_position = torch.zeros(3, device = pyredner.get_device())
            d_cam_look = torch.zeros(3, device = pyredner.get_device())
            d_cam_up = torch.zeros(3, device = pyredner.get_device())
            d_cam_to_world = None
            d_world_to_cam = None
        else:
            d_cam_position = None
            d_cam_look = None
            d_cam_up = None
            d_cam_to_world = torch.zeros(4, 4, device = pyredner.get_device())
            d_world_to_cam = torch.zeros(4, 4, device = pyredner.get_device())
        d_intrinsic_mat_inv = torch.zeros(3, 3, device = pyredner.get_device())
        d_intrinsic_mat = torch.zeros(3, 3, device = pyredner.get_device())
        if camera.use_look_at:
            d_camera = redner.DCamera(redner.float_ptr(d_cam_position.data_ptr()),
                                      redner.float_ptr(d_cam_look.data_ptr()),
                                      redner.float_ptr(d_cam_up.data_ptr()),
                                      redner.float_ptr(0), # cam_to_world
                                      redner.float_ptr(0), # world_to_cam
                                      redner.float_ptr(d_intrinsic_mat_inv.data_ptr()),
                                      redner.float_ptr(d_intrinsic_mat.data_ptr()))
        else:
            d_camera = redner.DCamera(redner.float_ptr(0), # pos
                                      redner.float_ptr(0), # look
                                      redner.float_ptr(0), # up
                                      redner.float_ptr(d_cam_to_world.data_ptr()),
                                      redner.float_ptr(d_world_to_cam.data_ptr()),
                                      redner.float_ptr(d_intrinsic_mat_inv.data_ptr()),
                                      redner.float_ptr(d_intrinsic_mat.data_ptr()))
        d_vertices_list = []
        d_uvs_list = []
        d_normals_list = []
        d_colors_list = []
        d_shapes = []
        for shape in ctx.shapes:
            num_vertices = shape.num_vertices
            num_uv_vertices = shape.num_uv_vertices
            num_normal_vertices = shape.num_normal_vertices
            d_vertices = torch.zeros(num_vertices, 3,
                device = pyredner.get_device())
            d_uvs = torch.zeros(num_uv_vertices, 2,
                device = pyredner.get_device()) if shape.has_uvs() else None
            d_normals = torch.zeros(num_normal_vertices, 3,
                device = pyredner.get_device()) if shape.has_normals() else None
            d_colors = torch.zeros(num_vertices, 3,
                device = pyredner.get_device()) if shape.has_colors() else None
            d_vertices_list.append(d_vertices)
            d_uvs_list.append(d_uvs)
            d_normals_list.append(d_normals)
            d_colors_list.append(d_colors)
            d_shapes.append(redner.DShape(\
                redner.float_ptr(d_vertices.data_ptr()),
                redner.float_ptr(d_uvs.data_ptr() if d_uvs is not None else 0),
                redner.float_ptr(d_normals.data_ptr() if d_normals is not None else 0),
                redner.float_ptr(d_colors.data_ptr() if d_colors is not None else 0)))

        d_diffuse_list = []
        d_diffuse_uv_scale_list = []
        d_specular_list = []
        d_specular_uv_scale_list = []
        d_roughness_list = []
        d_roughness_uv_scale_list = []
        d_generic_list = []
        d_generic_uv_scale_list = []
        d_normal_map_list = []
        d_normal_map_uv_scale_list = []
        d_materials = []
        for material in ctx.materials:
            if material.get_diffuse_size(0)[0] == 0:
                d_diffuse = [torch.zeros(3, device = pyredner.get_device())]
            else:
                d_diffuse = []
                for l in range(material.get_diffuse_levels()):
                    diffuse_size = material.get_diffuse_size(l)
                    d_diffuse.append(\
                        torch.zeros(diffuse_size[1],
                                    diffuse_size[0],
                                    3, device = pyredner.get_device()))

            if material.get_specular_size(0)[0] == 0:
                d_specular = [torch.zeros(3, device = pyredner.get_device())]
            else:
                d_specular = []
                for l in range(material.get_specular_levels()):
                    specular_size = material.get_specular_size(l)
                    d_specular.append(\
                        torch.zeros(specular_size[1],
                                    specular_size[0],
                                    3, device = pyredner.get_device()))

            if material.get_roughness_size(0)[0] == 0:
                d_roughness = [torch.zeros(1, device = pyredner.get_device())]
            else:
                d_roughness = []
                for l in range(material.get_roughness_levels()):
                    roughness_size = material.get_roughness_size(l)
                    d_roughness.append(\
                        torch.zeros(roughness_size[1],
                                    roughness_size[0],
                                    1, device = pyredner.get_device()))

            if material.get_generic_levels() == 0:
                d_generic = None
            else:
                d_generic = []
                for l in range(material.get_generic_levels()):
                    generic_size = material.get_generic_size(l)
                    d_generic.append(\
                        torch.zeros(generic_size[2],
                                    generic_size[1],
                                    generic_size[0], device = pyredner.get_device()))

            if material.get_normal_map_levels() == 0:
                d_normal_map = None
            else:
                d_normal_map = []
                for l in range(material.get_normal_map_levels()):
                    normal_map_size = material.get_normal_map_size(l)
                    d_normal_map.append(\
                        torch.zeros(normal_map_size[1],
                                    normal_map_size[0],
                                    3, device = pyredner.get_device()))

            d_diffuse_list.append(d_diffuse)
            d_specular_list.append(d_specular)
            d_roughness_list.append(d_roughness)
            d_generic_list.append(d_generic)
            d_normal_map_list.append(d_normal_map)
            d_diffuse_uv_scale = torch.zeros(2, device = pyredner.get_device())
            d_specular_uv_scale = torch.zeros(2, device = pyredner.get_device())
            d_roughness_uv_scale = torch.zeros(2, device = pyredner.get_device())
            d_diffuse_uv_scale_list.append(d_diffuse_uv_scale)
            d_specular_uv_scale_list.append(d_specular_uv_scale)
            d_roughness_uv_scale_list.append(d_roughness_uv_scale)
            if d_generic is None:
                d_generic_uv_scale = None
            else:
                d_generic_uv_scale = torch.zeros(2, device = pyredner.get_device())
            if d_normal_map is None:
                d_normal_map_uv_scale = None
            else:
                d_normal_map_uv_scale = torch.zeros(2, device = pyredner.get_device())

            d_generic_uv_scale_list.append(d_generic_uv_scale)
            d_normal_map_uv_scale_list.append(d_normal_map_uv_scale)
            if d_diffuse[0].dim() == 1:
                d_diffuse_tex = redner.Texture3(\
                    [redner.float_ptr(d_diffuse[0].data_ptr())],
                    [0],
                    [0],
                    3,
                    redner.float_ptr(d_diffuse_uv_scale.data_ptr()))
            else:
                d_diffuse_tex = redner.Texture3(\
                    [redner.float_ptr(x.data_ptr()) for x in d_diffuse],
                    [x.shape[1] for x in d_diffuse],
                    [x.shape[0] for x in d_diffuse],
                    3,
                    redner.float_ptr(d_diffuse_uv_scale.data_ptr()))

            if d_specular[0].dim() == 1:
                d_specular_tex = redner.Texture3(\
                    [redner.float_ptr(d_specular[0].data_ptr())],
                    [0],
                    [0],
                    3,
                    redner.float_ptr(d_specular_uv_scale.data_ptr()))
            else:
                d_specular_tex = redner.Texture3(\
                    [redner.float_ptr(x.data_ptr()) for x in d_specular],
                    [x.shape[1] for x in d_specular],
                    [x.shape[0] for x in d_specular],
                    3,
                    redner.float_ptr(d_specular_uv_scale.data_ptr()))

            if d_roughness[0].dim() == 1:
                d_roughness_tex = redner.Texture1(\
                    [redner.float_ptr(d_roughness[0].data_ptr())],
                    [0],
                    [0],
                    1,
                    redner.float_ptr(d_roughness_uv_scale.data_ptr()))
            else:
                d_roughness_tex = redner.Texture1(\
                    [redner.float_ptr(x.data_ptr()) for x in d_roughness],
                    [x.shape[1] for x in d_roughness],
                    [x.shape[0] for x in d_roughness],
                    1,
                    redner.float_ptr(d_roughness_uv_scale.data_ptr()))

            if d_generic is None:
                d_generic_tex = redner.TextureN(\
                    [], [], [], 0, redner.float_ptr(0))
            else:
                d_generic_tex = redner.TextureN(\
                    [redner.float_ptr(x.data_ptr()) for x in d_generic],
                    [x.shape[1] for x in d_generic],
                    [x.shape[0] for x in d_generic],
                    d_generic[0].shape[2],
                    redner.float_ptr(d_generic_uv_scale.data_ptr()))

            if d_normal_map is None:
                d_normal_map = redner.Texture3(\
                    [], [], [], 0, redner.float_ptr(0))
            else:
                d_normal_map = redner.Texture3(\
                    [redner.float_ptr(x.data_ptr()) for x in d_normal_map],
                    [x.shape[1] for x in d_normal_map],
                    [x.shape[0] for x in d_normal_map],
                    3,
                    redner.float_ptr(d_normal_map_uv_scale.data_ptr()))
            d_materials.append(redner.DMaterial(\
                d_diffuse_tex, d_specular_tex, d_roughness_tex,
                d_generic_tex, d_normal_map))

        d_intensity_list = []
        d_area_lights = []
        for light in ctx.area_lights:
            d_intensity = torch.zeros(3, device = pyredner.get_device())
            d_intensity_list.append(d_intensity)
            d_area_lights.append(\
                redner.DAreaLight(redner.float_ptr(d_intensity.data_ptr())))

        d_envmap = None
        if ctx.envmap is not None:
            envmap = ctx.envmap
            d_envmap_values = []
            for l in range(envmap.get_levels()):
                size = envmap.get_size(l)
                d_envmap_values.append(\
                    torch.zeros(size[1],
                                size[0],
                                3, device = pyredner.get_device()))
            d_envmap_uv_scale = torch.zeros(2, device = pyredner.get_device())
            d_envmap_tex = redner.Texture3(\
                [redner.float_ptr(x.data_ptr()) for x in d_envmap_values],
                [x.shape[1] for x in d_envmap_values],
                [x.shape[0] for x in d_envmap_values],
                3,
                redner.float_ptr(d_envmap_uv_scale.data_ptr()))
            d_world_to_env = torch.zeros(4, 4, device = pyredner.get_device())
            d_envmap = redner.DEnvironmentMap(\
                d_envmap_tex,
                redner.float_ptr(d_world_to_env.data_ptr()))

        d_scene = redner.DScene(d_camera,
                                d_shapes,
                                d_materials,
                                d_area_lights,
                                d_envmap,
                                pyredner.get_use_gpu(),
                                pyredner.get_device().index if pyredner.get_device().index is not None else -1)
        if not get_use_correlated_random_number():
            # Decouple the forward/backward random numbers by adding a big prime number
            options.seed += 1000003

        options.num_samples = ctx.num_samples[1]
        start = time.time()
        redner.render(scene, options,
                      redner.float_ptr(0),
                      redner.float_ptr(grad_img.data_ptr()),
                      d_scene,
                      redner.float_ptr(0))
        time_elapsed = time.time() - start
        if print_timing:
            print('Backward pass, time: %.5f s' % time_elapsed)

        # For debugging
        # pyredner.imwrite(grad_img, 'grad_img.exr')
        # grad_img = torch.ones(256, 256, 3, device = pyredner.get_device())
        # debug_img = torch.zeros(256, 256, 3)
        # start = time.time()
        # redner.render(scene, options,
        #               redner.float_ptr(0),
        #               redner.float_ptr(grad_img.data_ptr()),
        #               d_scene,
        #               redner.float_ptr(debug_img.data_ptr()))
        # time_elapsed = time.time() - start
        # if print_timing:
        #     print('Backward pass, time: %.5f s' % time_elapsed)
        # debug_img = debug_img[:, :, 0]
        # pyredner.imwrite(debug_img, 'debug.exr')
        # pyredner.imwrite(-debug_img, 'debug_.exr')
        # debug_img = debug_img.numpy()
        # print(np.max(debug_img))
        # print(np.unravel_index(np.argmax(debug_img), debug_img.shape))
        # print(np.min(debug_img))
        # print(np.unravel_index(np.argmin(debug_img), debug_img.shape))
        # print(np.sum(debug_img) / 3)
        # debug_max = 0.5
        # debug_min = -0.5
        # debug_img = np.clip((debug_img - debug_min) / (debug_max - debug_min), 0, 1)
        # # debug_img = debug_img[:, :, 0]
        # import matplotlib.cm as cm
        # debug_img = cm.viridis(debug_img)
        # skimage.io.imsave('debug.png', np.power(debug_img, 1/2.2))
        # exit()

        ret_list = []
        ret_list.append(None) # seed
        ret_list.append(None) # num_shapes
        ret_list.append(None) # num_materials
        ret_list.append(None) # num_lights
        if camera.use_look_at:
            ret_list.append(d_cam_position.cpu())
            ret_list.append(d_cam_look.cpu())
            ret_list.append(d_cam_up.cpu())
            ret_list.append(None) # cam_to_world
            ret_list.append(None) # world_to_cam
        else:
            ret_list.append(None) # pos
            ret_list.append(None) # look
            ret_list.append(None) # up
            ret_list.append(d_cam_to_world.cpu())
            ret_list.append(d_world_to_cam.cpu())
        ret_list.append(d_intrinsic_mat_inv.cpu())
        ret_list.append(d_intrinsic_mat.cpu())
        ret_list.append(None) # clip near
        ret_list.append(None) # resolution
        ret_list.append(None) # camera_type

        num_shapes = len(ctx.shapes)
        for i in range(num_shapes):
            ret_list.append(d_vertices_list[i])
            ret_list.append(None) # indices
            ret_list.append(d_uvs_list[i])
            ret_list.append(d_normals_list[i])
            ret_list.append(None) # uv_indices
            ret_list.append(None) # normal_indices
            ret_list.append(d_colors_list[i])
            ret_list.append(None) # material id
            ret_list.append(None) # light id

        num_materials = len(ctx.materials)
        for i in range(num_materials):
            ret_list.append(None) # num_levels
            for d_diffuse in d_diffuse_list[i]:
                ret_list.append(d_diffuse)
            ret_list.append(d_diffuse_uv_scale_list[i])
            ret_list.append(None) # num_levels
            for d_specular in d_specular_list[i]:
                ret_list.append(d_specular)
            ret_list.append(d_specular_uv_scale_list[i])
            ret_list.append(None) # num_levels
            for d_roughness in d_roughness_list[i]:
                ret_list.append(d_roughness)
            ret_list.append(d_roughness_uv_scale_list[i])
            if d_generic_list[i] is None:
                ret_list.append(None) # num_levels
            else:
                ret_list.append(None) # num_levels
                for d_generic in d_generic_list[i]:
                    ret_list.append(d_generic)
                ret_list.append(d_generic_uv_scale_list[i])
            if d_normal_map_list[i] is None:
                ret_list.append(None) # num_levels
            else:
                ret_list.append(None) # num_levels
                for d_normal_map in d_normal_map_list[i]:
                    ret_list.append(d_normal_map)
                ret_list.append(d_normal_map_uv_scale_list[i])
            ret_list.append(None) # compute_specular_lighting
            ret_list.append(None) # two sided
            ret_list.append(None) # use_vertex_color

        num_area_lights = len(ctx.area_lights)
        for i in range(num_area_lights):
            ret_list.append(None) # shape id
            ret_list.append(d_intensity_list[i].cpu())
            ret_list.append(None) # two sided

        if ctx.envmap is not None:
            ret_list.append(None) # num_levels
            for d_values in d_envmap_values:
                ret_list.append(d_values)
            ret_list.append(d_envmap_uv_scale)
            ret_list.append(None) # env_to_world
            ret_list.append(d_world_to_env.cpu())
            ret_list.append(None) # sample_cdf_ys
            ret_list.append(None) # sample_cdf_xs
            ret_list.append(None) # pdf_norm
        else:
            ret_list.append(None)

        ret_list.append(None) # num samples
        ret_list.append(None) # num bounces
        ret_list.append(None) # channels
        ret_list.append(None) # sampler type
        ret_list.append(None) # use_primary_edge_sampling
        ret_list.append(None) # use_secondary_edge_sampling

        return tuple(ret_list)
