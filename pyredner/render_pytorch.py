import torch
from torch.autograd import Variable
import numpy as np
import redner
import pyredner
import time

# There is a bias-variance trade off in the backward pass.
# If the forward pass and the backward pass are correlated
# the gradients are biased for L2 loss.
# (E[d/dx(f(x) - y)^2] = E[(f(x) - y) d/dx f(x)])
#                      = E[f(x) - y] E[d/dx f(x)]
# The last equation only holds when f(x) and d/dx f(x) are independent.
# It is usually better to use the unbiased one, but we left it as an option here
use_correlated_random_number = False
def set_use_correlated_random_number(v):
    global use_correlated_random_number
    use_gpu = v

def get_use_correlated_random_number():
    global use_correlated_random_number
    return use_correlated_random_number

print_timing = True

class RenderFunction(torch.autograd.Function):
    """
        The PyTorch interface of Redner.
    """

    @staticmethod
    def serialize_scene(scene,
                        num_samples,
                        max_bounces,
                        channels = [redner.channels.radiance]):
        """
            Given a PyRedner scene, convert it to a linear list of argument,
            so that we can use it in PyTorch.
        """
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
        args.append(cam.cam_to_world)
        args.append(cam.world_to_cam)
        args.append(cam.ndc_to_cam)
        args.append(cam.cam_to_ndc)
        args.append(cam.clip_near)
        args.append(cam.resolution)
        args.append(cam.fisheye)
        for shape in scene.shapes:
            args.append(shape.vertices)
            args.append(shape.indices)
            args.append(shape.uvs)
            args.append(shape.normals)
            args.append(shape.material_id)
            args.append(shape.light_id)
        for material in scene.materials:
            args.append(material.diffuse_reflectance.mipmap)
            args.append(material.diffuse_reflectance.uv_scale)
            args.append(material.specular_reflectance.mipmap)
            args.append(material.specular_reflectance.uv_scale)
            args.append(material.roughness.mipmap)
            args.append(material.roughness.uv_scale)
            args.append(material.two_sided)
        for light in scene.area_lights:
            args.append(light.shape_id)
            args.append(light.intensity)
            args.append(light.two_sided)
        if scene.envmap is not None:
            args.append(scene.envmap.values.mipmap)
            args.append(scene.envmap.values.uv_scale)
            args.append(scene.envmap.env_to_world)
            args.append(scene.envmap.world_to_env)
            args.append(scene.envmap.sample_cdf_ys)
            args.append(scene.envmap.sample_cdf_xs)
            args.append(scene.envmap.pdf_norm)
        else:
            args.append(None)
            args.append(None)
            args.append(None)
            args.append(None)
            args.append(None)
            args.append(None)
            args.append(None)
        args.append(num_samples)
        args.append(max_bounces)
        args.append(channels)

        return args
    
    @staticmethod
    def forward(ctx,
                seed,
                *args):
        """
            Forward rendering pass: given a scene and output an image.
        """
        # Unpack arguments
        current_index = 0
        num_shapes = args[current_index]
        current_index += 1
        num_materials = args[current_index]
        current_index += 1
        num_lights = args[current_index]
        current_index += 1
        cam_to_world = args[current_index]
        current_index += 1
        world_to_cam = args[current_index]
        current_index += 1
        ndc_to_cam = args[current_index]
        current_index += 1
        cam_to_ndc = args[current_index]
        current_index += 1
        clip_near = args[current_index]
        current_index += 1
        resolution = args[current_index]
        current_index += 1
        fisheye = args[current_index]
        current_index += 1
        assert(cam_to_world.is_contiguous())
        assert(world_to_cam.is_contiguous())
        camera = redner.Camera(resolution[1],
                               resolution[0],
                               redner.float_ptr(cam_to_world.data_ptr()),
                               redner.float_ptr(world_to_cam.data_ptr()),
                               redner.float_ptr(ndc_to_cam.data_ptr()),
                               redner.float_ptr(cam_to_ndc.data_ptr()),
                               clip_near,
                               fisheye)
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
            shapes.append(redner.Shape(\
                redner.float_ptr(vertices.data_ptr()),
                redner.int_ptr(indices.data_ptr()),
                redner.float_ptr(uvs.data_ptr() if uvs is not None else 0),
                redner.float_ptr(normals.data_ptr() if normals is not None else 0),
                int(vertices.shape[0]),
                int(indices.shape[0]),
                material_id,
                light_id))
        materials = []
        for i in range(num_materials):
            diffuse_reflectance = args[current_index]
            current_index += 1
            diffuse_uv_scale = args[current_index]
            current_index += 1
            specular_reflectance = args[current_index]
            current_index += 1
            specular_uv_scale = args[current_index]
            current_index += 1
            roughness = args[current_index]
            current_index += 1
            roughness_uv_scale = args[current_index]
            current_index += 1
            two_sided = args[current_index]
            current_index += 1
            assert(diffuse_reflectance.is_contiguous())
            if diffuse_reflectance.dim() == 1:
                diffuse_reflectance = redner.Texture3(\
                    redner.float_ptr(diffuse_reflectance.data_ptr()), 0, 0, 0,
                    redner.float_ptr(diffuse_uv_scale.data_ptr()))
            else:
                diffuse_reflectance = redner.Texture3(\
                    redner.float_ptr(diffuse_reflectance.data_ptr()),
                    int(diffuse_reflectance.shape[2]), # width
                    int(diffuse_reflectance.shape[1]), # height
                    int(diffuse_reflectance.shape[0]), # num levels
                    redner.float_ptr(diffuse_uv_scale.data_ptr()))
            assert(specular_reflectance.is_contiguous())
            if specular_reflectance.dim() == 1:
                specular_reflectance = redner.Texture3(\
                    redner.float_ptr(specular_reflectance.data_ptr()), 0, 0, 0,
                    redner.float_ptr(specular_uv_scale.data_ptr()))
            else:
                specular_reflectance = redner.Texture3(\
                    redner.float_ptr(specular_reflectance.data_ptr()),
                    int(specular_reflectance.shape[2]), # width
                    int(specular_reflectance.shape[1]), # height
                    int(specular_reflectance.shape[0]), # num levels
                    redner.float_ptr(specular_uv_scale.data_ptr()))
            assert(roughness.is_contiguous())
            if roughness.dim() == 1:
                roughness = redner.Texture1(\
                    redner.float_ptr(roughness.data_ptr()), 0, 0, 0,
                    redner.float_ptr(roughness_uv_scale.data_ptr()))
            else:
                assert(roughness.dim() == 4)
                roughness = redner.Texture1(\
                    redner.float_ptr(roughness.data_ptr()),
                    int(roughness.shape[2]), # width
                    int(roughness.shape[1]), # height
                    int(roughness.shape[0]), # num levels
                    redner.float_ptr(roughness_uv_scale.data_ptr()))
            materials.append(redner.Material(\
                diffuse_reflectance,
                specular_reflectance,
                roughness,
                two_sided))

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
            values = args[current_index]
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
                redner.float_ptr(values.data_ptr()),
                int(values.shape[2]), # width
                int(values.shape[1]), # height
                int(values.shape[0]), # num levels
                redner.float_ptr(envmap_uv_scale.data_ptr()))
            envmap = redner.EnvironmentMap(\
                values,
                redner.float_ptr(env_to_world.data_ptr()),
                redner.float_ptr(world_to_env.data_ptr()),
                redner.float_ptr(sample_cdf_ys.data_ptr()),
                redner.float_ptr(sample_cdf_xs.data_ptr()),
                pdf_norm)
        else:
            current_index += 7

        scene = redner.Scene(camera,
                             shapes,
                             materials,
                             area_lights,
                             envmap,
                             pyredner.get_use_gpu())
        num_samples = args[current_index]
        current_index += 1
        max_bounces = args[current_index]
        current_index += 1
        channels = args[current_index]
        current_index += 1

        options = redner.RenderOptions(seed, num_samples, max_bounces, channels)
        num_channels = redner.compute_num_channels(channels)
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

        ctx.shapes = shapes
        ctx.materials = materials
        ctx.area_lights = area_lights
        ctx.envmap = envmap
        ctx.scene = scene
        ctx.options = options
        return rendered_image

    @staticmethod
    def backward(ctx,
                 grad_img):
        if not grad_img.is_contiguous():
            grad_img = grad_img.contiguous()
        scene = ctx.scene
        options = ctx.options

        d_cam_to_world = torch.zeros(4, 4)
        d_world_to_cam = torch.zeros(4, 4)
        d_ndc_to_cam = torch.zeros(3, 3)
        d_cam_to_ndc = torch.zeros(3, 3)
        d_camera = redner.DCamera(redner.float_ptr(d_cam_to_world.data_ptr()),
                                  redner.float_ptr(d_world_to_cam.data_ptr()),
                                  redner.float_ptr(d_ndc_to_cam.data_ptr()),
                                  redner.float_ptr(d_cam_to_ndc.data_ptr()))
        d_vertices_list = []
        d_uvs_list = []
        d_normals_list = []
        d_shapes = []
        for shape in ctx.shapes:
            num_vertices = shape.num_vertices
            d_vertices = torch.zeros(num_vertices, 3,
                device = pyredner.get_device())
            d_uvs = torch.zeros(num_vertices, 2,
                device = pyredner.get_device()) if shape.has_uvs() else None
            d_normals = torch.zeros(num_vertices, 3,
                device = pyredner.get_device()) if shape.has_normals() else None
            d_vertices_list.append(d_vertices)
            d_uvs_list.append(d_uvs)
            d_normals_list.append(d_normals)
            d_shapes.append(redner.DShape(\
                redner.float_ptr(d_vertices.data_ptr()),
                redner.float_ptr(d_uvs.data_ptr() if d_uvs is not None else 0),
                redner.float_ptr(d_normals.data_ptr() if d_normals is not None else 0)))

        d_diffuse_list = []
        d_specular_list = []
        d_roughness_list = []
        d_materials = []
        for material in ctx.materials:
            diffuse_size = material.get_diffuse_size()
            specular_size = material.get_specular_size()
            roughness_size = material.get_roughness_size()
            if diffuse_size[0] == 0:
                d_diffuse = torch.zeros(3, device = pyredner.get_device())
            else:
                d_diffuse = torch.zeros(diffuse_size[2],
                                        diffuse_size[1],
                                        diffuse_size[0],
                                        3, device = pyredner.get_device())
            if specular_size[0] == 0:
                d_specular = torch.zeros(3, device = pyredner.get_device())
            else:
                d_specular = torch.zeros(specular_size[2],
                                         specular_size[1],
                                         specular_size[0],
                                         3, device = pyredner.get_device())
            if roughness_size[0] == 0:
                d_roughness = torch.zeros(1, device = pyredner.get_device())
            else:
                d_roughness = torch.zeros(roughness_size[2],
                                          roughness_size[1],
                                          roughness_size[0],
                                          1, device = pyredner.get_device())
            d_diffuse_list.append(d_diffuse)
            d_specular_list.append(d_specular)
            d_roughness_list.append(d_roughness)
            d_diffuse_uv_scale = torch.zeros(2)
            d_specular_uv_scale = torch.zeros(2)
            d_roughness_uv_scale = torch.zeros(2)
            d_diffuse_tex = redner.Texture3(\
                redner.float_ptr(d_diffuse.data_ptr()),
                diffuse_size[0], diffuse_size[1], diffuse_size[2],
                redner.float_ptr(d_diffuse_uv_scale.data_ptr()))
            d_specular_tex = redner.Texture3(\
                redner.float_ptr(d_specular.data_ptr()),
                specular_size[0], specular_size[1], specular_size[2],
                redner.float_ptr(d_specular_uv_scale.data_ptr()))
            d_roughness_tex = redner.Texture1(\
                redner.float_ptr(d_roughness.data_ptr()),
                roughness_size[0], roughness_size[1], roughness_size[2],
                redner.float_ptr(d_roughness_uv_scale.data_ptr()))
            d_materials.append(redner.DMaterial(\
                d_diffuse_tex, d_specular_tex, d_roughness_tex))

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
            size = envmap.get_size()
            d_envmap_values = \
                torch.zeros(size[2],
                            size[1],
                            size[0],
                            3,
                            device = pyredner.get_device())
            d_envmap_uv_scale = torch.zeros(2)
            d_envmap_tex = redner.Texture3(\
                redner.float_ptr(d_envmap_values.data_ptr()),
                size[0], size[1], size[2],
                redner.float_ptr(d_envmap_uv_scale.data_ptr()))
            d_world_to_env = torch.zeros(4, 4)
            d_envmap = redner.DEnvironmentMap(\
                d_envmap_tex,
                redner.float_ptr(d_world_to_env.data_ptr()))

        d_scene = redner.DScene(d_camera,
                                d_shapes,
                                d_materials,
                                d_area_lights,
                                d_envmap,
                                pyredner.get_use_gpu())
        if not get_use_correlated_random_number():
            # Decouple the forward/backward random numbers by adding a big prime number
            options.seed += 1000003
        start = time.time()
        redner.render(scene, options,
                      redner.float_ptr(0),
                      redner.float_ptr(grad_img.data_ptr()),
                      d_scene,
                      redner.float_ptr(0))
        time_elapsed = time.time() - start
        if print_timing:
            print('Backward pass, time: %.5f s' % time_elapsed)

        # # For debugging
        # pyredner.imwrite(grad_img, 'grad_img.exr')
        # grad_img = torch.ones(256, 256, 3)
        # debug_img = torch.zeros(256, 256, 3)
        # redner.render(scene, options,
        #               redner.float_ptr(0),
        #               redner.float_ptr(grad_img.data_ptr()),
        #               d_scene,
        #               redner.float_ptr(debug_img.data_ptr()))
        # pyredner.imwrite(debug_img, 'debug.exr')
        # pyredner.imwrite(-debug_img, 'debug_.exr')
        # exit()

        ret_list = []
        ret_list.append(None) # seed
        ret_list.append(None) # num_shapes
        ret_list.append(None) # num_materials
        ret_list.append(None) # num_lights
        ret_list.append(d_cam_to_world)
        ret_list.append(d_world_to_cam)
        ret_list.append(d_ndc_to_cam)
        ret_list.append(d_cam_to_ndc)
        ret_list.append(None) # clip near
        ret_list.append(None) # resolution
        ret_list.append(None) # fisheye

        num_shapes = len(ctx.shapes)
        for i in range(num_shapes):
            ret_list.append(d_vertices_list[i])
            ret_list.append(None) # indices
            ret_list.append(d_uvs_list[i])
            ret_list.append(d_normals_list[i])
            ret_list.append(None) # material id
            ret_list.append(None) # light id

        num_materials = len(ctx.materials)
        for i in range(num_materials):
            ret_list.append(d_diffuse_list[i])
            ret_list.append(None) # diffuse_uv_scale
            ret_list.append(d_specular_list[i])
            ret_list.append(None) # specular_uv_scale
            ret_list.append(d_roughness_list[i])
            ret_list.append(None) # roughness_uv_scale
            ret_list.append(None) # two sided

        num_area_lights = len(ctx.area_lights)
        for i in range(num_area_lights):
            ret_list.append(None) # shape id
            ret_list.append(d_intensity_list[i].cpu())
            ret_list.append(None) # two sided

        if ctx.envmap is not None:
            ret_list.append(d_envmap_values)
            ret_list.append(None) # uv_scale
            ret_list.append(None) # env_to_world
            ret_list.append(d_world_to_env)
            ret_list.append(None) # sample_cdf_ys
            ret_list.append(None) # sample_cdf_xs
            ret_list.append(None) # pdf_norm
        else:
            ret_list.append(None)
            ret_list.append(None)
            ret_list.append(None)
            ret_list.append(None)
            ret_list.append(None)
            ret_list.append(None)
            ret_list.append(None)
        
        ret_list.append(None) # num samples
        ret_list.append(None) # num bounces
        ret_list.append(None) # channels

        return tuple(ret_list)
