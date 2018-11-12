import torch
from torch.autograd import Variable
import numpy as np
import redner
import pyredner

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

class RenderFunction(torch.autograd.Function):
    """
        The PyTorch interface of Redner.
    """

    @staticmethod
    def serialize_scene(scene,
                        num_samples,
                        max_bounces):
        """
            Given a PyRedner scene, convert it to a linear list of argument,
            so that we can use it in PyTorch.
        """
        cam = scene.camera
        num_shapes = len(scene.shapes)
        num_materials = len(scene.materials)
        num_lights = len(scene.lights)
        for light_id, light in enumerate(scene.lights):
            scene.shapes[light.shape_id].light_id = light_id
        args = []
        args.append(num_shapes)
        args.append(num_materials)
        args.append(num_lights)
        args.append(cam.cam_to_world)
        args.append(cam.world_to_cam)
        args.append(cam.fov_factor)
        args.append(cam.clip_near)
        args.append(cam.resolution)
        args.append(cam.fisheye)
        for shape in scene.shapes:
            args.append(shape.vertices)
            args.append(shape.indices)
            args.append(shape.uvs)
            args.append(shape.normals)
            args.append(shape.mat_id)
            args.append(shape.light_id)
        for material in scene.materials:
            args.append(material.diffuse_reflectance)
            args.append(material.specular_reflectance)
            args.append(material.roughness)
            args.append(material.diffuse_uv_scale)
            args.append(material.specular_uv_scale)
            args.append(material.roughness_uv_scale)
            args.append(material.two_sided)
        for light in scene.lights:
            args.append(light.shape_id)
            args.append(light.intensity)
        args.append(num_samples)
        args.append(max_bounces)

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
        fov_factor = args[current_index]
        current_index += 1
        clip_near = args[current_index]
        current_index += 1
        resolution = args[current_index]
        current_index += 1
        fisheye = args[current_index]
        current_index += 1
        camera = redner.Camera(resolution[1],
                               resolution[0],
                               redner.float_ptr(cam_to_world.data_ptr()),
                               redner.float_ptr(world_to_cam.data_ptr()),
                               fov_factor.item(),
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
            shapes.append(redner.Shape(redner.float_ptr(vertices.data_ptr()),
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
            specular_reflectance = args[current_index]
            current_index += 1
            roughness = args[current_index]
            current_index += 1
            diffuse_uv_scale = args[current_index]
            current_index += 1
            specular_uv_scale = args[current_index]
            current_index += 1
            roughness_uv_scale = args[current_index]
            current_index += 1
            two_sided = args[current_index]
            current_index += 1
            if diffuse_reflectance.dim() == 1:
                diffuse_reflectance = redner.Texture3(\
                    redner.float_ptr(diffuse_reflectance.data_ptr()), 0, 0)
            else:
                diffuse_reflectance = redner.Texture3(\
                    redner.float_ptr(diffuse_reflectance.data_ptr()),
                    int(diffuse_reflectance.shape[1]),
                    int(diffuse_reflectance.shape[0]))
            if specular_reflectance.dim() == 1:
                specular_reflectance = redner.Texture3(\
                    redner.float_ptr(specular_reflectance.data_ptr()), 0, 0)
            else:
                specular_reflectance = redner.Texture3(\
                    redner.float_ptr(specular_reflectance.data_ptr()),
                    int(specular_reflectance.shape[1]),
                    int(specular_reflectance.shape[0]))
            if roughness.dim() == 1:
                roughness = redner.Texture1(\
                    redner.float_ptr(roughness.data_ptr()), 0, 0)
            else:
                roughness = redner.Texture1(\
                    redner.float_ptr(roughness.data_ptr()),
                    int(roughness.shape[1]),
                    int(roughness.shape[0]))
            materials.append(redner.Material(\
                diffuse_reflectance,
                specular_reflectance,
                roughness,
                redner.float_ptr(diffuse_uv_scale.data_ptr()),
                redner.float_ptr(specular_uv_scale.data_ptr()),
                redner.float_ptr(roughness_uv_scale.data_ptr()),
                two_sided))

        lights = []
        for i in range(num_lights):
            shape_id = args[current_index]
            current_index += 1
            intensity = args[current_index]
            current_index += 1

            lights.append(redner.Light(shape_id,
                                       redner.float_ptr(intensity.data_ptr())))

        scene = redner.Scene(camera,
                             shapes,
                             materials,
                             lights,
                             pyredner.get_use_gpu())
        num_samples = args[current_index]
        current_index += 1
        max_bounces = args[current_index]
        current_index += 1
        options = redner.RenderOptions(seed, num_samples, max_bounces)
        rendered_image = torch.zeros(resolution[0], resolution[1], 3, device = pyredner.get_device())
        redner.render(scene,
                      options,
                      redner.float_ptr(rendered_image.data_ptr()),
                      redner.float_ptr(0),
                      None,
                      redner.float_ptr(0))

        ctx.shapes = shapes
        ctx.materials = materials
        ctx.lights = lights
        ctx.scene = scene
        ctx.options = options
        return rendered_image

    @staticmethod
    def backward(ctx,
                 grad_img):
        scene = ctx.scene
        options = ctx.options

        d_fov_factor = torch.zeros(1)
        d_cam_to_world = torch.zeros(4, 4)
        d_world_to_cam = torch.zeros(4, 4)
        d_camera = redner.DCamera(redner.float_ptr(d_cam_to_world.data_ptr()),
                                  redner.float_ptr(d_world_to_cam.data_ptr()),
                                  redner.float_ptr(d_fov_factor.data_ptr()))
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
            d_diffuse = torch.zeros(3, device = pyredner.get_device()) if diffuse_size[0] == 0 else \
                        torch.zeros(diffuse_size[1], diffuse_size[0], 3, device = pyredner.get_device())
            d_specular = torch.zeros(3, device = pyredner.get_device()) if specular_size[0] == 0 else \
                         torch.zeros(specular_size[1], specular_size[0], 3, device = pyredner.get_device())
            d_roughness = torch.zeros(1, device = pyredner.get_device()) if roughness_size[0] == 0 else \
                          torch.zeros(roughness_size[1], roughness_size[0], device = pyredner.get_device())
            d_diffuse_list.append(d_diffuse)
            d_specular_list.append(d_specular)
            d_roughness_list.append(d_roughness)
            d_diffuse_tex = redner.Texture3(\
                redner.float_ptr(d_diffuse.data_ptr()), diffuse_size[0], diffuse_size[1])
            d_specular_tex = redner.Texture3(\
                redner.float_ptr(d_specular.data_ptr()), specular_size[0], specular_size[1])
            d_roughness_tex = redner.Texture1(\
                redner.float_ptr(d_roughness.data_ptr()), roughness_size[0], roughness_size[1])
            d_materials.append(redner.DMaterial(d_diffuse_tex, d_specular_tex, d_roughness_tex))

        d_intensity_list = []
        d_lights = []
        for light in ctx.lights:
            d_intensity = torch.zeros(3, device = pyredner.get_device())
            d_intensity_list.append(d_intensity)
            d_lights.append(redner.DLight(redner.float_ptr(d_intensity.data_ptr())))

        d_scene = redner.DScene(d_camera,
                                d_shapes,
                                d_materials,
                                d_lights,
                                pyredner.get_use_gpu())
        if not get_use_correlated_random_number():
            # Decouple the forward/backward random numbers by adding a big prime number
            options.seed += 1000003
        redner.render(scene, options,
                      redner.float_ptr(0),
                      redner.float_ptr(grad_img.data_ptr()),
                      d_scene,
                      redner.float_ptr(0))

        # # For debugging
        # grad_img = torch.ones(256, 256, 3)
        # debug_img = torch.zeros(256, 256, 3)
        # redner.render(scene, options,
        #               redner.float_ptr(0),
        #               redner.float_ptr(grad_img.data_ptr()),
        #               d_scene,
        #               redner.float_ptr(debug_img.data_ptr()))
        # pyredner.imwrite(debug_img, 'debug.exr')
        # exit()

        ret_list = []
        ret_list.append(None) # seed
        ret_list.append(None) # num_shapes
        ret_list.append(None) # num_materials
        ret_list.append(None) # num_lights
        ret_list.append(d_cam_to_world)
        ret_list.append(d_world_to_cam)
        ret_list.append(d_fov_factor)
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
            ret_list.append(d_specular_list[i])
            ret_list.append(d_roughness_list[i])
            ret_list.append(None) # diffuse_uv_scale
            ret_list.append(None) # specular_uv_scale
            ret_list.append(None) # roughness_uv_scale
            ret_list.append(None) # two sided

        num_lights = len(ctx.lights)
        for i in range(num_lights):
            ret_list.append(None) # shape id
            ret_list.append(d_intensity_list[i].cpu())
        
        ret_list.append(None) # num samples
        ret_list.append(None) # num bounces

        return tuple(ret_list)
