from typing import List
import tensorflow as tf
import numpy as np
import redner
import pyredner_tensorflow as pyredner
import time
import weakref
import os

__EMPTY_TENSOR = tf.constant([])
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
    use_correlated_random_number = v

def get_use_correlated_random_number():
    global use_correlated_random_number
    return use_correlated_random_number

def get_tensor_dimension(t):
    """Return dimension of the TF tensor in Int

    `get_shape()` returns `TensorShape`.
    
    """
    return len(t.get_shape())

def is_empty_tensor(tensor):
    return  tf.equal(tf.size(tensor), 0)
    

class Context: pass

__ctx = Context()
print_timing = True

def serialize_scene(scene: pyredner.Scene,
                    num_samples: int,
                    max_bounces: int,
                    channels = [redner.channels.radiance],
                    sampler_type = redner.SamplerType.independent,
                    use_primary_edge_sampling = True,
                    use_secondary_edge_sampling = True) -> List:
    """
        Given a PyRedner scene & rendering options, convert them to a linear list of argument,
        so that we can use it in TensorFlow.

        Keyword arguments:
        scene -- A pyredner.Scene
        num_samples -- Number of samples per pixel for forward and backward passes,
                        can be an integer or a tuple of 2 integers.
        max_bounces -- Number of bounces for global illumination, 1 means direct lighting only.
        channels -- A list of channels that should present in the output image.
                    Following channels are supported:
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
                        redner.channels.shape_id,
                        redner.channels.material_id
                    All channels, except for shape id and material id, are differentiable.
        sampler_type -- Which sampling pattern to use.
                        See Chapter 7 of the PBRT book for an explanation of the difference between
                        different samplers.
                        http://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction.html
                        Following samplers are supported:
                            redner.SamplerType.independent
                            redner.SamplerType.sobol
        use_primary_edge_sampling -- A boolean
        use_secondary_edge_sampling -- A boolean

        tf.custom_gradient in Tensorflow can take only tf.Tensor objects as arguments.
        Hense, map `None` to False boolean tensors
    """
    global __ctx
    ctx = __ctx

    ctx.pyredner_scene = scene

    cam = scene.camera
    num_shapes = len(scene.shapes)
    num_materials = len(scene.materials)
    num_lights = len(scene.area_lights)
    num_channels = len(channels)

    for light_id, light in enumerate(scene.area_lights):
        scene.shapes[light.shape_id].light_id = light_id

    args = []
    args.append(tf.constant(num_shapes))
    args.append(tf.constant(num_materials))
    args.append(tf.constant(num_lights))
    if cam.position is None:
        args.append(__EMPTY_TENSOR)
        args.append(__EMPTY_TENSOR)
        args.append(__EMPTY_TENSOR)
    else:
        args.append(cam.position)
        args.append(cam.look_at)
        args.append(cam.up)
    if cam.cam_to_world is None:
        args.append(__EMPTY_TENSOR)
        args.append(__EMPTY_TENSOR)
    else:
        args.append(cam.cam_to_world)
        args.append(cam.world_to_cam)
    args.append(cam.ndc_to_cam)
    args.append(cam.cam_to_ndc)
    args.append(tf.constant(cam.clip_near))
    args.append(tf.constant(cam.resolution))
    args.append(pyredner.RednerCameraType.asTensor(cam.camera_type))
    for shape in scene.shapes:
        args.append(shape.vertices)
        args.append(shape.indices)
        if shape.uvs is None:
            args.append(__EMPTY_TENSOR)
        else:
            args.append(shape.uvs) 
        if shape.normals is None:
            args.append(__EMPTY_TENSOR)
        else:
            args.append(shape.normals) 
        if shape.uv_indices is None:
            args.append(__EMPTY_TENSOR)
        else:
            args.append(shape.uv_indices)
        if shape.normal_indices is None:
            args.append(__EMPTY_TENSOR)
        else:
            args.append(shape.normal_indices)
        args.append(tf.constant(shape.material_id))
        args.append(tf.constant(shape.light_id))
    for material in scene.materials: 
        args.append(material.diffuse_reflectance.mipmap)
        args.append(material.diffuse_reflectance.uv_scale)
        args.append(material.specular_reflectance.mipmap)
        args.append(material.specular_reflectance.uv_scale)
        args.append(material.roughness.mipmap)
        args.append(material.roughness.uv_scale)
        if material.normal_map is not None:
            args.append(material.normal_map.mipmap)
            args.append(material.normal_map.uv_scale)
        else:
            args.append(__EMPTY_TENSOR)
            args.append(__EMPTY_TENSOR)
        args.append(tf.constant(material.two_sided))
    for light in scene.area_lights:
        args.append(tf.constant(light.shape_id))
        args.append(light.intensity)
        args.append(tf.constant(light.two_sided))
    if scene.envmap is not None:
        args.append(scene.envmap.values.mipmap)
        args.append(scene.envmap.values.uv_scale)
        args.append(scene.envmap.env_to_world)
        args.append(scene.envmap.world_to_env)
        args.append(scene.envmap.sample_cdf_ys)
        args.append(scene.envmap.sample_cdf_xs)
        args.append(scene.envmap.pdf_norm)
    else:
        args.append(__EMPTY_TENSOR)
        args.append(__EMPTY_TENSOR)
        args.append(__EMPTY_TENSOR)
        args.append(__EMPTY_TENSOR)
        args.append(__EMPTY_TENSOR)
        args.append(__EMPTY_TENSOR)
        args.append(__EMPTY_TENSOR)

    args.append(tf.constant(num_samples))
    args.append(tf.constant(max_bounces))
    args.append(tf.constant(num_channels))
    for ch in channels:
        args.append(pyredner.RednerChannels.asTensor(ch))

    args.append(pyredner.RednerSamplerType.asTensor(sampler_type))
    args.append(tf.constant(use_primary_edge_sampling))
    args.append(tf.constant(use_secondary_edge_sampling))
    return args

def forward(seed:int, *args):
    """
        Forward rendering pass: given a scene and output an image.
    """
    global __ctx
    ctx = __ctx

    # Unpack arguments
    current_index = 0
    num_shapes = int(args[current_index])
    current_index += 1
    num_materials = int(args[current_index])
    current_index += 1
    num_lights = int(args[current_index])
    current_index += 1

    # Camera arguments
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
    ndc_to_cam = args[current_index]
    current_index += 1
    cam_to_ndc = args[current_index]
    current_index += 1
    clip_near = float(args[current_index])
    current_index += 1
    resolution = args[current_index].numpy() # Tuple[int, int]
    current_index += 1
    camera_type = pyredner.RednerCameraType.asCameraType(args[current_index]) # FIXME: Map to custom type
    current_index += 1

    with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
        if is_empty_tensor(cam_to_world):
            camera = redner.Camera(resolution[1],
                                   resolution[0],
                                   redner.float_ptr(pyredner.data_ptr(cam_position)),
                                   redner.float_ptr(pyredner.data_ptr(cam_look_at)),
                                   redner.float_ptr(pyredner.data_ptr(cam_up)),
                                   redner.float_ptr(0), # cam_to_world
                                   redner.float_ptr(0), # world_to_cam
                                   redner.float_ptr(pyredner.data_ptr(ndc_to_cam)),
                                   redner.float_ptr(pyredner.data_ptr(cam_to_ndc)),
                                   clip_near,
                                   camera_type)
        else:
            camera = redner.Camera(resolution[1],
                                   resolution[0],
                                   redner.float_ptr(0),
                                   redner.float_ptr(0),
                                   redner.float_ptr(0),
                                   redner.float_ptr(pyredner.data_ptr(cam_to_world)),
                                   redner.float_ptr(pyredner.data_ptr(world_to_cam)),
                                   redner.float_ptr(pyredner.data_ptr(ndc_to_cam)),
                                   redner.float_ptr(pyredner.data_ptr(cam_to_ndc)),
                                   clip_near,
                                   camera_type)

    with tf.device(pyredner.get_device_name()):
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
            material_id = int(args[current_index])
            current_index += 1
            light_id = int(args[current_index])
            current_index += 1
            shapes.append(redner.Shape(\
                redner.float_ptr(pyredner.data_ptr(vertices)),
                redner.int_ptr(pyredner.data_ptr(indices)),
                redner.float_ptr(pyredner.data_ptr(uvs) if uvs is not None else 0),
                redner.float_ptr(pyredner.data_ptr(normals) if normals is not None else 0),
                redner.int_ptr(pyredner.data_ptr(uv_indices) if uv_indices is not None else 0),
                redner.int_ptr(pyredner.data_ptr(normal_indices) if normal_indices is not None else 0),
                int(vertices.shape[0]),
                int(uvs.shape[0]) if uvs is not None else 0,
                int(normals.shape[0]) if normals is not None else 0,
                int(indices.shape[0]),
                material_id,
                light_id))

    materials = []
    with tf.device(pyredner.get_device_name()):
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
            normal_map = args[current_index]
            current_index += 1
            normal_map_uv_scale = args[current_index]
            current_index += 1
            two_sided = bool(args[current_index])
            current_index += 1
        
            diffuse_reflectance_ptr = redner.float_ptr(pyredner.data_ptr(diffuse_reflectance))
            specular_reflectance_ptr = redner.float_ptr(pyredner.data_ptr(specular_reflectance))
            roughness_ptr = redner.float_ptr(pyredner.data_ptr(roughness))
            if normal_map.shape[0] > 0:
                normal_map_ptr = redner.float_ptr(pyredner.data_ptr(normal_map))
            diffuse_uv_scale_ptr = redner.float_ptr(pyredner.data_ptr(diffuse_uv_scale))
            specular_uv_scale_ptr = redner.float_ptr(pyredner.data_ptr(specular_uv_scale))
            roughness_uv_scale_ptr = redner.float_ptr(pyredner.data_ptr(roughness_uv_scale))
            if normal_map.shape[0] > 0:
                normal_map_uv_scale_ptr = redner.float_ptr(pyredner.data_ptr(normal_map_uv_scale))
            if get_tensor_dimension(diffuse_reflectance) == 1:
                diffuse_reflectance = redner.Texture3(diffuse_reflectance_ptr, 0, 0, 0, diffuse_uv_scale_ptr)
            else:
                diffuse_reflectance = redner.Texture3(\
                    diffuse_reflectance_ptr,
                    int(diffuse_reflectance.shape[2]), # width
                    int(diffuse_reflectance.shape[1]), # height
                    int(diffuse_reflectance.shape[0]), # num levels
                    diffuse_uv_scale_ptr)
            if get_tensor_dimension(specular_reflectance) == 1:
                specular_reflectance = redner.Texture3(specular_reflectance_ptr, 0, 0, 0, specular_uv_scale_ptr)
            else:
                specular_reflectance = redner.Texture3(\
                    specular_reflectance_ptr,
                    int(specular_reflectance.shape[2]), # width
                    int(specular_reflectance.shape[1]), # height
                    int(specular_reflectance.shape[0]), # num levels
                    specular_uv_scale_ptr)
            if get_tensor_dimension(roughness) == 1:
                roughness = redner.Texture1(roughness_ptr, 0, 0, 0, roughness_uv_scale_ptr)
            else:
                assert(get_tensor_dimension(roughness) == 4)
                roughness = redner.Texture1(\
                    roughness_ptr,
                    int(roughness.shape[2]), # width
                    int(roughness.shape[1]), # height
                    int(roughness.shape[0]), # num levels
                    roughness_uv_scale_ptr)
            if normal_map.shape[0] > 0:
                normal_map = redner.Texture3(\
                    normal_map_ptr,
                    int(normal_map.shape[2]),
                    int(normal_map.shape[1]),
                    int(normal_map.shape[0]),
                    normal_map_uv_scale_ptr)
            else:
                normal_map = redner.Texture3(\
                    redner.float_ptr(0), 0, 0, 0, redner.float_ptr(0))
            materials.append(redner.Material(\
                diffuse_reflectance,
                specular_reflectance,
                roughness,
                normal_map,
                two_sided))

    with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
        area_lights = []
        for i in range(num_lights):
            shape_id = int(args[current_index])
            current_index += 1
            intensity = args[current_index]
            current_index += 1
            two_sided = bool(args[current_index])
            current_index += 1

            area_lights.append(redner.AreaLight(
                shape_id,
                redner.float_ptr(pyredner.data_ptr(intensity)),
                two_sided))

    envmap = None
    if not is_empty_tensor(args[current_index]):
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
        pdf_norm = float(args[current_index])
        current_index += 1

        assert isinstance(pdf_norm, float)
        with tf.device(pyredner.get_device_name()):
            values_ptr = redner.float_ptr(pyredner.data_ptr(values))
            sample_cdf_ys = redner.float_ptr(pyredner.data_ptr(sample_cdf_ys))
            sample_cdf_xs = redner.float_ptr(pyredner.data_ptr(sample_cdf_xs))
            envmap_uv_scale = redner.float_ptr(pyredner.data_ptr(envmap_uv_scale))
        with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
            env_to_world = redner.float_ptr(pyredner.data_ptr(env_to_world))
            world_to_env = redner.float_ptr(pyredner.data_ptr(world_to_env))
        values = redner.Texture3(
            values_ptr,
            int(values.shape[2]), # width
            int(values.shape[1]), # height
            int(values.shape[0]), # num levels
            envmap_uv_scale)
        envmap = redner.EnvironmentMap(\
            values,
            env_to_world,
            world_to_env,
            sample_cdf_ys,
            sample_cdf_xs,
            pdf_norm)
    else:
        current_index += 7

    # Options
    num_samples = int(args[current_index])
    current_index += 1
    max_bounces = int(args[current_index])
    current_index += 1

    __num_channels = int(args[current_index])
    current_index += 1

    channels = []
    for _ in range(__num_channels):
        ch = args[current_index]
        ch = pyredner.RednerChannels.asChannel(ch)
        channels.append(ch)
        current_index += 1

    sampler_type = args[current_index]
    sampler_type = pyredner.RednerSamplerType.asSamplerType(sampler_type)
    current_index += 1

    use_primary_edge_sampling = args[current_index]
    current_index += 1
    use_secondary_edge_sampling = args[current_index]
    current_index += 1

    scene = redner.Scene(camera,
                         shapes,
                         materials,
                         area_lights,
                         envmap,
                         pyredner.get_use_gpu(),
                         pyredner.get_gpu_device_id(),
                         use_primary_edge_sampling,
                         use_secondary_edge_sampling)

    # check that num_samples is a tuple
    if isinstance(num_samples, int):
        num_samples = (num_samples, num_samples)

    options = redner.RenderOptions(seed, 
                                    num_samples[0], 
                                    max_bounces, 
                                    channels,
                                    sampler_type)
    num_channels = redner.compute_num_channels(channels)

    with tf.device(pyredner.get_device_name()):
        rendered_image = tf.zeros(
            shape=[resolution[0], resolution[1], num_channels], 
            dtype=tf.float32)

        start = time.time()

        # pdb.set_trace()
        redner.render(scene,
                      options,
                      redner.float_ptr(pyredner.data_ptr(rendered_image)),
                      redner.float_ptr(0),
                      None,
                      redner.float_ptr(0))
        time_elapsed = time.time() - start
        if print_timing:
            print('Forward pass, time: %.5f s' % time_elapsed)

        # # For debugging
        # debug_img = tf.zeros((256, 256, 3), dtype=tf.float32)
        # redner.render(scene,
        #               options,
        #               redner.float_ptr(pyredner.data_ptr(rendered_image)),
        #               redner.float_ptr(0),
        #               None,
        #               redner.float_ptr(pyredner.data_ptr(debug_img)))
        # pyredner.imwrite(debug_img, 'debug.png')
        # exit()

        # import pdb; pdb.set_trace()

    ctx.camera = camera
    ctx.shapes = shapes
    ctx.materials = materials
    ctx.area_lights = area_lights
    ctx.envmap = envmap
    ctx.scene = scene
    ctx.options = options
    ctx.num_samples = num_samples
    ctx.num_channels = __num_channels
    return rendered_image

@tf.custom_gradient
def render(*x):
    assert(tf.executing_eagerly())
    if pyredner.get_use_gpu() and os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] != 'true':
        print('******************** WARNING ********************')
        print('Tensorflow by default allocates all GPU memory,')
        print('causing huge amount of page faults when rendering.')
        print('Please set the environment variable TF_FORCE_GPU_ALLOW_GROWTH to true,')
        print('so that Tensorflow allocates memory on demand.')
        print('*************************************************')

    seed, args = int(x[0]), x[1:]
    img = forward(seed, *args)

    def backward(grad_img):
        global __ctx
        ctx = __ctx
        camera = ctx.camera
        scene = ctx.scene
        options = ctx.options

        with tf.device(pyredner.get_device_name()):
            if camera.use_look_at:
                d_position = tf.zeros(3, dtype=tf.float32)
                d_look_at = tf.zeros(3, dtype=tf.float32)
                d_up = tf.zeros(3, dtype=tf.float32)
                d_cam_to_world = None
                d_wolrd_to_cam = None
            else:
                d_position = None
                d_look_at = None
                d_up = None
                d_cam_to_world = tf.zeros([4, 4], dtype=tf.float32)
                d_wolrd_to_cam = tf.zeros([4, 4], dtype=tf.float32)
            d_ndc_to_cam = tf.zeros([3,3], dtype=tf.float32)
            d_cam_to_ndc = tf.zeros([3,3], dtype=tf.float32)
            if camera.use_look_at:
                d_camera = redner.DCamera(redner.float_ptr(pyredner.data_ptr(d_position)),
                                          redner.float_ptr(pyredner.data_ptr(d_look_at)),
                                          redner.float_ptr(pyredner.data_ptr(d_up)),
                                          redner.float_ptr(0), # cam_to_world
                                          redner.float_ptr(0), # world_to_cam
                                          redner.float_ptr(pyredner.data_ptr(d_ndc_to_cam)),
                                          redner.float_ptr(pyredner.data_ptr(d_cam_to_ndc)))
            else:
                d_camera = redner.DCamera(redner.float_ptr(0),
                                          redner.float_ptr(0),
                                          redner.float_ptr(0),
                                          redner.float_ptr(pyredner.data_ptr(d_cam_to_world)),
                                          redner.float_ptr(pyredner.data_ptr(d_world_to_cam)),
                                          redner.float_ptr(pyredner.data_ptr(d_ndc_to_cam)),
                                          redner.float_ptr(pyredner.data_ptr(d_cam_to_ndc)))

        d_vertices_list = []
        d_uvs_list = []
        d_normals_list = []
        d_shapes = []
        with tf.device(pyredner.get_device_name()):
            for i, shape in enumerate(ctx.shapes):
                num_vertices = shape.num_vertices
                d_vertices = tf.zeros([num_vertices, 3], dtype=tf.float32)
                d_uvs = tf.zeros([num_vertices, 2], dtype=tf.float32) if shape.has_uvs() else None
                d_normals = tf.zeros([num_vertices, 3], dtype=tf.float32) if shape.has_normals() else None
                d_vertices_list.append(d_vertices)
                d_uvs_list.append(d_uvs)
                d_normals_list.append(d_normals)
                d_shapes.append(redner.DShape(\
                    redner.float_ptr(pyredner.data_ptr(d_vertices)),
                    redner.float_ptr(pyredner.data_ptr(d_uvs) if d_uvs is not None else 0),
                    redner.float_ptr(pyredner.data_ptr(d_normals) if d_normals is not None else 0)))

        d_diffuse_list = []
        d_specular_list = []
        d_roughness_list = []
        d_normal_map_list = []
        d_diffuse_uv_scale_list = []
        d_specular_uv_scale_list = []
        d_roughness_uv_scale_list = []
        d_normal_map_uv_scale_list = []
        d_materials = []
        with tf.device(pyredner.get_device_name()):
            for material in ctx.materials:
                diffuse_size = material.get_diffuse_size()
                specular_size = material.get_specular_size()
                roughness_size = material.get_roughness_size()
                normal_map_size = material.get_normal_map_size()
                if diffuse_size[0] == 0:
                    d_diffuse = tf.zeros(3, dtype=tf.float32)
                else:
                    d_diffuse = tf.zeros([diffuse_size[2],
                                          diffuse_size[1],
                                          diffuse_size[0],
                                          3], dtype=tf.float32)
                if specular_size[0] == 0:
                    d_specular = tf.zeros(3, dtype=tf.float32)
                else:
                    d_specular = tf.zeros([specular_size[2],
                                           specular_size[1],
                                           specular_size[0],
                                           3], dtype=tf.float32)
                if roughness_size[0] == 0:
                    d_roughness = tf.zeros(1, dtype=tf.float32)
                else:
                    d_roughness = tf.zeros([roughness_size[2],
                                            roughness_size[1],
                                            roughness_size[0],
                                            1], dtype=tf.float32)
                # HACK: tensorflow's eager mode uses a cache to store scalar
                #       constants to avoid memory copy. If we pass scalar tensors
                #       into the C++ code and modify them, we would corrupt the
                #       cache, causing incorrect result in future scalar constant
                #       creations. Thus we force tensorflow to copy by plusing a zero
                # (also see https://github.com/tensorflow/tensorflow/issues/11186
                #  for more discussion regarding copying tensors)
                if d_roughness.shape.num_elements() == 1:
                    d_roughness = d_roughness + 0
                if normal_map_size[0] == 0:
                    d_normal_map = None
                else:
                    d_normal_map = tf.zeros([normal_map_size[2],
                                             normal_map_size[1],
                                             normal_map_size[0],
                                             3], dtype=tf.float32)
                    
                d_diffuse_list.append(d_diffuse)
                d_specular_list.append(d_specular)
                d_roughness_list.append(d_roughness)
                d_normal_map_list.append(d_normal_map)
                d_diffuse = redner.float_ptr(pyredner.data_ptr(d_diffuse))
                d_specular = redner.float_ptr(pyredner.data_ptr(d_specular))
                d_roughness = redner.float_ptr(pyredner.data_ptr(d_roughness))
                if normal_map_size[0] > 0:
                    d_normal_map = redner.float_ptr(pyredner.data_ptr(d_normal_map))
                d_diffuse_uv_scale = tf.zeros([2], dtype=tf.float32)
                d_specular_uv_scale = tf.zeros([2], dtype=tf.float32)
                d_roughness_uv_scale = tf.zeros([2], dtype=tf.float32)
                if normal_map_size[0] > 0:
                    d_normal_map_uv_scale = tf.zeros([2], dtype=tf.float32)
                else:
                    d_normal_map_uv_scale = None
                d_diffuse_uv_scale_list.append(d_diffuse_uv_scale)
                d_specular_uv_scale_list.append(d_specular_uv_scale)
                d_roughness_uv_scale_list.append(d_roughness_uv_scale)
                d_normal_map_uv_scale_list.append(d_normal_map_uv_scale)
                d_diffuse_uv_scale = redner.float_ptr(pyredner.data_ptr(d_diffuse_uv_scale))
                d_specular_uv_scale = redner.float_ptr(pyredner.data_ptr(d_specular_uv_scale))
                d_roughness_uv_scale = redner.float_ptr(pyredner.data_ptr(d_roughness_uv_scale))
                if normal_map_size[0] > 0:
                    d_normal_map_uv_scale = redner.float_ptr(pyredner.data_ptr(d_normal_map_uv_scale))
                d_diffuse_tex = redner.Texture3(\
                    d_diffuse, diffuse_size[0], diffuse_size[1], diffuse_size[2], d_diffuse_uv_scale)
                d_specular_tex = redner.Texture3(\
                    d_specular, specular_size[0], specular_size[1], specular_size[2], d_specular_uv_scale)
                d_roughness_tex = redner.Texture1(\
                    d_roughness, roughness_size[0], roughness_size[1], roughness_size[2],  d_roughness_uv_scale)
                if normal_map_size[0] > 0:
                    d_normal_map_tex = redner.Texture3(\
                        d_normal_map, normal_map_size[0], normal_map_size[1], normal_map_size[2], d_normal_map_uv_scale)
                else:
                    d_normal_map_tex = redner.Texture3(\
                        redner.float_ptr(0), 0, 0, 0, redner.float_ptr(0))
                d_materials.append(redner.DMaterial(d_diffuse_tex, d_specular_tex, d_roughness_tex, d_normal_map_tex))

        d_intensity_list = []
        d_area_lights = []
        with tf.device(pyredner.get_device_name()):
            for light in ctx.area_lights:
                d_intensity = tf.zeros(3, dtype=tf.float32)
                d_intensity_list.append(d_intensity)
                d_area_lights.append(\
                    redner.DAreaLight(redner.float_ptr(pyredner.data_ptr(d_intensity))))

        d_envmap = None
        if ctx.envmap is not None:
            envmap = ctx.envmap
            size = envmap.get_size()
            with tf.device(pyredner.get_device_name()):
                d_envmap_values = tf.zeros([size[2], size[1], size[0], 3], dtype=tf.float32)
                d_envmap_values_ptr = redner.float_ptr(pyredner.data_ptr(d_envmap_values))
                d_envmap_uv_scale = tf.zeros([2], dtype=tf.float32)
                d_envmap_uv_scale_ptr = redner.float_ptr(pyredner.data_ptr(d_envmap_uv_scale))
                d_world_to_env = tf.zeros([4, 4], dtype=tf.float32)
                d_world_to_env_ptr = redner.float_ptr(pyredner.data_ptr(d_world_to_env))
            d_envmap_tex = redner.Texture3(\
                d_envmap_values_ptr, size[0], size[1], size[2], d_envmap_uv_scale_ptr)
            d_envmap = redner.DEnvironmentMap(d_envmap_tex, d_world_to_env_ptr)

        d_scene = redner.DScene(d_camera,
                                d_shapes,
                                d_materials,
                                d_area_lights,
                                d_envmap,
                                pyredner.get_use_gpu(),
                                -1)
        if not get_use_correlated_random_number():
            # Decod_uple the forward/backward random numbers by adding a big prime number
            options.seed += 1000003
        start = time.time()

        options.num_samples = ctx.num_samples[1]
        with tf.device(pyredner.get_device_name()):
            if pyredner.get_use_gpu():
                grad_img = grad_img.gpu(pyredner.get_gpu_device_id())
            else:
                grad_img = grad_img.cpu()
            redner.render(scene,  
                          options,
                          redner.float_ptr(0),    # rendered_image
                          redner.float_ptr(pyredner.data_ptr(grad_img)),
                          d_scene,
                          redner.float_ptr(0))    # debug_image
        time_elapsed = time.time() - start

        if print_timing:
            print('Backward pass, time: %.5f s' % time_elapsed)

        # # For debugging
        # pyredner.imwrite(grad_img, 'grad_img.exr')
        # grad_img = tf.ones([256, 256, 3], dtype=tf.float32)
        # debug_img = tf.zeros([256, 256, 3], dtype=tf.float32)
        # redner.render(scene, options,
        #               redner.float_ptr(0),
        #               redner.float_ptr(pyredner.data_ptr(grad_img)),
        #               d_scene,
        #               redner.float_ptr(pyredner.data_ptr(debug_img)))
        # pyredner.imwrite(debug_img, 'debug.exr')
        # pyredner.imwrite(-debug_img, 'debug_.exr')
        # exit()

        ret_list = []
        ret_list.append(None) # seed
        ret_list.append(None) # num_shapes
        ret_list.append(None) # num_materials 
        ret_list.append(None) # num_lights
        if camera.use_look_at:
            ret_list.append(d_position)
            ret_list.append(d_look_at)
            ret_list.append(d_up)
            ret_list.append(None) # cam_to_world
            ret_list.append(None) # world_to_cam
        else:
            ret_list.append(None) # pos
            ret_list.append(None) # look
            ret_list.append(None) # up
            ret_list.append(d_cam_to_world)
            ret_list.append(d_world_to_cam)
        ret_list.append(d_ndc_to_cam)
        ret_list.append(d_cam_to_ndc)
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
            ret_list.append(None) # material id
            ret_list.append(None) # light id

        num_materials = len(ctx.materials)
        for i in range(num_materials):
            ret_list.append(d_diffuse_list[i])
            ret_list.append(d_diffuse_uv_scale_list[i])
            ret_list.append(d_specular_list[i])
            ret_list.append(d_specular_uv_scale_list[i])
            ret_list.append(d_roughness_list[i])
            ret_list.append(d_roughness_uv_scale_list[i])
            ret_list.append(d_normal_map_list[i])
            ret_list.append(d_normal_map_uv_scale_list[i])
            ret_list.append(None) # two sided

        num_area_lights = len(ctx.area_lights)
        for i in range(num_area_lights):
            ret_list.append(None) # shape id          
            ret_list.append(d_intensity_list[i].cpu())
            ret_list.append(None) # two sided         

        if ctx.envmap is not None:
            ret_list.append(d_envmap_values)
            ret_list.append(d_envmap_uv_scale)
            ret_list.append(None) # env_to_world
            ret_list.append(d_world_to_env.cpu())
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
        ret_list.append(None) # num channels
        for _ in range(ctx.num_channels):
            ret_list.append(None) # channel

        ret_list.append(None) # sampler type
        ret_list.append(None) # use_primary_edge_sampling
        ret_list.append(None) # use_secondary_edge_sampling

        return ret_list

    return img, backward
