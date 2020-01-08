# Tensorflow by default allocates all GPU memory, leaving very little for rendering.
# We set the environment variable TF_FORCE_GPU_ALLOW_GROWTH to true to enforce on demand
# memory allocation to reduce page faults.
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pyredner_tensorflow as pyredner
import redner

# Automatic UV mapping example adapted from tutorial 2

# Use GPU if available
pyredner.set_use_gpu(tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None))

# Set up the pyredner scene for rendering:
with tf.device(pyredner.get_device_name()):
    material_map, mesh_list, light_map = pyredner.load_obj('scenes/teapot.obj')
    for _, mesh in mesh_list:
        mesh.normals = pyredner.compute_vertex_normal(mesh.vertices, mesh.indices)
        mesh.uvs, mesh.uv_indices = pyredner.compute_uvs(mesh.vertices, mesh.indices)

# Setup camera
with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
    cam = pyredner.Camera(position = tf.Variable([0.0, 30.0, 200.0], dtype=tf.float32),
                          look_at = tf.Variable([0.0, 30.0, 0.0], dtype=tf.float32),
                          up = tf.Variable([0.0, 1.0, 0.0], dtype=tf.float32),
                          fov = tf.Variable([45.0], dtype=tf.float32), # in degree
                          clip_near = 1e-2, # needs to > 0
                          resolution = (256, 256),
                          fisheye = False)

# Setup materials
material_id_map = {}
materials = []
count = 0
for key, value in material_map.items():
    material_id_map[key] = count
    count += 1
    materials.append(value)

# Setup geometries
shapes = []
with tf.device(pyredner.get_device_name()):
    for mtl_name, mesh in mesh_list:
        shapes.append(pyredner.Shape(
            vertices = mesh.vertices,
            indices = mesh.indices,
            uvs = mesh.uvs,
            normals = mesh.normals,
            material_id = material_id_map[mtl_name]))

with tf.device(pyredner.get_device_name()):
    envmap = pyredner.imread('sunsky.exr')
    envmap = pyredner.EnvironmentMap(envmap)

# Construct the scene
scene = pyredner.Scene(cam, shapes, materials, area_lights = [], envmap = envmap)
# Serialize the scene
# Here we specify the output channels as "depth", "shading_normal"
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 512,
    max_bounces = 1)
img = pyredner.render(0, *scene_args)
pyredner.imwrite(img, 'results/test_compute_uvs/target.exr')
pyredner.imwrite(img, 'results/test_compute_uvs/target.png')
target = pyredner.imread('results/test_compute_uvs/target.exr')
