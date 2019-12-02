# Tensorflow by default allocates all GPU memory, leaving very little for rendering.
# We set the environment variable TF_FORCE_GPU_ALLOW_GROWTH to true to enforce on demand
# memory allocation to reduce page faults.
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pyredner_tensorflow as pyredner
import redner

# Use GPU if available
pyredner.set_use_gpu(tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None))

with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
    cam = pyredner.Camera(position = tf.Variable([0.0, 0.0, -5.0], dtype=tf.float32),
                          look_at = tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32),
                          up = tf.Variable([0.0, 1.0, 0.0], dtype=tf.float32),
                          fov = tf.Variable([45.0], dtype=tf.float32), # in degree
                          clip_near = 1e-2, # needs to > 0
                          resolution = (256, 256),
                          fisheye = False)

with tf.device(pyredner.get_device_name()):
    mat_vertex_color = pyredner.Material(use_vertex_color = True)
materials = [mat_vertex_color]

with tf.device(pyredner.get_device_name()):
    # For the target we randomize the vertex color.
    vertices, indices, uvs, normals = pyredner.generate_sphere(128, 64)
    vertex_color = tf.random.uniform(vertices.shape, 0.0, 1.0)
    shape_sphere = pyredner.Shape(
        vertices = vertices,
        indices = indices,
        uvs = uvs,
        normals = normals,
        colors = vertex_color,
        material_id = 0)
shapes = [shape_sphere]

with tf.device(pyredner.get_device_name()):
    envmap = pyredner.imread('sunsky.exr')
    envmap = pyredner.EnvironmentMap(envmap)
scene = pyredner.Scene(camera=cam,
                       shapes=shapes,
                       materials=materials,
                       area_lights=[],
                       envmap=envmap)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 256,
    max_bounces = 1,
    channels = [redner.channels.radiance, redner.channels.vertex_color])

img = pyredner.render(0, *scene_args)
img_radiance = img[:, :, :3]
img_vertex_color = img[:, :, 3:]
pyredner.imwrite(img_radiance, 'results/test_vertex_color/target.exr')
pyredner.imwrite(img_radiance, 'results/test_vertex_color/target.png')
pyredner.imwrite(img_vertex_color, 'results/test_vertex_color/target_color.png')
target_radiance = pyredner.imread('results/test_vertex_color/target.exr')

# Initial guess. Set to 0.5 for all vertices.
with tf.device(pyredner.get_device_name()):
    shape_sphere.colors = tf.Variable(tf.zeros_like(vertices) + 0.5)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 256,
    max_bounces = 1,
    channels = [redner.channels.radiance, redner.channels.vertex_color])
img = pyredner.render(1, *scene_args)
img_radiance = img[:, :, :3]
img_vertex_color = img[:, :, 3:]
pyredner.imwrite(img_radiance, 'results/test_vertex_color/init.png')
pyredner.imwrite(img_vertex_color, 'results/test_vertex_color/init_color.png')
diff = tf.abs(target_radiance - img_radiance)
pyredner.imwrite(diff, 'results/test_vertex_color/init_diff.png')

optimizer = tf.compat.v1.train.AdamOptimizer(1e-2)
for t in range(100):
    print('iteration:', t)
    with tf.GradientTape() as tape:
        scene_args = pyredner.serialize_scene(
            scene = scene,
            num_samples = 4,
            max_bounces = 1,
            channels = [redner.channels.radiance, redner.channels.vertex_color])
        img = pyredner.render(t+1, *scene_args)
        img_radiance = img[:, :, :3]
        img_vertex_color = img[:, :, 3:]

        loss = tf.reduce_sum(tf.square(img_radiance - target_radiance))
        print('loss:', loss)

    pyredner.imwrite(img_radiance, 'results/test_vertex_color/iter_{}.png'.format(t))
    pyredner.imwrite(img_vertex_color, 'results/test_vertex_color/iter_color_{}.png'.format(t))

    grads = tape.gradient(loss, [shape_sphere.colors])
    optimizer.apply_gradients(zip(grads, [shape_sphere.colors]))

scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 256,
    max_bounces = 1,
    channels = [redner.channels.radiance, redner.channels.vertex_color])
img = pyredner.render(102, *scene_args)
img_radiance = img[:, :, :3]
img_vertex_color = img[:, :, 3:]
pyredner.imwrite(img_radiance, 'results/test_vertex_color/final.exr')
pyredner.imwrite(img_radiance, 'results/test_vertex_color/final.png')
pyredner.imwrite(img_vertex_color, 'results/test_vertex_color/final_color.png')
pyredner.imwrite(tf.abs(target_radiance - img_radiance), 'results/test_vertex_color/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_vertex_color/iter_%d.png", "-vb", "20M",
    "results/test_vertex_color/out.mp4"])
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_vertex_color/iter_color_%d.png", "-vb", "20M",
    "results/test_vertex_color/out_color.mp4"])