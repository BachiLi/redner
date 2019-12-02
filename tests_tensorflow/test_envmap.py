# Tensorflow by default allocates all GPU memory, leaving very little for rendering.
# We set the environment variable TF_FORCE_GPU_ALLOW_GROWTH to true to enforce on demand
# memory allocation to reduce page faults.
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pyredner_tensorflow as pyredner

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
    mat_grey = pyredner.Material(
        diffuse_reflectance = tf.Variable([0.4, 0.4, 0.4], dtype=tf.float32),
        specular_reflectance = tf.Variable([0.5, 0.5, 0.5], dtype=tf.float32),
        roughness = tf.Variable([0.05], dtype=tf.float32))

materials = [mat_grey]

with tf.device(pyredner.get_device_name()):
    vertices, indices, uvs, normals = pyredner.generate_sphere(128, 64)
    shape_sphere = pyredner.Shape(
        vertices = vertices,
        indices = indices,
        uvs = uvs,
        normals = normals,
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
    max_bounces = 1)

img = pyredner.render(0, *scene_args)
pyredner.imwrite(img, 'results/test_envmap/target.exr')
pyredner.imwrite(img, 'results/test_envmap/target.png')
target = pyredner.imread('results/test_envmap/target.exr')

with tf.device(pyredner.get_device_name()):
    envmap_texels = tf.Variable(0.5 * tf.ones([32, 64, 3], dtype=tf.float32),
                                trainable=True)
    envmap = pyredner.EnvironmentMap(tf.abs(envmap_texels))
scene = pyredner.Scene(camera=cam,
                       shapes=shapes,
                       materials=materials,
                       area_lights=[],
                       envmap=envmap)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 256,
    max_bounces = 1)
img = pyredner.render(1, *scene_args)
pyredner.imwrite(img, 'results/test_envmap/init.png')
diff = tf.abs(target - img)
pyredner.imwrite(diff, 'results/test_envmap/init_diff.png')

optimizer = tf.compat.v1.train.AdamOptimizer(1e-2)
for t in range(600):
    print('iteration:', t)
    with tf.GradientTape() as tape:
        envmap = pyredner.EnvironmentMap(tf.abs(envmap_texels))
        scene = pyredner.Scene(camera=cam,
                               shapes=shapes,
                               materials=materials,
                               area_lights=[],
                               envmap=envmap)
        scene_args = pyredner.serialize_scene(
            scene = scene,
            num_samples = 4,
            max_bounces = 1)
        img = pyredner.render(t+1, *scene_args)

        loss = tf.reduce_sum(tf.square(img - target))
        print('loss:', loss)

    pyredner.imwrite(img, 'results/test_envmap/iter_{}.png'.format(t))
    pyredner.imwrite(tf.abs(envmap_texels), 'results/test_envmap/envmap_{}.exr'.format(t))

    grads = tape.gradient(loss, [envmap_texels])
    optimizer.apply_gradients(zip(grads, [envmap_texels]))

scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 256,
    max_bounces = 1)
img = pyredner.render(602, *scene_args)
pyredner.imwrite(img, 'results/test_envmap/final.exr')
pyredner.imwrite(img, 'results/test_envmap/final.png')
pyredner.imwrite(tf.abs(target - img), 'results/test_envmap/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_envmap/iter_%d.png", "-vb", "20M",
    "results/test_envmap/out.mp4"])
