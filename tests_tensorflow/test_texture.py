# Tensorflow by default allocates all GPU memory, leaving very little for rendering.
# We set the environment variable TF_FORCE_GPU_ALLOW_GROWTH to true to enforce on demand
# memory allocation to reduce page faults.
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pyredner_tensorflow as pyredner

# Optimize four vertices of a textured patch

# Use GPU if available
pyredner.set_use_gpu(tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None))

# Set up the scene using Pytorch tensor
with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
    position = tf.Variable([0.0, 0.0, -5.0], dtype=tf.float32)
    look_at = tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32)
    up = tf.Variable([0.0, 1.0, 0.0], dtype=tf.float32)
    fov = tf.Variable([45.0], dtype=tf.float32)
    clip_near = 1e-2
    resolution = (256, 256)
    cam = pyredner.Camera(position = position,
                          look_at = look_at,
                          up = up,
                          fov = fov,
                          clip_near = clip_near,
                          resolution = resolution)

with tf.device(pyredner.get_device_name()):
    checkerboard_texture = pyredner.imread('checkerboard.exr')
    mat_checkerboard = pyredner.Material(
        diffuse_reflectance = checkerboard_texture)
    mat_black = pyredner.Material(
        diffuse_reflectance = tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32))
    materials = [mat_checkerboard, mat_black]
    vertices = tf.Variable([[-1.0,-1.0,0.0], [-1.0,1.0,0.0], [1.0,-1.0,0.0], [1.0,1.0,0.0]],
        dtype=tf.float32)
    indices = tf.constant([[0, 1, 2], [1, 3, 2]], dtype=tf.int32)
    uvs = tf.Variable([[0.05, 0.05], [0.05, 0.95], [0.95, 0.05], [0.95, 0.95]],
	dtype=tf.float32)
    shape_plane = pyredner.Shape(vertices, indices, 0, uvs)
    light_vertices = tf.Variable([[-1.0,-1.0,-7.0],[1.0,-1.0,-7.0],[-1.0,1.0,-7.0],[1.0,1.0,-7.0]],
        dtype=tf.float32)
    light_indices = tf.constant([[0,1,2],[1,3,2]], dtype=tf.int32)
    shape_light = pyredner.Shape(light_vertices, light_indices, 1)
    shapes = [shape_plane, shape_light]

with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
    light_intensity = tf.Variable([20.0, 20.0, 20.0], dtype=tf.float32)
# The first argument is the shape id of the light
light = pyredner.AreaLight(1, light_intensity)
area_lights = [light]
scene = pyredner.Scene(cam, shapes, materials, area_lights)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)

# Alias of the render function

# Render our target
img = pyredner.render(0, *scene_args)
pyredner.imwrite(img, 'results/test_texture/target.exr')
pyredner.imwrite(img, 'results/test_texture/target.png')
target = pyredner.imread('results/test_texture/target.exr')

# Perturb the scene, this is our initial guess
with tf.device(pyredner.get_device_name()):
    shape_plane.vertices = tf.Variable(
        [[-1.1,-1.2,0.0], [-1.3,1.1,0.0], [1.1,-1.1,0.0], [0.8,1.2,0.0]],
        dtype=tf.float32,
        trainable=True)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
# Render the initial guess
img = pyredner.render(1, *scene_args)
pyredner.imwrite(img, 'results/test_texture/init.png')
diff = tf.abs(target - img)
pyredner.imwrite(diff, 'results/test_texture/init_diff.png')

# Optimize for triangle vertices
optimizer = tf.compat.v1.train.AdamOptimizer(5e-2)
for t in range(200):
    print('iteration:', t)

    with tf.GradientTape() as tape:
        # Forward pass: render the image
        scene_args = pyredner.serialize_scene(
            scene = scene,
            num_samples = 4,
            max_bounces = 1)
        img = pyredner.render(t+1, *scene_args)
        pyredner.imwrite(img, 'results/test_texture/iter_{}.png'.format(t))
        loss = tf.reduce_sum(tf.square(img - target))
    
    print('loss:', loss)

    grads = tape.gradient(loss, [shape_plane.vertices])
    optimizer.apply_gradients(zip(grads, [shape_plane.vertices]))

    print('grad:', grads)
    print('vertices:', shape_plane.vertices)

scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
img = pyredner.render(202, *scene_args)
pyredner.imwrite(img, 'results/test_texture/final.exr')
pyredner.imwrite(img, 'results/test_texture/final.png')
pyredner.imwrite(tf.abs(target - img), 'results/test_texture/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_texture/iter_%d.png", "-vb", "20M",
    "results/test_texture/out.mp4"])
