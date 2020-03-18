import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pyredner_tensorflow as pyredner

# From the test_single_triangle.py test case but with viewport

# Use GPU if available
pyredner.set_use_gpu(tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None))

with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
    cam = pyredner.Camera(position = tf.Variable([0.0, 0.0, -5.0], dtype=tf.float32),
                          look_at = tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32),
                          up = tf.Variable([0.0, 1.0, 0.0], dtype=tf.float32),
                          fov = tf.Variable([45.0], dtype=tf.float32), # in degree
                          clip_near = 1e-2, # needs to > 0
                          resolution = (1024, 1024),
                          viewport = (200, 300, 700, 800))

with tf.device(pyredner.get_device_name()):
    mat_grey = pyredner.Material(
        diffuse_reflectance = tf.Variable([0.5, 0.5, 0.5], dtype=tf.float32))
materials = [mat_grey]

with tf.device(pyredner.get_device_name()):
    shape_triangle = pyredner.Shape(
        vertices = tf.Variable([[-1.7, 1.0, 0.0], [1.0, 1.0, 0.0], [-0.5, -1.0, 0.0]],
            dtype=tf.float32),
        indices = tf.constant([[0, 1, 2]], dtype=tf.int32),
        uvs = None,
        normals = None,
        material_id = 0)
    shape_light = pyredner.Shape(
        vertices = tf.Variable([[-1.0, -1.0, -7.0],
                                [ 1.0, -1.0, -7.0],
                                [-1.0,  1.0, -7.0],
                                [ 1.0,  1.0, -7.0]], dtype=tf.float32),
        indices = tf.constant([[0, 1, 2],[1, 3, 2]], dtype=tf.int32),
        uvs = None,
        normals = None,
        material_id = 0)
shapes = [shape_triangle, shape_light]

with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
    light = pyredner.AreaLight(shape_id = 1, 
                               intensity = tf.Variable([20.0,20.0,20.0], dtype=tf.float32))
area_lights = [light]

scene = pyredner.Scene(cam, shapes, materials, area_lights)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)

img = pyredner.render(0, *scene_args)
pyredner.imwrite(img, 'results/test_single_triangle/target.exr')
pyredner.imwrite(img, 'results/test_single_triangle/target.png')
target = pyredner.imread('results/test_single_triangle/target.exr')
if pyredner.get_use_gpu():
    target = target.gpu()

with tf.device(pyredner.get_device_name()):
    shape_triangle.vertices = tf.Variable(
        [[-2.0,1.5,0.3], [0.9,1.2,-0.3], [-0.4,-1.4,0.2]],
        dtype=tf.float32,
        trainable=True) # Set trainable to True since we want to optimize this
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
img = pyredner.render(1, *scene_args)
pyredner.imwrite(img, 'results/test_single_triangle/init.png')
diff = tf.abs(target - img)
pyredner.imwrite(diff, 'results/test_single_triangle/init_diff.png')

optimizer = tf.compat.v1.train.AdamOptimizer(5e-2)

def loss(output, target):
    error = output - target
    return tf.reduce_sum(tf.square(error))

def optimize(scene_args, grads, lr=5e-2):
    updates = []
    for var, grad in zip(scene_args, grads):
        if grad is None: 
            updates.append(var)
            continue
        updates.append(var - lr * grad)

    return updates

# Run 200 Adam iterations.
for t in range(1, 201):
    print('iteration:', t)

    with tf.GradientTape() as tape:
        # Forward pass: render the image.
        
        # Important to use a different seed every iteration, otherwise the result
        # would be biased.
        scene_args = pyredner.serialize_scene(
            scene = scene,
            num_samples = 4, # We use less samples in the Adam loop.
            max_bounces = 1)
        img = pyredner.render(t, *scene_args)
        loss_value = loss(img, target)

    print(f"loss_value: {loss_value}")
    pyredner.imwrite(img, 'results/test_single_triangle/iter_{}.png'.format(t))

    grads = tape.gradient(loss_value, [shape_triangle.vertices])
    optimizer.apply_gradients(zip(grads, [shape_triangle.vertices]))

    print('grad:', grads[0])
    print('vertices:', shape_triangle.vertices)

scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
img = pyredner.render(202, *scene_args)
pyredner.imwrite(img, 'results/test_single_triangle/final.exr')
pyredner.imwrite(img, 'results/test_single_triangle/final.png')
pyredner.imwrite(tf.abs(target - img), 'results/test_single_triangle/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_single_triangle/iter_%d.png", "-vb", "20M",
    "results/test_single_triangle/out.mp4"])
