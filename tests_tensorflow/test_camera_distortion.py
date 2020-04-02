# Tensorflow by default allocates all GPU memory, leaving very little for rendering.
# We set the environment variable TF_FORCE_GPU_ALLOW_GROWTH to true to enforce on demand
# memory allocation to reduce page faults.
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pyredner_tensorflow as pyredner

pyredner.set_use_gpu(tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None))

position = tf.Variable([1.0, 0.0, -3.0], dtype=tf.float32)
look_at = tf.Variable([1.0, 0.0, 0.0], dtype=tf.float32)
up = tf.Variable([0.0, 1.0, 0.0], dtype=tf.float32)
fov = tf.Variable([45.0], dtype=tf.float32)
clip_near = 1e-2

# randomly generate distortion parameters
tf.random.set_seed(1234)
target_distort_params = (tf.random.uniform([8]) - 0.5) * 0.05
resolution = (256, 256)
cam = pyredner.Camera(position = position,
                      look_at = look_at,
                      up = up,
                      fov = fov,
                      clip_near = clip_near,
                      resolution = resolution,
                      distortion_params = target_distort_params)

checkerboard_texture = pyredner.imread('scenes/teapot.png')
mat_checkerboard = pyredner.Material(\
    diffuse_reflectance = checkerboard_texture)
mat_black = pyredner.Material(\
    diffuse_reflectance = tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32))

plane = pyredner.Object(vertices = tf.Variable([[-1.0,-1.0, 0.0],
                                                [-1.0, 1.0, 0.0],
                                                [ 1.0,-1.0, 0.0],
                                                [ 1.0, 1.0, 0.0]]),
                        indices = tf.constant([[0, 1, 2],
                                               [1, 3, 2]],
                                              dtype = tf.int32),
                        uvs = tf.Variable([[0.05, 0.05],
                                           [0.05, 0.95],
                                           [0.95, 0.05],
                                           [0.95, 0.95]]),
                        material = mat_checkerboard)
scene = pyredner.Scene(camera=cam, objects=[plane])
img = pyredner.render_albedo(scene=scene)
pyredner.imwrite(img, 'results/test_camera_distortion/target.exr')
pyredner.imwrite(img, 'results/test_camera_distortion/target.png')
# Read the target image we just saved.
target = pyredner.imread('results/test_camera_distortion/target.exr')

cam.distortion_params = tf.Variable(tf.zeros(8), trainable=True)
scene = pyredner.Scene(camera=cam, objects=[plane])
img = pyredner.render_albedo(scene=scene)
pyredner.imwrite(img, 'results/test_camera_distortion/init.exr')
pyredner.imwrite(img, 'results/test_camera_distortion/init.png')

# Optimize for triangle vertices.
optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
for t in range(200):
    print('iteration:', t)

    with tf.GradientTape() as tape:
        scene = pyredner.Scene(camera=cam, objects=[plane])
        img = pyredner.render_albedo(scene=scene)
        pyredner.imwrite(img, 'results/test_camera_distortion/iter_{}.png'.format(t))
        loss = tf.reduce_sum(tf.square(img - target))

    print('loss:', loss)

    grads = tape.gradient(loss, [cam.distortion_params])
    optimizer.apply_gradients(zip(grads, [cam.distortion_params]))

    print('grad:', grads)

    print('distortion_params:', cam.distortion_params)

img = pyredner.render_albedo(scene=scene)
pyredner.imwrite(img, 'results/test_camera_distortion/final.exr')
pyredner.imwrite(img, 'results/test_camera_distortion/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_camera_distortion/iter_%d.png", "-vb", "20M",
    "results/test_camera_distortion/out.mp4"])
