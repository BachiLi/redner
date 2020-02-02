# Tensorflow by default allocates all GPU memory, leaving very little for rendering.
# We set the environment variable TF_FORCE_GPU_ALLOW_GROWTH to true to enforce on demand
# memory allocation to reduce page faults.
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pyredner_tensorflow as pyredner
from matplotlib import cm
import os
import numpy as np
import skimage

def normalize(x, min_, max_):
    range = max(abs(min_), abs(max_))
    return (x + range) / (2 * range)

# Use GPU if available
pyredner.set_use_gpu(tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None))

objects = pyredner.load_obj('scenes/teapot.obj', return_objects=True)
camera = pyredner.automatic_camera_placement(objects, resolution=(512, 512))
scene = pyredner.Scene(camera = camera, objects = objects)
screen_gradient_img = pyredner.visualize_screen_gradient(
	grad_img = None,
    seed = 0,
    scene = scene,
    num_samples = 4,
    max_bounces = 0,
    channels = [pyredner.channels.diffuse_reflectance])

directory = 'results/test_screen_gradient'
if directory != '' and not os.path.exists(directory):
    os.makedirs(directory)

clamp_factor = 0.2
x_diff = screen_gradient_img[:, :, 0].numpy()
dx = cm.viridis(normalize(x_diff, np.min(x_diff) * clamp_factor, np.max(x_diff) * clamp_factor))
skimage.io.imsave('results/test_screen_gradient/dx.png', (dx * 255).astype(np.uint8))
y_diff = screen_gradient_img[:, :, 1]
dy = cm.viridis(normalize(y_diff, np.min(y_diff) * clamp_factor, np.max(y_diff) * clamp_factor))
skimage.io.imsave('results/test_screen_gradient/dy.png', (dy * 255).astype(np.uint8))
