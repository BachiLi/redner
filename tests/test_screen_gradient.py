import pyredner
import torch
from matplotlib import cm
import os
import numpy as np
import skimage

def normalize(x, min_, max_):
    range = max(abs(min_), abs(max_))
    return (x + range) / (2 * range)

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

objects = pyredner.load_obj('scenes/teapot.obj', return_objects=True)
camera = pyredner.automatic_camera_placement(objects, resolution=(512, 512))
scene = pyredner.Scene(camera = camera, objects = objects)
screen_gradient_img = pyredner.RenderFunction.visualize_screen_gradient(
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
x_diff = screen_gradient_img[:, :, 0]
dx = cm.viridis(normalize(x_diff, x_diff.min() * clamp_factor, x_diff.max() * clamp_factor).cpu().numpy())
skimage.io.imsave('results/test_screen_gradient/dx.png', (dx * 255).astype(np.uint8))
y_diff = screen_gradient_img[:, :, 1]
dy = cm.viridis(normalize(y_diff, y_diff.min() * clamp_factor, y_diff.max() * clamp_factor).cpu().numpy())
skimage.io.imsave('results/test_screen_gradient/dy.png', (dy * 255).astype(np.uint8))
