import pyredner
import torch
from matplotlib import cm
import os
import numpy as np
import skimage

import sys

def normalize(x, min_, max_):
    range = max(abs(min_), abs(max_))
    return (x + range) / (2 * range)

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

objects = pyredner.load_obj('scenes/teapot.obj', return_objects=True)
camera = pyredner.automatic_camera_placement(objects, resolution=(512, 512))
print(camera.position)
print(camera.look_at)
print(camera.up)
#sys.exit(1)
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
np.save('results/test_screen_gradient/dx.npy', x_diff.cpu().numpy())
skimage.io.imsave('results/test_screen_gradient/dx.png', (dx * 255).astype(np.uint8))
y_diff = screen_gradient_img[:, :, 1]
dy = cm.viridis(normalize(y_diff, y_diff.min() * clamp_factor, y_diff.max() * clamp_factor).cpu().numpy())
np.save('results/test_screen_gradient/dy.npy', y_diff.cpu().numpy())
skimage.io.imsave('results/test_screen_gradient/dy.png', (dy * 255).astype(np.uint8))


# Use the warp field renderer.

pyredner.set_use_gpu(False)
integrator = pyredner.integrators.WarpFieldIntegrator(
                num_samples = 16,
                max_bounces = 0,
                channels = [pyredner.channels.diffuse_reflectance],
                use_warp_fields=True,
                kernel_parameters=pyredner.integrators.KernelParameters(
                                    vMFConcentration=10000,
                                    auxPrimaryGaussianStddev=0.5,
                                )
            )
screen_gradient_img_warp = pyredner.RenderFunction.visualize_screen_gradient_class(
    grad_img = None,
    seed = 0,
    scene = scene,
    integrator = integrator
    )

#serialized = pyredner.RenderFunction.serialize_scene_class(scene, integrator=integrator)
#image = pyredner.RenderFunction.apply(0, *serialized)
#np.save('results/test_screen_gradient/image.npy', image)

directory = 'results/test_screen_gradient'
if directory != '' and not os.path.exists(directory):
    os.makedirs(directory)



clamp_factor = 0.2
x_diff = screen_gradient_img_warp[:, :, 0]
dx = cm.viridis(normalize(x_diff, x_diff.min() * clamp_factor, x_diff.max() * clamp_factor).numpy())
np.save('results/test_screen_gradient/dx_was.npy', x_diff)
skimage.io.imsave('results/test_screen_gradient/dx_was.png', (dx * 255).astype(np.uint8))
y_diff = screen_gradient_img_warp[:, :, 1]
dy = cm.viridis(normalize(y_diff, y_diff.min() * clamp_factor, y_diff.max() * clamp_factor).numpy())
np.save('results/test_screen_gradient/dy_was.npy', y_diff)
skimage.io.imsave('results/test_screen_gradient/dy_was.png', (dy * 255).astype(np.uint8))
