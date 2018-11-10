import pyredner
import numpy as np
import torch

# Optimize for material parameters and camera pose

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

# Load the scene from a Mitsuba scene file
scene = pyredner.load_mitsuba('scenes/teapot.xml')

# The last material is the teapot material, set it to the target
scene.materials[-1].diffuse_reflectance = \
  torch.tensor([0.3, 0.2, 0.2], device = pyredner.get_device())
scene.materials[-1].specular_reflectance = \
  torch.tensor([0.6, 0.6, 0.6], device = pyredner.get_device())
scene.materials[-1].roughness = torch.tensor([0.05], device = pyredner.get_device())
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 1024,
    max_bounces = 2)

# Alias of the render function
render = pyredner.RenderFunction.apply
# Render our target. The first argument is the seed for RNG in the renderer.
img = render(0, *args)
pyredner.imwrite(img.cpu(), 'results/test_teapot_reflectance/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_teapot_reflectance/target.png')
target = pyredner.imread('results/test_teapot_reflectance/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda()

# Perturb the scene, this is our initial guess
cam = scene.camera
cam_position = cam.position
cam_translation = torch.tensor([-0.1, 0.1, -0.1], requires_grad = True)
scene.materials[-1].diffuse_reflectance = \
    torch.tensor([0.3, 0.3, 0.3], device = pyredner.get_device(), requires_grad = True)
scene.materials[-1].specular_reflectance = \
    torch.tensor([0.5, 0.5, 0.5], device = pyredner.get_device(), requires_grad = True)
scene.materials[-1].roughness = \
    torch.tensor([0.2], device = pyredner.get_device(), requires_grad = True)
scene.camera = pyredner.Camera(position     = cam_position + cam_translation,
                               look_at      = cam.look_at,
                               up           = cam.up,
                               fov          = cam.fov,
                               clip_near    = cam.clip_near,
                               resolution   = cam.resolution,
                               fisheye      = False)
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 1024,
    max_bounces = 2)
# Render the initial guess
img = render(1, *args)
pyredner.imwrite(img.cpu(), 'results/test_teapot_reflectance/init.png')
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/test_teapot_reflectance/init_diff.png')

lr = 1e-2
optimizer = torch.optim.Adam([scene.materials[-1].diffuse_reflectance,
                              scene.materials[-1].specular_reflectance,
                              scene.materials[-1].roughness,
                              cam_translation], lr=lr)
num_iteration = 400
for t in range(num_iteration):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image
    # need to rerun Camera constructor for autodiff 
    scene.camera = pyredner.Camera(position     = cam_position + cam_translation,
                                   look_at      = cam.look_at,
                                   up           = cam.up,
                                   fov          = cam.fov,
                                   clip_near    = cam.clip_near,
                                   resolution   = cam.resolution,
                                   fisheye      = False)
    args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4,
        max_bounces = 2)
    img = render(t+1, *args)
    pyredner.imwrite(img.cpu(), 'results/test_teapot_reflectance/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    loss.backward()
    print('diffuse_reflectance.grad:', scene.materials[-1].diffuse_reflectance.grad)
    print('specular_reflectance.grad:', scene.materials[-1].specular_reflectance.grad)
    print('roughness.grad:', scene.materials[-1].roughness.grad)
    print('cam_translation.grad:', cam_translation.grad)

    optimizer.step()
    print('diffuse_reflectance:', scene.materials[-1].diffuse_reflectance)
    print('specular_reflectance:', scene.materials[-1].specular_reflectance)
    print('roughness:', scene.materials[-1].roughness)
    print('cam_translation:', cam_translation)

    # Linearly reduce the learning rate
    lr = 1e-2 * float(num_iteration - t) / float(num_iteration)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 1024,
    max_bounces = 2)
img = render(num_iteration + 2, *args)
pyredner.imwrite(img.cpu(), 'results/test_teapot_reflectance/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_teapot_reflectance/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/test_teapot_reflectance/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_teapot_reflectance/iter_%d.png", "-vb", "20M",
    "results/test_teapot_reflectance/out.mp4"])
