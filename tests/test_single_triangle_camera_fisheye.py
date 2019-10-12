import pyredner
import numpy as np
import torch

# Optimize fisheye camera parameters of a single triangle rendering

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

# Set up the scene using Pytorch tensor
position = torch.tensor([0.0, 0.0, -1.0])
look_at = torch.tensor([0.0, 0.0, 0.0])
up = torch.tensor([0.0, 1.0, 0.0])
fov = torch.tensor([45.0])
clip_near = 1e-2

resolution = (256, 256)
cam = pyredner.Camera(position = position,
                      look_at = look_at,
                      up = up,
                      fov = fov,
                      clip_near = clip_near,
                      resolution = resolution,
                      fisheye = True)

mat_grey = pyredner.Material(\
    diffuse_reflectance = torch.tensor([0.5, 0.5, 0.5],
                                       device = pyredner.get_device()))
materials = [mat_grey]
vertices = torch.tensor([[-1.7,1.0,0.0], [1.0,1.0,0.0], [-0.5,-1.0,0.0]], device = pyredner.get_device())
indices = torch.tensor([[0, 1, 2]], dtype = torch.int32, device = pyredner.get_device())
shape_triangle = pyredner.Shape(vertices, indices, 0)
light_vertices = torch.tensor([[-1.0,-1.0,-9.0],[1.0,-1.0,-9.0],[-1.0,1.0,-9.0],[1.0,1.0,-9.0]],
                              device = pyredner.get_device())
light_indices = torch.tensor([[0,1,2],[1,3,2]], dtype = torch.int32, device = pyredner.get_device())
shape_light = pyredner.Shape(light_vertices, light_indices, 0)
shapes = [shape_triangle, shape_light]
light_intensity = torch.tensor([30.0,30.0,30.0])
light = pyredner.AreaLight(1, light_intensity)
area_lights = [light]
scene = pyredner.Scene(cam, shapes, materials, area_lights)
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)

# Alias of the render function
render = pyredner.RenderFunction.apply
# Render our target
img = render(0, *args)
pyredner.imwrite(img.cpu(), 'results/test_single_triangle_camera_fisheye/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_single_triangle_camera_fisheye/target.png')
target = pyredner.imread('results/test_single_triangle_camera_fisheye/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

# Perturb the scene, this is our initial guess
position = torch.tensor([0.5, -0.5, -3.0], requires_grad = True)
scene.camera = pyredner.Camera(position = position,
                               look_at = look_at,
                               up = up,
                               fov = fov,
                               clip_near = clip_near,
                               resolution = resolution,
                               fisheye = True)
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
# Render the initial guess
img = render(1, *args)
pyredner.imwrite(img.cpu(), 'results/test_single_triangle_camera_fisheye/init.png')
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/test_single_triangle_camera_fisheye/init_diff.png')

# Optimize for camera pose
optimizer = torch.optim.Adam([position], lr=2e-2)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Need to rerun the Camera constructor for PyTorch autodiff to compute the derivatives
    scene.camera = pyredner.Camera(position   = position,
                                   look_at    = look_at,
                                   up         = up,
                                   fov        = fov,
                                   clip_near  = clip_near,
                                   resolution = resolution,
                                   fisheye    = True)
    args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4,
        max_bounces = 1)
    img = render(t+1, *args)
    pyredner.imwrite(img.cpu(), 'results/test_single_triangle_camera_fisheye/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    loss.backward()
    print('position.grad:', position.grad)

    optimizer.step()
    print('position:', position)

args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
img = render(202, *args)
pyredner.imwrite(img.cpu(), 'results/test_single_triangle_camera_fisheye/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_single_triangle_camera_fisheye/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/test_single_triangle_camera_fisheye/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_single_triangle_camera_fisheye/iter_%d.png", "-vb", "20M",
    "results/test_single_triangle_camera_fisheye/out.mp4"])
