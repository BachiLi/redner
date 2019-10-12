import pyredner
import numpy as np
import torch

# Optimize three vertices of a single triangle.
# One of the vertices is behind the camera

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

# Set up the scene using Pytorch tensor
position = torch.tensor([0.0, 0.0, -5.0])
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
                      resolution = resolution)

mat_grey = pyredner.Material(\
    diffuse_reflectance = torch.tensor([0.5, 0.5, 0.5], device = pyredner.get_device()))
materials = [mat_grey]
vertices = torch.tensor([[-1.3,1.0,0.0], [1.0,1.0,0.0], [-0.5,-2.0,-7.0]], device = pyredner.get_device())
indices = torch.tensor([[0, 1, 2]], dtype = torch.int32, device = pyredner.get_device())
shape_triangle = pyredner.Shape(vertices, indices, 0)
light_vertices = torch.tensor([[-1.0,-1.0,-7.0],[1.0,-1.0,-7.0],[-1.0,1.0,-7.0],[1.0,1.0,-7.0]],
                              device = pyredner.get_device())
light_indices = torch.tensor([[0,1,2],[1,3,2]], dtype = torch.int32, device = pyredner.get_device())
shape_light = pyredner.Shape(light_vertices, light_indices, 0)
shapes = [shape_triangle, shape_light]
light_intensity = torch.tensor([20.0,20.0,20.0])
light = pyredner.AreaLight(1, light_intensity)
area_lights = [light]
scene = pyredner.Scene(cam, shapes, materials, area_lights)
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    max_bounces = 1)

# Alias of the render function
render = pyredner.RenderFunction.apply
# Render our target
img = render(0, *args)
pyredner.imwrite(img.cpu(), 'results/test_single_triangle_clipped/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_single_triangle_clipped/target.png')
target = pyredner.imread('results/test_single_triangle_clipped/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

# Perturb the scene, this is our initial guess
shape_triangle.vertices = torch.tensor(\
    [[-1.0,1.5,0.3], [0.9,1.2,-0.3], [0.0,-3.0,-6.5]],
    device = pyredner.get_device(),
    requires_grad=True)
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    max_bounces = 1)
# Render the initial guess
img = render(1, *args)
pyredner.imwrite(img.cpu(), 'results/test_single_triangle_clipped/init.png')
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/test_single_triangle_clipped/init_diff.png')

# Optimize for triangle vertices
optimizer = torch.optim.Adam([shape_triangle.vertices], lr=2e-2)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image
    args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4,
        max_bounces = 1)
    img = render(t+1, *args)
    pyredner.imwrite(img.cpu(), 'results/test_single_triangle_clipped/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    loss.backward()
    print('grad:', shape_triangle.vertices.grad)

    optimizer.step()
    print('vertices:', shape_triangle.vertices)

args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    max_bounces = 1)
img = render(202, *args)
pyredner.imwrite(img.cpu(), 'results/test_single_triangle_clipped/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_single_triangle_clipped/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/test_single_triangle_clipped/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_single_triangle_clipped/iter_%d.png", "-vb", "20M",
    "results/test_single_triangle_clipped/out.mp4"])
