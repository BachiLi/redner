import pyredner
import numpy as np
import torch

# Optimize six vertices of a two triangles

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

mat_green = pyredner.Material(\
    diffuse_reflectance = torch.tensor([0.35, 0.75, 0.35],
    device = pyredner.get_device()))
mat_red = pyredner.Material(\
    diffuse_reflectance = torch.tensor([0.75, 0.35, 0.35],
    device = pyredner.get_device()))
mat_black = pyredner.Material(\
    diffuse_reflectance = torch.tensor([0.0, 0.0, 0.0],
    device = pyredner.get_device()))
materials = [mat_green,mat_red,mat_black]
tri0_vertices = torch.tensor(\
    [[-1.7,1.0,0.0], [1.0,1.0,0.0], [-0.5,-1.0,0.0]],
    device = pyredner.get_device())
tri1_vertices = torch.tensor(\
    [[-1.0,1.5,1.0], [0.2,1.5,1.0], [0.2,-1.5,1.0]],
    device = pyredner.get_device())
tri0_indices = torch.tensor([[0, 1, 2]], dtype = torch.int32, device = pyredner.get_device())
tri1_indices = torch.tensor([[0, 1, 2]], dtype = torch.int32, device = pyredner.get_device())
shape_tri0 = pyredner.Shape(tri0_vertices, tri0_indices, 0)
shape_tri1 = pyredner.Shape(tri1_vertices, tri1_indices, 1)
light_vertices = torch.tensor(\
    [[-1.0,-1.0,-7.0],[1.0,-1.0,-7.0],[-1.0,1.0,-7.0],[1.0,1.0,-7.0]],
    device = pyredner.get_device())
light_indices = torch.tensor([[0,1,2],[1,3,2]], dtype = torch.int32, device = pyredner.get_device())
shape_light = pyredner.Shape(light_vertices, light_indices, 2)
shapes = [shape_tri0, shape_tri1, shape_light]
light_intensity = torch.tensor([20.0,20.0,20.0])
# The first argument is the shape id of the light
light = pyredner.AreaLight(2, light_intensity)
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
pyredner.imwrite(img.cpu(), 'results/test_two_triangles/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_two_triangles/target.png')
target = pyredner.imread('results/test_two_triangles/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

# Perturb the scene, this is our initial guess
shape_tri0.vertices = torch.tensor(\
    [[-1.3,1.5,0.1], [1.5,0.7,-0.2], [-0.8,-1.1,0.2]],
    device = pyredner.get_device(),
    requires_grad=True)
shape_tri1.vertices = torch.tensor(\
    [[-0.5,1.2,1.2], [0.3,1.7,1.0], [0.5,-1.8,1.3]],
    device = pyredner.get_device(),
    requires_grad=True)
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    max_bounces = 1)
# Render the initial guess
img = render(1, *args)
pyredner.imwrite(img.cpu(), 'results/test_two_triangles/init.png')
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/test_two_triangles/init_diff.png')

# Optimize for triangle vertices
optimizer = torch.optim.Adam([shape_tri0.vertices, shape_tri1.vertices], lr=5e-2)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image
    args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4,
        max_bounces = 1)
    img = render(t+1, *args)
    pyredner.imwrite(img.cpu(), 'results/test_two_triangles/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    loss.backward()
    print('tri0.grad:', shape_tri0.vertices.grad)
    print('tri1.grad:', shape_tri1.vertices.grad)

    optimizer.step()
    print('tri0:', shape_tri0.vertices)
    print('tri1:', shape_tri1.vertices)

args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    max_bounces = 1)
img = render(202, *args)
pyredner.imwrite(img.cpu(), 'results/test_two_triangles/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_two_triangles/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/test_two_triangles/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_two_triangles/iter_%d.png", "-vb", "20M",
    "results/test_two_triangles/out.mp4"])
