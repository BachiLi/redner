import pyredner
import numpy as np
import torch

# From the test_single_triangle.py test case but with viewport

pyredner.set_use_gpu(torch.cuda.is_available())

cam = pyredner.Camera(position = torch.tensor([0.0, 0.0, -5.0]),
                      look_at = torch.tensor([0.0, 0.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([45.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (1024, 1024),
                      viewport = (200, 300, 700, 800))

mat_grey = pyredner.Material(\
    diffuse_reflectance = \
        torch.tensor([0.5, 0.5, 0.5], device = pyredner.get_device()))
materials = [mat_grey]

shape_triangle = pyredner.Shape(\
    vertices = torch.tensor([[-1.7, 1.0, 0.0], [1.0, 1.0, 0.0], [-0.5, -1.0, 0.0]],
        device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2]], dtype = torch.int32,
        device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)
shape_light = pyredner.Shape(\
    vertices = torch.tensor([[-1.0, -1.0, -7.0],
                             [ 1.0, -1.0, -7.0],
                             [-1.0,  1.0, -7.0],
                             [ 1.0,  1.0, -7.0]], device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)
shapes = [shape_triangle, shape_light]

light = pyredner.AreaLight(shape_id = 1, 
                           intensity = torch.tensor([20.0,20.0,20.0]))
area_lights = [light]
scene = pyredner.Scene(cam, shapes, materials, area_lights)
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)

render = pyredner.RenderFunction.apply
img = render(0, *scene_args)
pyredner.imwrite(img.cpu(), 'results/test_viewport/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_viewport/target.png')
target = pyredner.imread('results/test_viewport/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

# Perturb the scene, this is our initial guess.
shape_triangle.vertices = torch.tensor(\
    [[-2.0,1.5,0.3], [0.9,1.2,-0.3], [-0.4,-1.4,0.2]],
    device = pyredner.get_device(),
    requires_grad = True) # Set requires_grad to True since we want to optimize this
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
img = render(1, *scene_args)
pyredner.imwrite(img.cpu(), 'results/test_single_triangle/init.png')
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/test_single_triangle/init_diff.png')

optimizer = torch.optim.Adam([shape_triangle.vertices], lr=5e-2)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4, # We use less samples in the Adam loop.
        max_bounces = 1)
    img = render(t+1, *scene_args)
    pyredner.imwrite(img.cpu(), 'results/test_viewport/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    loss.backward()
    print('grad:', shape_triangle.vertices.grad)

    optimizer.step()
    print('vertices:', shape_triangle.vertices)

scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
img = render(202, *scene_args)
pyredner.imwrite(img.cpu(), 'results/test_viewport/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_viewport/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/test_viewport/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_viewport/iter_%d.png", "-vb", "20M",
    "results/test_viewport/out.mp4"])
