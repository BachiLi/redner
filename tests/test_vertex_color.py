import pyredner
import redner
import numpy as np
import torch
import math

# Example of optimizing vertex color of a sphere.

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

cam = pyredner.Camera(position = torch.tensor([0.0, 0.0, -5.0]),
                      look_at = torch.tensor([0.0, 0.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([45.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256))

# Set "use_vertex_color = True" to use vertex color
mat_vertex_color = pyredner.Material(use_vertex_color = True)
materials = [mat_vertex_color]

vertices, indices, uvs, normals = pyredner.generate_sphere(128, 64)
# For the target we randomize the vertex color.
vertex_color = torch.zeros_like(vertices).uniform_(0.0, 1.0)
shape_sphere = pyredner.Shape(\
    vertices = vertices,
    indices = indices,
    uvs = uvs,
    normals = normals,
    colors = vertex_color, # use the 'colors' field in Shape to store the color
    material_id = 0)
shapes = [shape_sphere]

envmap = pyredner.imread('sunsky.exr')
if pyredner.get_use_gpu():
    envmap = envmap.cuda(device = pyredner.get_device())
envmap = pyredner.EnvironmentMap(envmap)
scene = pyredner.Scene(cam, shapes, materials, [], envmap)
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    max_bounces = 1,
    channels = [redner.channels.radiance, redner.channels.vertex_color])
render = pyredner.RenderFunction.apply
img = render(0, *scene_args)
img_radiance = img[:, :, :3]
img_vertex_color = img[:, :, 3:]
pyredner.imwrite(img_radiance.cpu(), 'results/test_vertex_color/target.exr')
pyredner.imwrite(img_radiance.cpu(), 'results/test_vertex_color/target.png')
pyredner.imwrite(img_vertex_color.cpu(), 'results/test_vertex_color/target_color.png')
target_radiance = pyredner.imread('results/test_vertex_color/target.exr')
if pyredner.get_use_gpu():
    target_radiance = target_radiance.cuda()

# Initial guess. Set to 0.5 for all vertices.
shape_sphere.colors = \
    torch.zeros_like(vertices, device = pyredner.get_device()) + 0.5
shape_sphere.colors.requires_grad = True
# We render both the radiance and the vertex color here.
# The vertex color is only for visualization.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    max_bounces = 1,
    channels = [redner.channels.radiance, redner.channels.vertex_color])
img = render(1, *scene_args)
img_radiance = img[:, :, :3]
img_vertex_color = img[:, :, 3:]
pyredner.imwrite(img_radiance.cpu(), 'results/test_vertex_color/init.png')
pyredner.imwrite(img_vertex_color.cpu(), 'results/test_vertex_color/init_color.png')
diff = torch.abs(target_radiance - img_radiance)
pyredner.imwrite(diff.cpu(), 'results/test_vertex_color/init_diff.png')

optimizer = torch.optim.Adam([shape_sphere.colors], lr=1e-2)
for t in range(100):
    print('iteration:', t)
    optimizer.zero_grad()
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4,
        max_bounces = 1,
        channels = [redner.channels.radiance, redner.channels.vertex_color])
    img = render(t+1, *scene_args)
    img_radiance = img[:, :, :3]
    img_vertex_color = img[:, :, 3:]
    pyredner.imwrite(img_radiance.cpu(), 'results/test_vertex_color/iter_{}.png'.format(t))
    pyredner.imwrite(img_vertex_color.cpu(), 'results/test_vertex_color/iter_color_{}.png'.format(t))

    loss = torch.pow(img_radiance - target_radiance, 2).sum()
    print('loss:', loss.item())

    loss.backward()
    optimizer.step()

    # Clamp the data to valid range.
    shape_sphere.colors.data.clamp_(0.0, 1.0)

scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    max_bounces = 1,
    channels = [redner.channels.radiance, redner.channels.vertex_color])
img = render(102, *scene_args)
img_radiance = img[:, :, :3]
img_vertex_color = img[:, :, 3:]
pyredner.imwrite(img_radiance.cpu(), 'results/test_vertex_color/final.exr')
pyredner.imwrite(img_radiance.cpu(), 'results/test_vertex_color/final.png')
pyredner.imwrite(img_vertex_color.cpu(), 'results/test_vertex_color/final_color.png')
pyredner.imwrite(torch.abs(target_radiance - img_radiance).cpu(), 'results/test_vertex_color/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_vertex_color/iter_%d.png", "-vb", "20M",
    "results/test_vertex_color/out.mp4"])
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_vertex_color/iter_color_%d.png", "-vb", "20M",
    "results/test_vertex_color/out_color.mp4"])