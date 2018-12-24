import pyredner
import numpy as np
import torch
import math

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

def generate_sphere(theta_steps, phi_steps):
    d_theta = math.pi / (theta_steps - 1)
    d_phi = (2 * math.pi) / (phi_steps - 1)

    vertices = torch.zeros(theta_steps * phi_steps, 3,
                           device = pyredner.get_device())
    uvs = torch.zeros(theta_steps * phi_steps, 2,
                      device = pyredner.get_device())
    vertices_index = 0
    for theta_index in range(theta_steps):
        sin_theta = math.sin(theta_index * d_theta)
        cos_theta = math.cos(theta_index * d_theta)
        for phi_index in range(phi_steps):
            sin_phi = math.sin(phi_index * d_phi)
            cos_phi = math.cos(phi_index * d_phi)
            vertices[vertices_index, :] = \
                torch.tensor([sin_theta * cos_phi, cos_theta, sin_theta * sin_phi],
                    device = pyredner.get_device())
            uvs[vertices_index, 0] = theta_index * d_theta / math.pi
            uvs[vertices_index, 1] = phi_index * d_phi / (2 * math.pi)
            vertices_index += 1

    indices = []
    for theta_index in range(1, theta_steps):
        for phi_index in range(phi_steps - 1):
            id0 = phi_steps * theta_index + phi_index
            id1 = phi_steps * theta_index + phi_index + 1
            id2 = phi_steps * (theta_index - 1) + phi_index
            id3 = phi_steps * (theta_index - 1) + phi_index + 1

            if (theta_index < theta_steps - 1):
                indices.append([id0, id2, id1])
            if (theta_index > 1):
                indices.append([id1, id2, id3])
    indices = torch.tensor(indices,
                           dtype = torch.int32,
                           device = pyredner.get_device())

    normals = vertices.clone()
    return (vertices, indices, uvs, normals)

cam = pyredner.Camera(position = torch.tensor([0.0, 0.0, -5.0]),
                      look_at = torch.tensor([0.0, 0.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([45.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      fisheye = False)

mat_grey = pyredner.Material(\
    diffuse_reflectance = \
        torch.tensor([0.4, 0.4, 0.4], device = pyredner.get_device()),
    specular_reflectance = \
        torch.tensor([0.5, 0.5, 0.5], device = pyredner.get_device()),
    roughness = \
        torch.tensor([0.05], device = pyredner.get_device()))

materials = [mat_grey]

vertices, indices, uvs, normals = generate_sphere(128, 64)
shape_sphere = pyredner.Shape(\
    vertices = vertices,
    indices = indices,
    uvs = uvs,
    normals = normals,
    material_id = 0)
shapes = [shape_sphere]

envmap = pyredner.imread('sunsky.exr')
if pyredner.get_use_gpu():
    envmap = envmap.cuda()
envmap = pyredner.EnvironmentMap(envmap)
scene = pyredner.Scene(cam, shapes, materials, [], envmap)
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    max_bounces = 1)
render = pyredner.RenderFunction.apply
img = render(0, *scene_args)
pyredner.imwrite(img.cpu(), 'results/test_envmap/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_envmap/target.png')
target = pyredner.imread('results/test_envmap/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda()

envmap_texels = torch.tensor(0.5 * torch.ones([32, 64, 3],
    device = pyredner.get_device()),
    requires_grad = True)
envmap = pyredner.EnvironmentMap(torch.abs(envmap_texels))
scene = pyredner.Scene(cam, shapes, materials, [], envmap)
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    max_bounces = 1)
img = render(1, *scene_args)
pyredner.imwrite(img.cpu(), 'results/test_envmap/init.png')
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/test_envmap/init_diff.png')

optimizer = torch.optim.Adam([envmap_texels], lr=1e-2)
for t in range(600):
    print('iteration:', t)
    optimizer.zero_grad()
    envmap = pyredner.EnvironmentMap(torch.abs(envmap_texels))
    scene = pyredner.Scene(cam, shapes, materials, [], envmap)
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4,
        max_bounces = 1)
    img = render(t+1, *scene_args)
    pyredner.imwrite(img.cpu(), 'results/test_envmap/iter_{}.png'.format(t))
    pyredner.imwrite(torch.abs(envmap_texels).cpu(), 'results/test_envmap/envmap_{}.exr'.format(t))
    loss = torch.pow(img - target, 2).sum()
    print('loss:', loss.item())

    loss.backward()
    optimizer.step()

scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    max_bounces = 1)
img = render(202, *scene_args)
pyredner.imwrite(img.cpu(), 'results/test_envmap/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_envmap/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/test_envmap/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_envmap/iter_%d.png", "-vb", "20M",
    "results/test_envmap/out.mp4"])
