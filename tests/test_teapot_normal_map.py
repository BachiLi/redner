import pyredner
import redner
import numpy as np
import torch
import skimage.transform

# Optimize the normal map of a teapot

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

# Set up the pyredner scene for rendering:
material_map, mesh_list, light_map = pyredner.load_obj('scenes/teapot.obj')
for _, mesh in mesh_list:
    mesh.normals = pyredner.compute_vertex_normal(mesh.vertices, mesh.indices)

# Setup camera
cam = pyredner.Camera(position = torch.tensor([0.0, 30.0, 200.0]),
                      look_at = torch.tensor([0.0, 30.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([45.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      fisheye = False)

# Load normal map (downloaded from https://worldwidemuseum.wordpress.com/2013/01/13/carriage-plates-step-2/)
normal_map = pyredner.imread('scenes/brick_normal.jpg', gamma=1.0)
if pyredner.get_use_gpu():
    normal_map = normal_map.cuda(device = pyredner.get_device())
normal_map = pyredner.Texture(normal_map,
    uv_scale = torch.tensor([4.0, 4.0], device = pyredner.get_device()))

# Setup materials
material_id_map = {}
materials = []
count = 0
for key, value in material_map.items():
    material_id_map[key] = count
    count += 1
    # assign normal map
    value.normal_map = normal_map
    materials.append(value)

# Setup geometries
shapes = []
for mtl_name, mesh in mesh_list:
    shapes.append(pyredner.Shape(\
        vertices = mesh.vertices,
        indices = mesh.indices,
        uvs = mesh.uvs,
        normals = mesh.normals,
        material_id = material_id_map[mtl_name]))

# Setup environment map
envmap = pyredner.imread('sunsky.exr')
if pyredner.get_use_gpu():
    envmap = envmap.cuda(device = pyredner.get_device())
envmap = pyredner.EnvironmentMap(envmap)

# Finally we construct our scene using all the variables we setup previously.
scene = pyredner.Scene(cam, shapes, materials, area_lights = [], envmap = envmap)
# Serialize the scene
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    sampler_type = redner.SamplerType.sobol,
    max_bounces = 1)
# Render the scene as our target image.
render = pyredner.RenderFunction.apply
# Render. The first argument is the seed for RNG in the renderer.
img = render(0, *scene_args)
pyredner.imwrite(img.cpu(), 'results/test_teapot_normal_map/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_teapot_normal_map/target.png')
target = pyredner.imread('results/test_teapot_normal_map/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

# Generate initial guess -- set the normal map to (0.5, 0.5, 1) for all pixels
normal_map = torch.zeros(512, 512, 3, device = pyredner.get_device())
normal_map[:, :, 0:2] = 0.5
normal_map[:, :, 2] = 1
normal_map.requires_grad = True
normal_map_tex = pyredner.Texture(normal_map)
for m in materials:
    m.normal_map = normal_map_tex
scene = pyredner.Scene(cam, shapes, materials, area_lights = [], envmap = envmap)
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    sampler_type = redner.SamplerType.sobol,
    max_bounces = 1)
# Render the initial guess.
img = render(1, *scene_args)
pyredner.imwrite(img.cpu(), 'results/test_teapot_normal_map/init.png')

# Optimize for normal map.
optimizer = torch.optim.Adam([normal_map], lr=1e-2)
# Run 200 Adam iterations.
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()

    # Reassign the texture for differentiating mipmap construction
    normal_map_tex = pyredner.Texture(normal_map)
    for m in materials:
        m.normal_map = normal_map_tex
    scene = pyredner.Scene(cam, shapes, materials, area_lights = [], envmap = envmap)
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4,
        sampler_type = redner.SamplerType.sobol,
        max_bounces = 1)

    img = render(t+1, *scene_args)
    # Save the intermediate render.
    pyredner.imwrite(img.cpu(), 'results/test_teapot_normal_map/iter_{}.png'.format(t))
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Take a gradient descent step.
    optimizer.step()
    # Save the normal map
    pyredner.imwrite(normal_map.cpu(), 'results/test_teapot_normal_map/normal_{}.png'.format(t), gamma=1.0)

# Render the final result.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 256,
    sampler_type = redner.SamplerType.sobol,
    max_bounces = 1)
img = render(202, *scene_args)
# Save the images and differences.
pyredner.imwrite(img.cpu(), 'results/test_teapot_normal_map/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_teapot_normal_map/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/test_teapot_normal_map/final_diff.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_teapot_normal_map/iter_%d.png", "-vb", "20M",
    "results/test_teapot_normal_map/out.mp4"])
