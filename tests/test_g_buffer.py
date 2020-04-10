import pyredner
import redner
import numpy as np
import torch
import skimage.transform

# Optimize depth and normal of a teapot

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

# Setup materials
material_id_map = {}
materials = []
count = 0
for key, value in material_map.items():
    material_id_map[key] = count
    count += 1
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

# We don't setup any light source here

# Construct the scene
scene = pyredner.Scene(cam, shapes, materials, area_lights = [], envmap = None)
# Serialize the scene
# Here we specify the output channels as "depth", "shading_normal"
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 0,
    channels = [redner.channels.depth, redner.channels.shading_normal])

# Render the scene as our target image.
render = pyredner.RenderFunction.apply
# Render. The first argument is the seed for RNG in the renderer.
img = render(0, *scene_args)
# Save the images.
depth = img[:, :, 0]
normal = img[:, :, 1:4]
pyredner.imwrite(depth.cpu(),
    'results/test_g_buffer/target_depth.exr')
pyredner.imwrite(depth.cpu(),
    'results/test_g_buffer/target_depth.png', normalize = True)
pyredner.imwrite(normal.cpu(),
    'results/test_g_buffer/target_normal.exr')
pyredner.imwrite(normal.cpu(),
    'results/test_g_buffer/target_normal.png', normalize = True)
# Read the target image we just saved.
target_depth = pyredner.imread('results/test_g_buffer/target_depth.exr')
target_normal = pyredner.imread('results/test_g_buffer/target_normal.exr')
if pyredner.get_use_gpu():
    target_depth = target_depth.cuda(device = pyredner.get_device())
    target_normal = target_normal.cuda(device = pyredner.get_device())

# Perturb the teapot by a translation and a rotation to the object
translation_params = torch.tensor([0.1, -0.1, 0.1],
    device = pyredner.get_device(), requires_grad=True)
translation = translation_params * 100.0
euler_angles = torch.tensor([0.1, -0.1, 0.1], requires_grad=True)
# These are the vertices we want to apply the transformation
shape0_vertices = shapes[0].vertices.clone()
shape1_vertices = shapes[1].vertices.clone()
# We can use pyredner.gen_rotate_matrix to generate 3x3 rotation matrices
rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
if pyredner.get_use_gpu():
    rotation_matrix = rotation_matrix.cuda()
center = torch.mean(torch.cat([shape0_vertices, shape1_vertices]), 0)
# Shift the vertices to the center, apply rotation matrix,
# shift back to the original space
shapes[0].vertices = \
    (shape0_vertices - center) @ torch.t(rotation_matrix) + \
    center + translation
shapes[1].vertices = \
    (shape1_vertices - center) @ torch.t(rotation_matrix) + \
    center + translation
# Since we changed the vertices, we need to regenerate the shading normals
shapes[0].normals = pyredner.compute_vertex_normal(shapes[0].vertices, shapes[0].indices)
shapes[1].normals = pyredner.compute_vertex_normal(shapes[1].vertices, shapes[1].indices)
# We need to serialize the scene again to get the new arguments.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 0,
    channels = [redner.channels.depth, redner.channels.shading_normal])
# Render the initial guess.
img = render(1, *scene_args)
depth = img[:, :, 0]
normal = img[:, :, 1:4]
# Save the images.
pyredner.imwrite(depth.cpu(),
    'results/test_g_buffer/init_depth.png',
    normalize = True)
pyredner.imwrite(depth.cpu(),
    'results/test_g_buffer/init_normal.png',
    normalize = True)
# Compute the difference and save the images.
diff_depth = torch.abs(target_depth - depth)
diff_normal = torch.abs(target_normal - normal)
pyredner.imwrite(diff_depth.cpu(),
    'results/test_g_buffer/init_depth_diff.png')
pyredner.imwrite(diff_normal.cpu(),
    'results/test_g_buffer/init_normal_diff.png')

# Optimize for triangle vertices.
optimizer = torch.optim.Adam([translation_params, euler_angles], lr=1e-2)
# Run 200 Adam iterations.
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: apply the mesh operation and render the image.
    translation = translation_params * 100.0
    rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
    if pyredner.get_use_gpu():
        rotation_matrix = rotation_matrix.cuda(device = pyredner.get_device())
    center = torch.mean(torch.cat([shape0_vertices, shape1_vertices]), 0)
    shapes[0].vertices = \
        (shape0_vertices - center) @ torch.t(rotation_matrix) + \
        center + translation
    shapes[1].vertices = \
        (shape1_vertices - center) @ torch.t(rotation_matrix) + \
        center + translation
    shapes[0].normals = pyredner.compute_vertex_normal(shapes[0].vertices, shapes[0].indices)
    shapes[1].normals = pyredner.compute_vertex_normal(shapes[1].vertices, shapes[1].indices)
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4, # We use less samples in the Adam loop.
        max_bounces = 0,
        channels = [redner.channels.depth, redner.channels.shading_normal])
    # Important to use a different seed every iteration, otherwise the result
    # would be biased.
    img = render(t+1, *scene_args)
    depth = img[:, :, 0]
    normal = img[:, :, 1:4]
    # Save the intermediate render.
    pyredner.imwrite(depth.cpu(),
        'results/test_g_buffer/iter_depth_{}.png'.format(t),
        normalize = True)
    pyredner.imwrite(normal.cpu(),
        'results/test_g_buffer/iter_normal_{}.png'.format(t),
        normalize = True)
    # Compute the loss function. Here it is L2.
    loss = (depth - target_depth).pow(2).sum() / 200.0 + (normal - target_normal).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients
    print('translation_params.grad:', translation_params.grad)
    print('euler_angles.grad:', euler_angles.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current pose parameters.
    print('translation:', translation)
    print('euler_angles:', euler_angles)

# Render the final result.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 0,
    channels = [redner.channels.depth, redner.channels.shading_normal])
img = render(202, *scene_args)
depth = img[:, :, 0]
normal = img[:, :, 1:4]
# Save the images.
pyredner.imwrite(depth.cpu(),
    'results/test_g_buffer/final_depth.exr')
pyredner.imwrite(depth.cpu(),
    'results/test_g_buffer/init_depth.png',
    normalize = True)
pyredner.imwrite(normal.cpu(),
    'results/test_g_buffer/final_normal.exr')
pyredner.imwrite(normal.cpu(),
    'results/test_g_buffer/final_normal.png',
    normalize = True)
diff_depth = torch.abs(target_depth - depth)
diff_normal = torch.abs(target_normal - normal)
pyredner.imwrite(diff_depth.cpu(),
    'results/test_g_buffer/init_depth_diff.png')
pyredner.imwrite(diff_normal.cpu(),
    'results/test_g_buffer/init_normal_diff.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_g_buffer/iter_depth_%d.png", "-vb", "20M",
    "results/test_g_buffer/out_depth.mp4"])
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_g_buffer/iter_normal_%d.png", "-vb", "20M",
    "results/test_g_buffer/out_normal.mp4"])
