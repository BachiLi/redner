import pyredner
import redner
import torch
import math

# Estimate the pose of a teapot object.
# This tutorial demonstrates:
# 1. how to render G buffer, such as depth, normal, albedo
# 2. how to use G buffer to do "deferred rendering" in pytorch, which bypasses the main path tracing
#    process in redner, resulting in fast approximate rendering
# You might want to read the wikipedia page first if you are not familiar with the concept
# of deferred rendering: https://en.wikipedia.org/wiki/Deferred_shading
#
# Like the second tutorial, we first render a target image, then perturb the
# rotation/translation parameters and optimize to match the target.

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

# Load from the teapot Wavefront object file just like tutorial 02
material_map, mesh_list, light_map = pyredner.load_obj('teapot.obj')
# Compute shading normal
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

# Get a list of materials from material_map
material_id_map = {}
materials = []
count = 0
for key, value in material_map.items():
    material_id_map[key] = count
    count += 1
    materials.append(value)

# Get a list of shapes
shapes = []
for mtl_name, mesh in mesh_list:
    shapes.append(pyredner.Shape(\
        vertices = mesh.vertices,
        indices = mesh.indices,
        uvs = mesh.uvs,
        normals = mesh.normals,
        material_id = material_id_map[mtl_name]))

# Construct the scene
# Unlike previous tutorials, here we don't setup any light sources
scene = pyredner.Scene(cam, shapes, materials, area_lights = [], envmap = None)
# Serialize the scene.
# There are two difference comparing to previous tutorials.
# 1. we set "max_bounces" to 0, so we do not perform path tracing at all.
# 2. there is an extra argument "channels", specifying the output as position, shading normal, and albedo
#    by default the channels is a list with a single item redner.channels.radiance, which contains
#    the path tracing output.
#    See channels.h for a full list of channels we support.
#    The channels argument can also be useful for e.g. RGBD rendering.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16, # Still need some samples for anti-aliasing
    max_bounces = 0,
    channels = [redner.channels.position,
                redner.channels.shading_normal,
                redner.channels.diffuse_reflectance])
render = pyredner.RenderFunction.apply
g_buffer = render(0, *scene_args)
# Now, since we specified the outputs to be position, normal, and albedo,
# g_buffer is a 9-channels image
pos = g_buffer[:, :, :3]
normal = g_buffer[:, :, 3:6]
albedo = g_buffer[:, :, 6:9]
# Next, we render the g-buffer into a final image
# For this we define a deferred_render function:
def deferred_render(pos, normal, albedo):
    # We assume a point light at the camera origin (0, 30, 200)
    # The lighting consists of a geometry term cos/d^2, albedo, and the light intensity
    light_pos = torch.tensor([0.0, 30.0, 200.0], device = pyredner.get_device())
    light_pos = light_pos.view(1, 1, 3)
    light_intensity = torch.tensor([10000.0, 10000.0, 10000.0], device = pyredner.get_device())
    light_intensity = light_intensity.view(1, 1, 3)
    light_dir = light_pos - pos
    # the d^2 term:
    light_dist_sq = torch.sum(light_dir * light_dir, 2, keepdim = True)
    light_dist = torch.sqrt(light_dist_sq)
    # Normalize light direction
    light_dir = light_dir / light_dist
    dot_l_n = torch.sum(light_dir * normal, 2, keepdim = True)
    return light_intensity * dot_l_n * (albedo / math.pi) / light_dist_sq 
img = deferred_render(pos, normal, albedo)
# Save the images
pyredner.imwrite(img.cpu(), 'results/fast_deferred_rendering/target.exr')
pyredner.imwrite(img.cpu(), 'results/fast_deferred_rendering/target.png')
# Load the targets back
target = pyredner.imread('results/fast_deferred_rendering/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda()

# Same as tutorial 2, perturb the scene by a translation and a rotation to the object
translation_params = torch.tensor([0.1, -0.1, 0.1],
    device = pyredner.get_device(), requires_grad=True)
translation = translation_params * 100.0
euler_angles = torch.tensor([0.1, -0.1, 0.1], requires_grad=True)
shape0_vertices = shapes[0].vertices.clone()
shape1_vertices = shapes[1].vertices.clone()
rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
if pyredner.get_use_gpu():
    rotation_matrix = rotation_matrix.cuda()
center = torch.mean(torch.cat([shape0_vertices, shape1_vertices]), 0)
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
    num_samples = 16, # Still need some samples for anti-aliasing
    max_bounces = 0,
    channels = [redner.channels.position,
                redner.channels.shading_normal,
                redner.channels.diffuse_reflectance])
# Render the initial guess.
g_buffer = render(1, *scene_args)
pos = g_buffer[:, :, :3]
normal = g_buffer[:, :, 3:6]
albedo = g_buffer[:, :, 6:9]
img = deferred_render(pos, normal, albedo)
# Save the images
pyredner.imwrite(img.cpu(), 'results/fast_deferred_rendering/init.png')
# Compute the difference and save the images.
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/fast_deferred_rendering/init_diff.png')

# Optimize for pose parameters.
optimizer = torch.optim.Adam([translation_params, euler_angles], lr=1e-2)
# Run 200 Adam iterations.
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: apply the mesh operation and render the image.
    translation = translation_params * 100.0
    rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
    if pyredner.get_use_gpu():
        rotation_matrix = rotation_matrix.cuda()
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
        num_samples = 4, # Use less in Adam iteration
        max_bounces = 0,
        channels = [redner.channels.position,
                    redner.channels.shading_normal,
                    redner.channels.diffuse_reflectance])
    # Important to use a different seed every iteration, otherwise the result
    # would be biased.
    g_buffer = render(t+1, *scene_args)
    pos = g_buffer[:, :, :3]
    normal = g_buffer[:, :, 3:6]
    albedo = g_buffer[:, :, 6:9]
    img = deferred_render(pos, normal, albedo)

    # Save the intermediate render.
    pyredner.imwrite(img.cpu(), 'results/fast_deferred_rendering/iter_{}.png'.format(t))
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
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
    channels = [redner.channels.position,
                redner.channels.shading_normal,
                redner.channels.diffuse_reflectance])
g_buffer = render(202, *scene_args)
pos = g_buffer[:, :, :3]
normal = g_buffer[:, :, 3:6]
albedo = g_buffer[:, :, 6:9]
img = deferred_render(pos, normal, albedo)
# Save the images and differences.
pyredner.imwrite(img.cpu(), 'results/fast_deferred_rendering/final.exr')
pyredner.imwrite(img.cpu(), 'results/fast_deferred_rendering/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/fast_deferred_rendering/final_diff.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/fast_deferred_rendering/iter_%d.png", "-vb", "20M",
    "results/fast_deferred_rendering/out.mp4"])