import pyredner
import torch

# Estimate the pose of a teapot object.
# In this tutorial we will learn a few more advanced features of redner, 
# through a pose estimation example. In particular, we demonstrate:
# 1. How to load a Wavefront object file
# 2. How to generate smooth vertex normals
# 3. apply environment lighting
# 4. How to apply mesh operations (rotation and translation in this case) in PyTorch
# Like the first tutorial, we first render a target image, then perturb the
# rotation/translation parameters and optimize to match the target.

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

# Load from the teapot Wavefront object file.
# load_obj function returns three lists/dicts
# material_map is a dict containing all the materials used in the obj file,
# where the key is the name of material, and the value is a pyredner.Material
#
# mesh_list is a list containing all the meshes in the obj file, grouped by use_mtl calls.
# Each element in the list is a tuple with length 2, the first is the name of material,
# the second is a pyredner.TriangleMesh with mesh information.
#
# light_map is a Python dict, where the key is the material names with non zeros Ke,
# and the values are the Ke
material_map, mesh_list, light_map = pyredner.load_obj('teapot.obj')
# The teapot we loaded is relatively low-poly and doesn't have vertex normal.
# Fortunately we can compute the vertex normal from the neighbor vertices.
# We can use pyredner.compute_vertex_normal for this task:
# (Try commenting out the following two lines to see the differences in target images!)
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

# Next, we convert the materials loaded from the Wavefront object files to a
# Python list of material. At the same time we keep track of the id of the materials,
# so that we can assign them to the shapes later.
material_id_map = {}
materials = []
count = 0
for key, value in material_map.items():
    material_id_map[key] = count
    count += 1
    materials.append(value)

# Now we build a list of shapes using the list loaded from the Wavefront object file.
# Meshes loaded from .obj files may have different indices for uvs and normals,
# we use mesh.uv_indices and mesh.normal_indices to access them.
# This mesh does not have normal_indices so the value is None.
shapes = []
for mtl_name, mesh in mesh_list:
    assert(mesh.normal_indices is None)
    shapes.append(pyredner.Shape(\
        vertices = mesh.vertices,
        indices = mesh.indices,
        material_id = material_id_map[mtl_name],
        uvs = mesh.uvs,
        normals = mesh.normals,
        uv_indices = mesh.uv_indices))

# The previous tutorial used a mesh area light for the scene lighting, 
# here we use an environment light,
# which is a texture representing infinitely far away light sources in 
# spherical coordinates.
envmap = pyredner.imread('sunsky.exr')
if pyredner.get_use_gpu():
    envmap = envmap.cuda()
envmap = pyredner.EnvironmentMap(envmap)

# Finally we construct our scene using all the variables we setup previously.
scene = pyredner.Scene(cam, shapes, materials, area_lights = [], envmap = envmap)
# Like the previous tutorial, we serialize and render the scene, 
# save it as our target
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 1)
render = pyredner.RenderFunction.apply
img = render(0, *scene_args)
pyredner.imwrite(img.cpu(), 'results/pose_estimation/target.exr')
pyredner.imwrite(img.cpu(), 'results/pose_estimation/target.png')
target = pyredner.imread('results/pose_estimation/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda()

# Now we want to generate the initial guess.
# We want to rotate and translate the teapot. We do this by declaring
# PyTorch tensors of translation and rotation parameters,
# then apply them to all teapot vertices.
# The translation and rotation parameters have very different ranges, so we normalize them
# by multiplying the translation parameters 100 to map to the actual translation amounts.
translation_params = torch.tensor([0.1, -0.1, 0.1],
    device = pyredner.get_device(), requires_grad=True)
translation = translation_params * 100.0
euler_angles = torch.tensor([0.1, -0.1, 0.1], requires_grad=True)
# We obtain the teapot vertices we want to apply the transformation on.
shape0_vertices = shapes[0].vertices.clone()
shape1_vertices = shapes[1].vertices.clone()
# We can use pyredner.gen_rotate_matrix to generate 3x3 rotation matrices
rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
if pyredner.get_use_gpu():
    rotation_matrix = rotation_matrix.cuda()
center = torch.mean(torch.cat([shape0_vertices, shape1_vertices]), 0)
# We shift the vertices to the center, apply rotation matrix,
# then shift back to the original space.
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
    num_samples = 512,
    max_bounces = 1)
# Render the initial guess.
img = render(1, *scene_args)
# Save the images.
pyredner.imwrite(img.cpu(), 'results/pose_estimation/init.png')
# Compute the difference and save the images.
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/pose_estimation/init_diff.png')

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
        num_samples = 4, # We use less samples in the Adam loop.
        max_bounces = 1)
    # Important to use a different seed every iteration, otherwise the result
    # would be biased.
    img = render(t+1, *scene_args)
    # Save the intermediate render.
    pyredner.imwrite(img.cpu(), 'results/pose_estimation/iter_{}.png'.format(t))
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
    num_samples = 512,
    max_bounces = 1)
img = render(202, *scene_args)
# Save the images and differences.
pyredner.imwrite(img.cpu(), 'results/pose_estimation/final.exr')
pyredner.imwrite(img.cpu(), 'results/pose_estimation/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/pose_estimation/final_diff.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/pose_estimation/iter_%d.png", "-vb", "20M",
    "results/pose_estimation/out.mp4"])
