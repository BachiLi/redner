import pyredner
import redner
import torch
import math

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

# Load from the teapot Wavefront object file
material_map, mesh_list, light_map = pyredner.load_obj('../tutorials/teapot.obj')
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

# Get a list of shapes
shapes = []
for mtl_name, mesh in mesh_list:
    shapes.append(pyredner.Shape(\
        vertices = mesh.vertices,
        indices = mesh.indices,
        uvs = mesh.uvs,
        normals = mesh.normals,
        material_id = 0)) # Set all materials to the generic texture

render = pyredner.RenderFunction.apply

tex_path='../tutorials/teapot.png'
tex_tensor = pyredner.imread(tex_path)
if pyredner.get_use_gpu():
    tex_tensor = tex_tensor.cuda(device = pyredner.get_device())


### TEST 1: regular 3-channels texture rasterization

generic_texture = tex_tensor

materials = [pyredner.Material(generic_texture=generic_texture)]

# Construct the scene.
# Don't setup any light sources, only use primary visibility.
scene = pyredner.Scene(cam, shapes, materials, area_lights = [], envmap = None)
# Serialize the scene.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 4, # Still need some samples for anti-aliasing
    max_bounces = 0,
    channels = [redner.channels.generic_texture])

# g buffer is the 16-channels texture
g_buffer = render(0, *scene_args)
print(g_buffer.shape)
img = g_buffer[:, :, 0:3]
print(img.shape)

# Save the images
pyredner.imwrite(img.cpu(), 'results/test_multichannels/target_test1.exr')
pyredner.imwrite(img.cpu(), 'results/test_multichannels/target_test1.png')

### TEST 2: 16-channels texture rasterization

generic_texture = generic_texture = torch.zeros(\
    128, 128, 16, device = pyredner.get_device())
generic_texture[:, :, 9:12] = tex_tensor

materials = [pyredner.Material(generic_texture=generic_texture)]

# Construct the scene.
# Don't setup any light sources, only use primary visibility.
scene = pyredner.Scene(cam, shapes, materials, area_lights = [], envmap = None)
# Serialize the scene.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 4, # Still need some samples for anti-aliasing
    max_bounces = 0,
    channels = [redner.channels.generic_texture])
# g buffer is the 16-channels texture
g_buffer = render(0, *scene_args)
print(g_buffer.shape)
img = g_buffer[:, :, 9:12]
print(img.shape)

# Save the images
pyredner.imwrite(img.cpu(), 'results/test_multichannels/target_test2.exr')
pyredner.imwrite(img.cpu(), 'results/test_multichannels/target_test2.png')





### TEST 3: test generic_texture gradients: we start from black texture, and must
#obtain at the end a 16-channels texture with the expected channels at the right positions

#we use test1 output as target
target = pyredner.imread('results/test_multichannels/target_test1.png')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

generic_texture = generic_texture = torch.zeros(\
    128, 128, 16, device = pyredner.get_device(), requires_grad=True)

materials = [pyredner.Material(generic_texture=generic_texture)]
scene = pyredner.Scene(cam, shapes, materials, area_lights = [], envmap = None)

optimizer = torch.optim.Adam([generic_texture], lr=1e-2)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()

    scene.materials[0].generic_texture = pyredner.Texture(generic_texture)

    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4, # Still need some samples for anti-aliasing
        max_bounces = 0,
        channels = [redner.channels.generic_texture])

    g_buffer = render(t, *scene_args)
    img = g_buffer[:, :, 12:15]
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())
    loss.backward()
    optimizer.step()

# Save the images
# To compute the loss we used g_buffer channels 12 to 15 so the optimization process
# should have caused generic_texture channels 12 to 15 to converge towards tex_tensor (the original teapot texture)
pyredner.imwrite(generic_texture[:, :, 12:15].cpu(), 'results/test_multichannels/target_test3.exr')
pyredner.imwrite(generic_texture[:, :, 12:15].cpu(), 'results/test_multichannels/target_test3.png')
