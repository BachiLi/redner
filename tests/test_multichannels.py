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

generic_texture = torch.zeros(\
    256, 256, 16, device = pyredner.get_device()).uniform_(0.01, 0.99)

materials = [pyredner.Material(generic_texture=generic_texture)]

# Get a list of shapes
shapes = []
for mtl_name, mesh in mesh_list:
    shapes.append(pyredner.Shape(\
        vertices = mesh.vertices,
        indices = mesh.indices,
        uvs = mesh.uvs,
        normals = mesh.normals,
        material_id = 0)) # Set all materials to the generic texture

# Construct the scene.
# Don't setup any light sources, only use primary visibility.
scene = pyredner.Scene(cam, shapes, materials, area_lights = [], envmap = None)
# Serialize the scene.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 4, # Still need some samples for anti-aliasing
    max_bounces = 0,
    channels = [redner.channels.generic_texture])
render = pyredner.RenderFunction.apply
# g buffer is the 16-channels texture
g_buffer = render(0, *scene_args)
print(g_buffer.shape)
img = g_buffer[:, :, :3]
print(img.shape)

# Save the images
pyredner.imwrite(img.cpu(), 'results/test_multichannels/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_multichannels/target.png')

