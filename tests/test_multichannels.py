import pyredner
import redner
import torch
import math


# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

# Load from the teapot Wavefront object file just like tutorial 02
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


Ns=30.0000
Ks=(0.0, 0.0, 0.0)
Kd = (1.0, 1.0, 1.0)
tex_path='../tutorials/teapot.png'

generic_texture = pyredner.imread(tex_path)
if pyredner.get_use_gpu():
    generic_texture = generic_texture.cuda(device = pyredner.get_device())

diffuse_reflectance = pyredner.imread(tex_path)
if pyredner.get_use_gpu():
    diffuse_reflectance = diffuse_reflectance.cuda(device = pyredner.get_device())

#diffuse_reflectance = torch.tensor(Kd,
#    dtype = torch.float32, device = pyredner.get_device())
roughness = torch.tensor([2.0 / (Ns + 2.0)],
    dtype = torch.float32, device = pyredner.get_device())
specular_reflectance = torch.tensor(Ks,
    dtype = torch.float32, device = pyredner.get_device())

materials = [
pyredner.Material(diffuse_reflectance, specular_reflectance, roughness, generic_texture)
]




# Get a list of shapes
shapes = []
for mtl_name, mesh in mesh_list:
    shapes.append(pyredner.Shape(\
        vertices = mesh.vertices,
        indices = mesh.indices,
        uvs = mesh.uvs,
        normals = mesh.normals,
        material_id = 0))

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
                redner.channels.diffuse_reflectance,
                redner.channels.generic_texture
               ])
render = pyredner.RenderFunction.apply
g_buffer = render(0, *scene_args)
# Now, since we specified the outputs to be position, normal, and albedo,
# g_buffer is a 9-channels image
pos = g_buffer[:, :, :3]
normal = g_buffer[:, :, 3:6]
albedo = g_buffer[:, :, 6:9]
generic_tex = g_buffer[:, :, 9:12]
# Next, we render the g-buffer into a final image
# For this we define a deferred_render function:
def deferred_render(pos, normal, albedo, generic_tex):
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
    return light_intensity * dot_l_n * (generic_tex / math.pi) / light_dist_sq 
img = deferred_render(pos, normal, albedo, generic_tex)
# Save the images
pyredner.imwrite(img.cpu(), '../target.exr')
pyredner.imwrite(img.cpu(), '../target.png')
#
