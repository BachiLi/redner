import pyredner
import torch

# Automatic UV mapping example adapted from tutorial 2

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

material_map, mesh_list, light_map = pyredner.load_obj('scenes/teapot.obj')
for _, mesh in mesh_list:
    mesh.normals = pyredner.compute_vertex_normal(mesh.vertices, mesh.indices)
    mesh.uvs, mesh.uv_indices = pyredner.compute_uvs(mesh.vertices, mesh.indices)

cam = pyredner.Camera(position = torch.tensor([0.0, 30.0, 200.0]),
                      look_at = torch.tensor([0.0, 30.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([45.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      fisheye = False)

material_id_map = {}
materials = []
count = 0
for key, value in material_map.items():
    material_id_map[key] = count
    count += 1
    materials.append(value)

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

envmap = pyredner.imread('sunsky.exr')
if pyredner.get_use_gpu():
    envmap = envmap.cuda()
envmap = pyredner.EnvironmentMap(envmap)

# Finally we construct our scene using all the variables we setup previously.
scene = pyredner.Scene(cam, shapes, materials, area_lights = [], envmap = envmap)
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 1)
render = pyredner.RenderFunction.apply
img = render(0, *scene_args)
pyredner.imwrite(img.cpu(), 'results/test_compute_uvs/img.exr')
pyredner.imwrite(img.cpu(), 'results/test_compute_uvs/img.png')
