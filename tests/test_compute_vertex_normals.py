import pyredner
import torch

objects = pyredner.load_obj('scenes/teapot.obj', return_objects=True)
camera = pyredner.automatic_camera_placement(objects, resolution=(512, 512))
scene = pyredner.Scene(camera = camera, objects = objects)

light = pyredner.PointLight(position = (camera.position + torch.tensor((0.0, 0.0, 100.0))).to(pyredner.get_device()),
                            intensity = torch.tensor((20000.0, 30000.0, 20000.0), device = pyredner.get_device()))

img = pyredner.render_deferred(scene = scene, lights = [light])
pyredner.imwrite(img.cpu(), 'results/test_compute_vertex_normals/no_vertex_normal.exr')

for obj in objects:
    obj.normals = pyredner.compute_vertex_normal(obj.vertices, obj.indices, 'max')
scene = pyredner.Scene(camera = camera, objects = objects)
img = pyredner.render_deferred(scene = scene, lights = [light])
pyredner.imwrite(img.cpu(), 'results/test_compute_vertex_normals/max_vertex_normal.exr')

for obj in objects:
    obj.normals = pyredner.compute_vertex_normal(obj.vertices, obj.indices, 'cotangent')
scene = pyredner.Scene(camera = camera, objects = objects)
img = pyredner.render_deferred(scene = scene, lights = [light])
pyredner.imwrite(img.cpu(), 'results/test_compute_vertex_normals/cotangent_vertex_normal.exr')