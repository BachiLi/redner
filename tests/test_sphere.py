import torch
import pyredner

vertices, indices, uvs, normals = pyredner.generate_sphere(64, 128)
m = pyredner.Material(diffuse_reflectance = torch.tensor((0.5, 0.5, 0.5), device = pyredner.get_device()))
obj = pyredner.Object(vertices = vertices,
                      indices = indices,
                      uvs = uvs,
                      normals = normals,
                      material = m)
cam = pyredner.automatic_camera_placement([obj], resolution = (480, 640))
scene = pyredner.Scene(objects = [obj], camera = cam)

img = pyredner.render_g_buffer(scene, channels = [pyredner.channels.uv, pyredner.channels.shading_normal])
uv_img = torch.cat([img[:, :, :2], torch.zeros(480, 640, 1)], dim=2)
normal_img = img[:, :, 2:]
pyredner.imwrite(uv_img, 'results/test_sphere/uv.png')
pyredner.imwrite(normal_img, 'results/test_sphere/normal.png')
