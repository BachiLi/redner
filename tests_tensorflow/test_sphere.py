# Tensorflow by default allocates all GPU memory, leaving very little for rendering.
# We set the environment variable TF_FORCE_GPU_ALLOW_GROWTH to true to enforce on demand
# memory allocation to reduce page faults.
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pyredner_tensorflow as pyredner

vertices, indices, uvs, normals = pyredner.generate_sphere(64, 128)
m = pyredner.Material(diffuse_reflectance = tf.constant((0.5, 0.5, 0.5)))
obj = pyredner.Object(vertices = vertices,
                      indices = indices,
                      uvs = uvs,
                      normals = normals,
                      material = m)
cam = pyredner.automatic_camera_placement([obj], resolution = (480, 640))
scene = pyredner.Scene(objects = [obj], camera = cam)

img = pyredner.render_g_buffer(scene, channels = [pyredner.channels.uv, pyredner.channels.shading_normal])
uv_img = tf.concat([img[:, :, :2], tf.zeros((480, 640, 1))], axis=2)
normal_img = img[:, :, 2:]
pyredner.imwrite(uv_img, 'results/test_sphere/uv.png')
pyredner.imwrite(normal_img, 'results/test_sphere/normal.png')
