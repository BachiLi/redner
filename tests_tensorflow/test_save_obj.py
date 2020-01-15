# Tensorflow by default allocates all GPU memory, leaving very little for rendering.
# We set the environment variable TF_FORCE_GPU_ALLOW_GROWTH to true to enforce on demand
# memory allocation to reduce page faults.
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pyredner_tensorflow as pyredner

vertices, indices, uvs, normals = pyredner.generate_sphere(16, 32)
m = pyredner.Material(diffuse_reflectance = tf.constant((0.5, 0.5, 0.5)))
obj = pyredner.Object(vertices = vertices,
                      indices = indices,
                      uvs = uvs,
                      normals = normals,
                      material = m)
filename = 'results/test_save_obj/sphere.obj'
directory = os.path.dirname(filename)
if directory != '' and not os.path.exists(directory):
    os.makedirs(directory)
pyredner.save_obj(obj, 'results/test_save_obj/sphere.obj')
