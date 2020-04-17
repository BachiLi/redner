# Tensorflow by default allocates all GPU memory, leaving very little for rendering.
# We set the environment variable TF_FORCE_GPU_ALLOW_GROWTH to true to enforce on demand
# memory allocation to reduce page faults.
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.compat.v1.enable_eager_execution() # redner only supports eager mode
import pyredner_tensorflow as pyredner

objects = pyredner.load_obj('scenes/teapot.obj', return_objects=True)
camera = pyredner.automatic_camera_placement(objects, resolution=(512, 512))
scene = pyredner.Scene(camera = camera, objects = objects)

light = pyredner.PointLight(position = (camera.position + tf.constant((0.0, 0.0, 100.0))),
                            intensity = tf.constant((20000.0, 30000.0, 20000.0)))

img = pyredner.render_deferred(scene = scene, lights = [light])
pyredner.imwrite(img, 'results/test_compute_vertex_normals/no_vertex_normal.exr')

for obj in objects:
    obj.normals = pyredner.compute_vertex_normal(obj.vertices, obj.indices, 'max')
scene = pyredner.Scene(camera = camera, objects = objects)
img = pyredner.render_deferred(scene = scene, lights = [light])
pyredner.imwrite(img, 'results/test_compute_vertex_normals/max_vertex_normal.exr')

for obj in objects:
    obj.normals = pyredner.compute_vertex_normal(obj.vertices, obj.indices, 'cotangent')
scene = pyredner.Scene(camera = camera, objects = objects)
img = pyredner.render_deferred(scene = scene, lights = [light])
pyredner.imwrite(img, 'results/test_compute_vertex_normals/cotangent_vertex_normal.exr')