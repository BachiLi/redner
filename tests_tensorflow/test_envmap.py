import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager

import pyrednertensorflow as pyredner
import numpy as np
import math
import pdb
# Use GPU if available
pyredner.set_use_gpu(False)

cam = pyredner.Camera(position = tf.Variable([0.0, 0.0, -5.0], dtype=tf.float32),
                      look_at = tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32),
                      up = tf.Variable([0.0, 1.0, 0.0], dtype=tf.float32),
                      fov = tf.Variable([45.0], dtype=tf.float32), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      fisheye = False)

mat_grey = pyredner.Material(
    diffuse_reflectance = \
        tf.Variable([0.4, 0.4, 0.4], ),
    specular_reflectance = \
        tf.Variable([0.5, 0.5, 0.5], ),
    roughness = \
        tf.Variable([0.05], ))

materials = [mat_grey]

vertices, indices, uvs, normals = pyredner.generate_sphere(128, 64)
shape_sphere = pyredner.Shape(
    vertices = vertices,
    indices = indices,
    uvs = uvs,
    normals = normals,
    material_id = 0)
shapes = [shape_sphere]

envmap = pyredner.imread('sunsky.exr')
envmap = pyredner.EnvironmentMap(envmap)
scene = pyredner.Scene(cam, shapes, materials, [], envmap)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 256,
    max_bounces = 1)

img = pyredner.render(0, *scene_args)
pyredner.imwrite(img, 'results/test_envmap/target.exr')
pyredner.imwrite(img, 'results/test_envmap/target.png')
target = pyredner.imread('results/test_envmap/target.exr')

envmap_texels = tf.Variable(0.5 * np.ones([32, 64, 3], dtype=np.float32),
    trainable=True)
envmap = pyredner.EnvironmentMap(tf.abs(envmap_texels))
scene = pyredner.Scene(cam, shapes, materials, [], envmap)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 256,
    max_bounces = 1)
img = pyredner.render(1, *scene_args)
pyredner.imwrite(img, 'results/test_envmap/init.png')
diff = tf.abs(target - img)
pyredner.imwrite(diff, 'results/test_envmap/init_diff.png')

# optimizer = torch.optim.Adam([envmap_texels], lr=1e-2)
"""NOTE:
envmap_texels -> EnvironmentMap.values -> Texture.texels
"""

optimizer = tf.train.AdamOptimizer(1e-2)
for t in range(600):
    print('iteration:', t)
    # with tf.GradientTape(persistent=True) as tape:
    with tf.GradientTape() as tape:
        envmap = pyredner.EnvironmentMap(tf.abs(envmap_texels))

        scene = pyredner.Scene(cam, shapes, materials, [], envmap)
        scene_args = pyredner.serialize_scene(
            scene = scene,
            num_samples = 4,
            max_bounces = 1)
        '''
        for i in range(len(pyredner.get_render_args(t+1, scene_args))): print(pyredner.get_render_args(t+1, scene_args)[i].shape, i)

        for i in range(len(scene_args)): print(scene_args)[i].shape, i)

        pyredner.render(t+1, *scene_args)
        '''
        img = pyredner.render(t+1, *scene_args)

        loss = tf.reduce_sum(tf.square(img - target))
        print('loss:', loss)

    pyredner.imwrite(img, 'results/test_envmap/iter_{}.png'.format(t))
    pyredner.imwrite(tf.abs(envmap_texels), 'results/test_envmap/envmap_{}.exr'.format(t))

    grads = tape.gradient(loss, envmap_texels)
    # pdb.set_trace()

    # tape.gradient(envmap.values.mipmap, envmap_texels)
    # tape2.gradient(loss, envmap.values.mipmap)
    # tape.gradient(loss, envmap_texels)
    # tape.gradient(loss, cam.position)
    # tape.gradient(loss, cam.up)
    # tape.gradient(loss, cam.look_at)
    # tape.gradient(loss, cam.ndc_to_cam)
    # tape.gradient(loss, cam.cam_to_ndc)
    # tape.gradient(loss, cam.)
    
    optimizer.apply_gradients(zip([grads], [envmap_texels]))


scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 256,
    max_bounces = 1)
img = pyredner.render(602, *scene_args)
pyredner.imwrite(img, 'results/test_envmap/final.exr')
pyredner.imwrite(img, 'results/test_envmap/final.png')
pyredner.imwrite(tf.abs(target - img), 'results/test_envmap/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_envmap/iter_%d.png", "-vb", "20M",
    "results/test_envmap/out.mp4"])
