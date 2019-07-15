import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager

import pyrednertensorflow as pyredner
import numpy as np
import pdb
# Optimize fisheye camera parameters of a single triangle rendering

# Use GPU if available
pyredner.set_use_gpu(False)

# Set up the scene using Pytorch tensor
position = tfe.Variable([0.0, 0.0, -1.0])
look_at = tfe.Variable([0.0, 0.0, 0.0], dtype=tf.float32)
up = tfe.Variable([0.0, 1.0, 0.0], dtype=tf.float32)
fov = tfe.Variable([45.0], dtype=tf.float32)
clip_near = 1e-2

resolution = (256, 256)
cam = pyredner.Camera(position = position,
                      look_at = look_at,
                      up = up,
                      fov = fov,
                      clip_near = clip_near,
                      resolution = resolution,
                      fisheye = True)

mat_grey = pyredner.Material(
    diffuse_reflectance = tfe.Variable([0.5, 0.5, 0.5],
                                       ))
materials = [mat_grey]
vertices = tfe.Variable([[-1.7,1.0,0.0], [1.0,1.0,0.0], [-0.5,-1.0,0.0]], )
indices = tfe.Variable([[0, 1, 2]], dtype=tf.int32, )
shape_triangle = pyredner.Shape(vertices, indices, None, None, 0)
light_vertices = tfe.Variable([[-1.0,-1.0,-9.0],[1.0,-1.0,-9.0],[-1.0,1.0,-9.0],[1.0,1.0,-9.0]],
                              )
light_indices = tfe.Variable([[0,1,2],[1,3,2]], dtype=tf.int32, )
shape_light = pyredner.Shape(light_vertices, light_indices, None, None, 0)
shapes = [shape_triangle, shape_light]
light_intensity = tfe.Variable([30.0,30.0,30.0])
light = pyredner.AreaLight(1, light_intensity)
area_lights = [light]
scene = pyredner.Scene(cam, shapes, materials, area_lights)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)

# Alias of the render function
# Render our target
img = pyredner.render(d, *scene_args)
pyredner.imwrite(img, 'results/test_single_triangle_camera_fisheye/target.exr')
pyredner.imwrite(img, 'results/test_single_triangle_camera_fisheye/target.png')
target = pyredner.imread('results/test_single_triangle_camera_fisheye/target.exr')

# Perturb the scene, this is our initial guess
position = tfe.Variable([0.5, -0.5, -3.0], trainable=True, name="camera_position")
scene.camera = pyredner.Camera(position = position,
                               look_at = look_at,
                               up = up,
                               fov = fov,
                               clip_near = clip_near,
                               resolution = resolution,
                               fisheye = True)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
# Render the initial guess
img = pyredner.render(d, *scene_args)
pyredner.imwrite(img, 'results/test_single_triangle_camera_fisheye/init.png')
diff = tf.abs(target - img)
pyredner.imwrite(diff, 'results/test_single_triangle_camera_fisheye/init_diff.png')

# Optimize for camera pose
# optimizer = torch.optim.Adam([position], lr=5e-2)
optimizer = tf.train.AdamOptimizer(2e-2)
for t in range(1000):
    print('iteration:', t)
    
    with tf.GradientTape() as tape:
        # Need to rerun the Camera constructor for PyTorch autodiff to compute the derivatives
        scene.camera = pyredner.Camera(position   = position,
                                    look_at    = look_at,
                                    up         = up,
                                    fov        = fov,
                                    clip_near  = clip_near,
                                    resolution = resolution,
                                    fisheye    = True)
        scene_args = pyredner.serialize_scene(
            scene = scene,
            num_samples = 4,
            max_bounces = 1)
        img = pyredner.render(d+1, *scene_args)
        pyredner.imwrite(img, 'results/test_single_triangle_camera_fisheye/iter_{}.png'.format(t))
        loss = tf.reduce_sum(tf.square(img - target))
    
    print('loss:', loss)

    grads = tape.gradient(loss, [position])

    optimizer.apply_gradients(
        zip(grads, [position])
        )

    print('d_position:', grads[0])
    print('position:', position)

scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
img = pyredner.render(d02, *scene_args)
pyredner.imwrite(img, 'results/test_single_triangle_camera_fisheye/final.exr')
pyredner.imwrite(img, 'results/test_single_triangle_camera_fisheye/final.png')
pyredner.imwrite(tf.abs(target - img).cpu(), 'results/test_single_triangle_camera_fisheye/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_single_triangle_camera_fisheye/iter_%d.png", "-vb", "20M",
    "results/test_single_triangle_camera_fisheye/out.mp4"])

