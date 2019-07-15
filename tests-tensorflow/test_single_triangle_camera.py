import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager
import numpy as np

import pyrednertensorflow as pyredner
import pdb
# Optimize camera parameters of a single triangle rendering

# Use GPU if available
pyredner.set_use_gpu(False)

# Set up the scene using Pytorch tensor
position = tfe.Variable([0.0, 0.0, -5.0], dtype=tf.float32)
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
                      resolution = resolution)

mat_grey = pyredner.Material(
    diffuse_reflectance = tfe.Variable([0.5, 0.5, 0.5], dtype=tf.float32))
materials = [mat_grey]
vertices = tfe.Variable([[-1.7,1.0,0.0], [1.0,1.0,0.0], [-0.5,-1.0,0.0]], dtype=tf.float32)
indices = tfe.Variable([[0, 1, 2]], dtype=tf.int32)
shape_triangle = pyredner.Shape(vertices, indices, None, None, 0)
light_vertices = tfe.Variable([[-1.0,-1.0,-9.0],[1.0,-1.0,-9.0],[-1.0,1.0,-9.0],[1.0,1.0,-9.0]], dtype=tf.float32)
light_indices = tfe.Variable([[0,1,2],[1,3,2]], dtype=tf.int32)
shape_light = pyredner.Shape(light_vertices, light_indices, None, None, 0)
shapes = [shape_triangle, shape_light]
light_intensity = tfe.Variable([30.0,30.0,30.0],dtype=tf.float32)
light = pyredner.AreaLight(1, light_intensity)
area_lights = [light]
scene = pyredner.Scene(cam, shapes, materials, area_lights)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)

# Alias of the render function

# Render our target
img = pyredner.render(0, *scene_args)
pyredner.imwrite(img, 'results/test_single_triangle_camera/target.exr')
pyredner.imwrite(img, 'results/test_single_triangle_camera/target.png')
target = pyredner.imread('results/test_single_triangle_camera/target.exr')

# Perturb the scene, this is our initial guess
position = tfe.Variable([0.0,  0.0, -3.0], dtype=tf.float32, trainable=True)
look_at = tfe.Variable([-0.5, -0.5,  0.0], dtype=tf.float32, trainable=True)
scene.camera = pyredner.Camera(position = position,
                               look_at = look_at,
                               up = up,
                               fov = fov,
                               clip_near = clip_near,
                               resolution = resolution)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
# Render the initial guess
img = pyredner.render(1, *scene_args)
pyredner.imwrite(img, 'results/test_single_triangle_camera/init.png')
diff = tf.abs(target - img)
pyredner.imwrite(diff, 'results/test_single_triangle_camera/init_diff.png')


scene.camera = pyredner.Camera(position   = position,
                                look_at    = look_at,
                                up         = up,
                                fov        = fov,
                                clip_near  = clip_near,
                                resolution = resolution)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 4,
    max_bounces = 1)

# Optimize for camera pose
# optimizer = torch.optim.Adam([position, look_at], lr=5e-2)
optimizer = tf.train.AdamOptimizer(2e-2)
for t in range(200):
    print('iteration:', t)
    
    with tf.GradientTape() as tape:
        # Need to rerun the Camera constructor for PyTorch autodiff to compute the derivatives
        img = pyredner.render(t+1, *scene_args)
        pyredner.imwrite(img, 'results/test_single_triangle_camera/iter_{}.png'.format(t))
        loss = tf.reduce_sum(tf.square(img - target))
    print('loss:', loss)

    grads = tape.gradient(loss, [position, look_at])

    # pdb.set_trace()
    optimizer.apply_gradients(
        zip(grads, [position, look_at])
        )

    print('position.grad:', grads[0])
    print('d_look_at:', grads[0])

    print('position:', position)
    print('look_at:', look_at)
    # pdb.set_trace()

scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
img = pyredner.render(202, *scene_args)
pyredner.imwrite(img, 'results/test_single_triangle_camera/final.exr')
pyredner.imwrite(img, 'results/test_single_triangle_camera/final.png')
pyredner.imwrite(tf.abs(target - img).cpu(), 'results/test_single_triangle_camera/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_single_triangle_camera/iter_%d.png", "-vb", "20M",
    "results/test_single_triangle_camera/out.mp4"])
