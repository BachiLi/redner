import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager
import pyrednertensorflow as pyredner
import redner
import numpy as np
import skimage.transform

# Optimize three vertices of a single triangle, with a SIGGRAPH logo background

# Use GPU if available
pyredner.set_use_gpu(False)

# Set up the pyredner scene for rendering:

# Setup camera
cam = pyredner.Camera(position = tfe.Variable([0.0, 0.0, -5.0], dtype=tf.float32),
                      look_at = tfe.Variable([0.0, 0.0, 0.0], dtype=tf.float32),
                      up = tfe.Variable([0.0, 1.0, 0.0], dtype=tf.float32),
                      fov = tfe.Variable([45.0], dtype=tf.float32), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      fisheye = False)

# Setup materials
mat_grey = pyredner.Material(
    diffuse_reflectance = \
        tfe.Variable([0.5, 0.5, 0.5], dtype=tf.float32))
# The material list of the scene
materials = [mat_grey]

# Setup geometries
shape_triangle = pyredner.Shape(
    vertices = tfe.Variable([[-1.7, 1.0, 0.0], [1.0, 1.0, 0.0], [-0.5, -1.0, 0.0]], dtype=tf.float32),
    indices = tfe.Variable([[0, 1, 2]], dtype=tf.int32),
    uvs = None,
    normals = None,
    material_id = 0)
# Setup light source shape
shape_light = pyredner.Shape(
    vertices = tfe.Variable([[-1.0, -1.0, -7.0],
                             [ 1.0, -1.0, -7.0],
                             [-1.0,  1.0, -7.0],
                             [ 1.0,  1.0, -7.0]], dtype=tf.float32),
    indices = tfe.Variable([[0, 1, 2],[1, 3, 2]], dtype=tf.int32),
    uvs = None,
    normals = None,
    material_id = 0)
# The shape list of the scene
shapes = [shape_triangle, shape_light]

# Setup light source
light = pyredner.AreaLight(shape_id = 1, 
                           intensity = tfe.Variable([20.0,20.0,20.0], dtype=tf.float32))
area_lights = [light]

# Construct the scene
scene = pyredner.Scene(cam, shapes, materials, area_lights)
# Serialize the scene
# Here we specify the output channels as "radiance" and "alpha"
# Render the scene as our target image.
scene_args = pyredner.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1,
    channels = [redner.channels.radiance, redner.channels.alpha])

# Render the scene as our target image.
# Render. The first argument is the seed for RNG in the renderer.
img = pyredner.render(0, *scene_args)

background = pyredner.imread('scenes/textures/siggraph.jpg')
background = tf.convert_to_tensor(skimage.transform.resize(background.numpy(), (256, 256, 3)), dtype=tf.float32)
img = img[:, :, :3] * img[:, :, 3:4] + background * (1 - img[:, :, 3:4])

# Save the images.
# The output image is in the GPU memory if you are using GPU.
pyredner.imwrite(img, 'results/test_single_triangle_background/target.exr')
pyredner.imwrite(img, 'results/test_single_triangle_background/target.png')
# Read the target image we just saved.
target = pyredner.imread('results/test_single_triangle_background/target.exr')

DEBUG = False
if not DEBUG:
    """
    NOTE: Pass channel! It's different from test_single_triangle.py 
    """
    scene_args = pyredner.serialize_scene(
        scene = scene,
        num_samples = 16,
        max_bounces = 1,
        channels = [redner.channels.radiance, redner.channels.alpha])

    # Render. The first argument is the seed for RNG in the renderer.
    img = pyredner.render(0, *scene_args)
    # Since we specified alpha as output channel, img has 4 channels now
    # We blend the image with a background image
    img = img[:, :, :3] * img[:, :, 3:4] + background * (1 - img[:, :, 3:4])

    # Save the images.
    # The output image is in the GPU memory if you are using GPU.
    pyredner.imwrite(img, 'results/test_single_triangle_background/target.exr')
    pyredner.imwrite(img, 'results/test_single_triangle_background/target.png')


    # Perturb the scene, this is our initial guess.
    shape_triangle.vertices = tfe.Variable(
        [[-2.0,1.5,0.3], [0.9,1.2,-0.3], [-0.4,-1.4,0.2]],
        dtype=tf.float32,
        trainable=True)
    # We need to serialize the scene again to get the new arguments.
    scene_args = pyredner.serialize_scene(
        scene = scene,
        num_samples = 16,
        max_bounces = 1,
        channels = [redner.channels.radiance, redner.channels.alpha])  # NOTE: 

    # Render the initial guess.
    img = pyredner.render(1, *scene_args)
    # Blend the image with a background image
    img = img[:, :, :3] * img[:, :, 3:4] + background * (1 - img[:, :, 3:4])
    # Save the images.
    pyredner.imwrite(img, 'results/test_single_triangle_background/init.png')
    # Compute the difference and save the images.
    diff = tf.abs(target - img)
    pyredner.imwrite(diff, 'results/test_single_triangle_background/init_diff.png')

# Optimize for triangle vertices.
optimizer = tf.train.AdamOptimizer(5e-2)
# Run 300 Adam iterations.

for t in range(300):
    print('iteration:', t)

    scene_args = pyredner.serialize_scene(
            scene = scene,
            num_samples = 4, # We use less samples in the Adam loop.
            max_bounces = 1,
            channels = [redner.channels.radiance, redner.channels.alpha])
    
    with tf.GradientTape() as tape:
        # Forward pass: render the image.
        # Important to use a different seed every iteration, otherwise the result
        # would be biased.
        img = pyredner.render(t+1, *scene_args)
        img4c = tf.identity(img)
        # Blend the image with a background image
        img = img4c[:, :, :3] * img4c[:, :, 3:4] + background * (1 - img4c[:, :, 3:4])
        # Save the intermediate render.
        pyredner.imwrite(img, 'results/test_single_triangle_background/iter_{}.png'.format(t))
        # Compute the loss function. Here it is L2.
        loss = tf.reduce_sum(tf.square(img - target))

    print('loss:', loss)

    grads = tape.gradient(loss, [shape_triangle.vertices])
    print('grad:', grads)
    optimizer.apply_gradients(
        zip(grads, [shape_triangle.vertices])
        )

    # Print the current three vertices.
    # print('vertices:', shape_triangle.vertices)

# Render the final result.
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1,
    channels = [redner.channels.radiance, redner.channels.alpha])
img = pyredner.render(302, *scene_args)
# Blend the image with a background image
img = img[:, :, :3] * img[:, :, 3:4] + background * (1 - img[:, :, 3:4])
# Save the images and differences.
pyredner.imwrite(img, 'results/test_single_triangle_background/final.exr')
pyredner.imwrite(img, 'results/test_single_triangle_background/final.png')
pyredner.imwrite(tf.abs(target - img).cpu(), 'results/test_single_triangle_background/final_diff.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_single_triangle_background/iter_%d.png", "-vb", "20M",
    "results/test_single_triangle_background/out.mp4"])
