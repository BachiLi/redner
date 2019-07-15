import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager

import pyrednertensorflow as pyredner
import numpy as np
import pdb
# Optimize three vertices of a single triangle
# We first render a target image, then perturb the three vertices and optimize
# to match the target.

# Use GPU if available
pyredner.set_use_gpu(False)

# Set up the pyredner scene for rendering:

# First, we set up the camera.
# redner assumes all the camera variables live in CPU memory,
# so you should allocate torch tensors in CPU
cam = pyredner.Camera(position = tfe.Variable([0.0, 0.0, -5.0], dtype=tf.float32),
                      look_at = tfe.Variable([0.0, 0.0, 0.0], dtype=tf.float32),
                      up = tfe.Variable([0.0, 1.0, 0.0], dtype=tf.float32),
                      fov = tfe.Variable([45.0], dtype=tf.float32), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      fisheye = False)

# Next, we setup the materials for the scene.
# All materials in the scene are stored in a Python list,
# the index of a material in the list is its material id.
# Our simple scene only has a single grey material with reflectance 0.5.
# If you are using GPU, make sure to copy the reflectance to GPU memory.
mat_grey = pyredner.Material(
    diffuse_reflectance = \
        tfe.Variable([0.5, 0.5, 0.5], dtype=tf.float32))
# The material list of the scene
materials = [mat_grey]

# Next, we setup the geometry for the scene.
# 3D objects in redner are called "Shape".
# All shapes in the scene are stored in a Python list,
# the index of a shape in the list is its shape id.
# Right now, a shape is always a triangle mesh, which has a list of
# triangle vertices and a list of triangle indices.
# The vertices are a Nx3 torch float tensor,
# and the indices are a Mx3 torch integer tensor.
# Optionally, for each vertex you can specify its UV coordinate for texture mapping,
# and a normal for Phong interpolation.
# Each shape also needs to be assigned a material using material id,
# which is the index of the material in the material array.
# If you are using GPU, make sure to copy all tensors of the shape to GPU memory.
shape_triangle = pyredner.Shape(
    vertices = tfe.Variable([[-1.7, 1.0, 0.0], [1.0, 1.0, 0.0], [-0.5, -1.0, 0.0]],
        dtype=tf.float32),
    indices = tfe.Variable([[0, 1, 2]], dtype=tf.int32),
    uvs = None,
    normals = None,
    material_id = 0)
# Merely having a single triangle is not enough for physically-based rendering.
# We need to have a light source. Here we setup the shape of a quad area light source,
# similary to the previous triangle.
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

# Now we assign some of the shapes in the scene as light sources.
# Again, all the area light sources in the scene are stored in a Python list.
# Each area light is attached to a shape using shape id, additionally we need to
# assign the intensity of the light, which is a length 3 float tensor in CPU. 
light = pyredner.AreaLight(shape_id = 1, 
                           intensity = tfe.Variable([20.0,20.0,20.0], dtype=tf.float32))
area_lights = [light]
# Finally we construct our scene using all the variables we setup previously.
scene = pyredner.Scene(cam, shapes, materials, area_lights)
# All PyTorch functions take a flat array of PyTorch tensors as input,
# therefore we need to serialize the scene into an array. The following
# function is doing this. We also specify how many Monte Carlo samples we want to 
# use per pixel and the number of bounces for indirect illumination here
# (one bounce means only direct illumination).
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)

# Render the scene as our target image.
# To render the scene, we use our custom PyTorch function in pyredner/render_pytorch.py
# First setup the alias of the render function

# Render. The first argument is the seed for RNG in the renderer.
img = pyredner.render(0, *scene_args)
# Save the images.
# The output image is in the GPU memory if you are using GPU.
pyredner.imwrite(img, 'results/test_single_triangle/target.exr')
pyredner.imwrite(img, 'results/test_single_triangle/target.png')
# Read the target image we just saved.
target = pyredner.imread('results/test_single_triangle/target.exr')
# if pyredner.get_use_gpu():
#     target = target.cuda()

# Perturb the scene, this is our initial guess.
shape_triangle.vertices = tfe.Variable(
    [[-2.0,1.5,0.3], [0.9,1.2,-0.3], [-0.4,-1.4,0.2]],
    dtype=tf.float32,
    trainable=True) # Set trainable to True since we want to optimize this
# We need to serialize the scene again to get the new arguments.
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
# Render the initial guess.
img = pyredner.render(1, *scene_args)
# Save the images.
pyredner.imwrite(img, 'results/test_single_triangle/init.png')
# Compute the difference and save the images.
diff = tf.abs(target - img)
pyredner.imwrite(diff, 'results/test_single_triangle/init_diff.png')

# Optimize for triangle vertices.
# # optimizer = torch.optim.Adam([shape_triangle.vertices], lr=5e-2)
optimizer = tf.train.AdamOptimizer(5e-2)

def loss(output, target):
    # Compute the loss function. Here it is L2.
    error = output - target
    return tf.reduce_sum(tf.square(error))

def optimize(scene_args, grads, lr=5e-2):
    updates = []
    for var, grad in zip(scene_args, grads):
        if grad is None: 
            updates.append(var)
            continue
        # print(grad)
        # var -= lr * grad
        updates.append(var - lr * grad)

    return updates

scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 4, # We use less samples in the Adam loop.
    max_bounces = 1)
# Run 200 Adam iterations.
for t in range(1, 201):
    print('iteration:', t)

    with tf.GradientTape() as tape:
        # Forward pass: render the image.
        
        # Important to use a different seed every iteration, otherwise the result
        # would be biased.
        img = pyredner.render(t, *scene_args)
        loss_value = loss(img, target)

    print(f"loss_value: {loss_value}")    
    pyredner.imwrite(img, 'results/test_single_triangle/iter_{}.png'.format(t))

    grads = tape.gradient(loss_value, [shape_triangle.vertices])
    optimizer.apply_gradients(
        zip(grads, [shape_triangle.vertices])
        )

    print('grad:', grads[0])
    print('vertices:', shape_triangle.vertices)

# Render the final result.
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
img = pyredner.render(202, *scene_args)
# Save the images and differences.
pyredner.imwrite(img, 'results/test_single_triangle/final.exr')
pyredner.imwrite(img, 'results/test_single_triangle/final.png')
pyredner.imwrite(tf.abs(target - img), 'results/test_single_triangle/final_diff.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_single_triangle/iter_%d.png", "-vb", "20M",
    "results/test_single_triangle/out.mp4"])
