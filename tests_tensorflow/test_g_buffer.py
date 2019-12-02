# Tensorflow by default allocates all GPU memory, leaving very little for rendering.
# We set the environment variable TF_FORCE_GPU_ALLOW_GROWTH to true to enforce on demand
# memory allocation to reduce page faults.
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pyredner_tensorflow as pyredner
import redner

# Optimize depth and normal of a teapot

# Use GPU if available
pyredner.set_use_gpu(tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None))

# Set up the pyredner scene for rendering:
with tf.device(pyredner.get_device_name()):
    material_map, mesh_list, light_map = pyredner.load_obj('scenes/teapot.obj')
    for _, mesh in mesh_list:
        mesh.normals = pyredner.compute_vertex_normal(mesh.vertices, mesh.indices)

# Setup camera
with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
    cam = pyredner.Camera(position = tf.Variable([0.0, 30.0, 200.0], dtype=tf.float32),
                          look_at = tf.Variable([0.0, 30.0, 0.0], dtype=tf.float32),
                          up = tf.Variable([0.0, 1.0, 0.0], dtype=tf.float32),
                          fov = tf.Variable([45.0], dtype=tf.float32), # in degree
                          clip_near = 1e-2, # needs to > 0
                          resolution = (256, 256),
                          fisheye = False)

# Setup materials
material_id_map = {}
materials = []
count = 0
for key, value in material_map.items():
    material_id_map[key] = count
    count += 1
    materials.append(value)

# Setup geometries
shapes = []
with tf.device(pyredner.get_device_name()):
    for mtl_name, mesh in mesh_list:
        shapes.append(pyredner.Shape(
            vertices = mesh.vertices,
            indices = mesh.indices,
            uvs = mesh.uvs,
            normals = mesh.normals,
            material_id = material_id_map[mtl_name]))

# We don't setup any light source here

# Construct the scene
scene = pyredner.Scene(cam, shapes, materials, area_lights = [], envmap = None)
# Serialize the scene
# Here we specify the output channels as "depth", "shading_normal"
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 0,
    channels = [redner.channels.depth, redner.channels.shading_normal])

# Render. The first argument is the seed for RNG in the renderer.
img = pyredner.render(0, *scene_args)
# Save the images.
depth = img[:, :, 0]
normal = img[:, :, 1:4]
pyredner.imwrite(depth, 'results/test_g_buffer/target_depth.exr')
pyredner.imwrite(depth, 'results/test_g_buffer/target_depth.png', normalize = True)
pyredner.imwrite(normal, 'results/test_g_buffer/target_normal.exr')
pyredner.imwrite(normal, 'results/test_g_buffer/target_normal.png', normalize = True)
# Read the target image we just saved.
target_depth = pyredner.imread('results/test_g_buffer/target_depth.exr')
target_depth = target_depth[:, :, 0]
target_normal = pyredner.imread('results/test_g_buffer/target_normal.exr')

with tf.device(pyredner.get_device_name()):
    # Perturb the teapot by a translation and a rotation to the object
    translation_params = tf.Variable([0.1, -0.1, 0.1], trainable=True)
    translation = translation_params * 100.0
    euler_angles = tf.Variable([0.1, -0.1, 0.1], trainable=True)

    # These are the vertices we want to apply the transformation
    shape0_vertices = tf.identity(shapes[0].vertices)
    shape1_vertices = tf.identity(shapes[1].vertices)
    # We can use pyredner.gen_rotate_matrix to generate 3x3 rotation matrices
    rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
    center = tf.math.reduce_mean(tf.concat([shape0_vertices, shape1_vertices], axis=0), axis=0)
    # Shift the vertices to the center, apply rotation matrix,
    # shift back to the original space
    shapes[0].vertices = \
        (shape0_vertices - center) @ tf.transpose(rotation_matrix) + \
        center + translation
    shapes[1].vertices = \
        (shape1_vertices - center) @ tf.transpose(rotation_matrix) + \
        center + translation
    # Since we changed the vertices, we need to regenerate the shading normals
    shapes[0].normals = pyredner.compute_vertex_normal(shapes[0].vertices, shapes[0].indices)
    shapes[1].normals = pyredner.compute_vertex_normal(shapes[1].vertices, shapes[1].indices)
# We need to serialize the scene again to get the new arguments.
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 0,
    channels = [redner.channels.depth, redner.channels.shading_normal])
# Render the initial guess.
img = pyredner.render(1, *scene_args)
depth = img[:, :, 0]
normal = img[:, :, 1:4]
# Save the images.
pyredner.imwrite(depth, 'results/test_g_buffer/init_depth.png', normalize = True)
pyredner.imwrite(depth, 'results/test_g_buffer/init_normal.png', normalize = True)
# Compute the difference and save the images.
diff_depth = tf.abs(target_depth - depth)
diff_normal = tf.abs(target_normal - normal)
pyredner.imwrite(diff_depth, 'results/test_g_buffer/init_depth_diff.png')
pyredner.imwrite(diff_normal, 'results/test_g_buffer/init_normal_diff.png')

# Optimize for triangle vertices.
optimizer = tf.compat.v1.train.AdamOptimizer(1e-2)
# Run 200 Adam iterations.
for t in range(200):
    print('iteration:', t)
    # Forward pass: apply the mesh operation and render the image.
    with tf.GradientTape() as tape:
        translation = translation_params * 100.0
        rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
        center = tf.math.reduce_mean(
            tf.concat([shape0_vertices, shape1_vertices], axis=0), 
            axis=0)
        shapes[0].vertices = \
            (shape0_vertices - center) @ tf.transpose(rotation_matrix) + \
            center + translation
        shapes[1].vertices = \
            (shape1_vertices - center) @ tf.transpose(rotation_matrix) + \
            center + translation

        shapes[0].normals = pyredner.compute_vertex_normal(shapes[0].vertices, shapes[0].indices)
        shapes[1].normals = pyredner.compute_vertex_normal(shapes[1].vertices, shapes[1].indices)
        scene_args = pyredner.serialize_scene(
            scene = scene,
            num_samples = 4, # We use less samples in the Adam loop.
            max_bounces = 0,
            channels = [redner.channels.depth, redner.channels.shading_normal])
        # Important to use a different seed every iteration, otherwise the result
        # would be biased.
        img = pyredner.render(t+1, *scene_args)

        depth = img[:, :, 0]
        normal = img[:, :, 1:4]

        # Save the intermediate render.
        pyredner.imwrite(depth, 'results/test_g_buffer/iter_depth_{}.png'.format(t), normalize = True)
        pyredner.imwrite(normal, 'results/test_g_buffer/iter_normal_{}.png'.format(t), normalize = True)
        # Compute the loss function. Here it is L2.
        loss = tf.reduce_sum(tf.square(depth - target_depth)) / 200.0 \
             + tf.reduce_sum(tf.square(normal - target_normal))
        print('loss:', loss)

    grads = tape.gradient(loss, [translation_params, euler_angles])

    optimizer.apply_gradients(zip(grads, [translation_params, euler_angles]))

    # Print the gradients
    print('translation_params.grad:', grads[0])
    print('euler_angles.grad:', grads[0])

    # Take a gradient descent step.
    # Print the current pose parameters.
    print('translation:', translation)
    print('euler_angles:', euler_angles)

# Render the final result.
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 0,
    channels = [redner.channels.depth, redner.channels.shading_normal])
img = pyredner.render(202, *scene_args)
depth = img[:, :, 0]
normal = img[:, :, 1:4]
# Save the images.
pyredner.imwrite(depth, 'results/test_g_buffer/final_depth.exr')
pyredner.imwrite(depth, 'results/test_g_buffer/init_depth.png', normalize = True)
pyredner.imwrite(normal, 'results/test_g_buffer/final_normal.exr')
pyredner.imwrite(normal, 'results/test_g_buffer/final_normal.png', normalize = True)
diff_depth = tf.abs(target_depth - depth)
diff_normal = tf.abs(target_normal - normal)
pyredner.imwrite(diff_depth, 'results/test_g_buffer/init_depth_diff.png')
pyredner.imwrite(diff_normal, 'results/test_g_buffer/init_normal_diff.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_g_buffer/iter_depth_%d.png", "-vb", "20M",
    "results/test_g_buffer/out_depth.mp4"])
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_g_buffer/iter_normal_%d.png", "-vb", "20M",
    "results/test_g_buffer/out_normal.mp4"])
