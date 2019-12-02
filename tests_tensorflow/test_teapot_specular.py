# Tensorflow by default allocates all GPU memory, leaving very little for rendering.
# We set the environment variable TF_FORCE_GPU_ALLOW_GROWTH to true to enforce on demand
# memory allocation to reduce page faults.
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pyredner_tensorflow as pyredner

# Optimize for a textured plane in a specular reflection

# Use GPU if available
pyredner.set_use_gpu(tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None))

# Load the scene from a Mitsuba scene file
scene = pyredner.load_mitsuba('scenes/teapot_specular.xml')

# The last material is the teapot material, set it to a specular material
with tf.device(pyredner.get_device_name()):
    scene.materials[-1].diffuse_reflectance = \
        pyredner.Texture(tf.Variable([0.15, 0.2, 0.15], dtype=tf.float32))
    scene.materials[-1].specular_reflectance = \
        pyredner.Texture(tf.Variable([0.8, 0.8, 0.8], dtype=tf.float32))
    scene.materials[-1].roughness = \
        pyredner.Texture(tf.Variable([0.0001], dtype=tf.float32))

scene_args=pyredner.serialize_scene(
    scene = scene,
    num_samples = 512,
    max_bounces = 2)

# Render our target. The first argument is the seed for RNG in the renderer.
img = pyredner.render(0, *scene_args)
pyredner.imwrite(img, 'results/test_teapot_specular/target.exr')
pyredner.imwrite(img, 'results/test_teapot_specular/target.png')
target = pyredner.imread('results/test_teapot_specular/target.exr')

# Perturb the scene, this is our initial guess
# We perturb the last shape, which is the SIGGRAPH logo
ref_pos = scene.shapes[-1].vertices
with tf.device(pyredner.get_device_name()):
    translation = tf.Variable([20.0, 0.0, 2.0], trainable=True)
    scene.shapes[-1].vertices = ref_pos + translation
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 512,
    max_bounces = 2)
# Render the initial guess
img = pyredner.render(1, *scene_args)
pyredner.imwrite(img, 'results/test_teapot_specular/init.png')
diff = tf.abs(target - img)
pyredner.imwrite(diff, 'results/test_teapot_specular/init_diff.png')

optimizer = tf.compat.v1.train.AdamOptimizer(0.5)
num_iteration = 400
for t in range(num_iteration):
    print('iteration:', t)
    
    with tf.GradientTape() as tape:
        scene.shapes[-1].vertices = ref_pos + translation
        scene_args = pyredner.serialize_scene(
            scene = scene,
            num_samples = 4,
            max_bounces = 2)
        img = pyredner.render(t+1, *scene_args)
        pyredner.imwrite(img, 'results/test_teapot_specular/iter_{}.png'.format(t))
        loss = tf.reduce_sum(tf.square(img - target))
    print('loss:', loss)

    grads = tape.gradient(loss, [translation])
    print('grad:', grads)

    optimizer.apply_gradients(zip(grads, [translation]))

    print('translation:', translation)

    # Linearly reduce the learning rate
    lr = 0.5 * float(num_iteration - t) / float(num_iteration)
    optimizer._lr = lr

scene.shapes[-1].vertices = ref_pos + translation
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 512,
    max_bounces = 2)
img = pyredner.render(num_iteration + 2, *scene_args)
pyredner.imwrite(img, 'results/test_teapot_specular/final.exr')
pyredner.imwrite(img, 'results/test_teapot_specular/final.png')
pyredner.imwrite(tf.abs(target - img), 'results/test_teapot_specular/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_teapot_specular/iter_%d.png", "-vb", "20M",
    "results/test_teapot_specular/out.mp4"])
