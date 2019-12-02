# Tensorflow by default allocates all GPU memory, leaving very little for rendering.
# We set the environment variable TF_FORCE_GPU_ALLOW_GROWTH to true to enforce on demand
# memory allocation to reduce page faults.
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pyredner_tensorflow as pyredner
import numpy as np
import scipy

# Optimize for material parameters and camera pose

# Use GPU if available
pyredner.set_use_gpu(tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None))

# Load the scene from a Mitsuba scene file
scene = pyredner.load_mitsuba('scenes/teapot.xml')

# The last material is the teapot material, set it to the target
with tf.device(pyredner.get_device_name()):
    scene.materials[-1].diffuse_reflectance = \
        pyredner.Texture(tf.Variable([0.3, 0.2, 0.2], dtype=tf.float32))
    scene.materials[-1].specular_reflectance = \
        pyredner.Texture(tf.Variable([0.6, 0.6, 0.6], dtype=tf.float32))
    scene.materials[-1].roughness = \
        pyredner.Texture(tf.Variable([0.05], dtype=tf.float32))

scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 1024,
    max_bounces = 2)

# Render our target. The first argument is the seed for RNG in the renderer.
img = pyredner.render(0, *scene_args)
pyredner.imwrite(img, 'results/test_teapot_reflectance/target.exr')
pyredner.imwrite(img, 'results/test_teapot_reflectance/target.png')
target = pyredner.imread('results/test_teapot_reflectance/target.exr')

# Perturb the scene, this is our initial guess
cam = scene.camera
cam_position = cam.position
with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
    cam_translation = tf.Variable([-0.2, 0.2, -0.2], dtype=tf.float32, trainable=True)
with tf.device(pyredner.get_device_name()):
    diffuse_reflectance = tf.Variable([0.3, 0.3, 0.3], dtype=tf.float32, trainable=True)
    specular_reflectance = tf.Variable([0.5, 0.5, 0.5], dtype=tf.float32, trainable=True)
    roughness = tf.Variable([0.2], dtype=tf.float32, trainable=True)
scene.materials[-1].diffuse_reflectance = pyredner.Texture(diffuse_reflectance)
scene.materials[-1].specular_reflectance = pyredner.Texture(specular_reflectance)
scene.materials[-1].roughness = pyredner.Texture(roughness)
scene.camera = pyredner.Camera(position     = cam_position + cam_translation,
                               look_at      = cam.look_at + cam_translation,
                               up           = cam.up,
                               fov          = cam.fov,
                               clip_near    = cam.clip_near,
                               resolution   = cam.resolution,
                               fisheye      = False)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 1024,
    max_bounces = 2)
# Render the initial guess
img = pyredner.render(1, *scene_args)
pyredner.imwrite(img, 'results/test_teapot_reflectance/init.png')
diff = tf.abs(target - img)
pyredner.imwrite(diff, 'results/test_teapot_reflectance/init_diff.png')

lr = 1e-2
optimizer = tf.compat.v1.train.AdamOptimizer(lr)
num_iteration = 200
for t in range(num_iteration):
    print('iteration:', t)
    
    with tf.GradientTape() as tape:
        # Forward pass: render the image
        # need to rerun Camera constructor for autodiff 
        scene.camera = pyredner.Camera(position     = cam_position + cam_translation,
                                       look_at      = cam.look_at + cam_translation,
                                       up           = cam.up,
                                       fov          = cam.fov,
                                       clip_near    = cam.clip_near,
                                       resolution   = cam.resolution,
                                       fisheye      = False)
        # need to rerun the reflectance for autodiff
        scene.materials[-1].diffuse_reflectance = pyredner.Texture(diffuse_reflectance)
        scene.materials[-1].specular_reflectance = pyredner.Texture(specular_reflectance)
        scene.materials[-1].roughness = pyredner.Texture(roughness)
        scene_args = pyredner.serialize_scene(
            scene = scene,
            num_samples = 4,
            max_bounces = 2)
        img = pyredner.render(t+1, *scene_args)
        pyredner.imwrite(img, 'results/test_teapot_reflectance/iter_{}.png'.format(t))

        #loss = tf.reduce_sum(tf.square(img - target))
        with tf.device(pyredner.get_device_name()):
            dirac = np.zeros([7,7], dtype = np.float32)
            dirac[3,3] = 1.0
            f = np.zeros([7, 7, 3, 3], dtype = np.float32)
            gf = scipy.ndimage.filters.gaussian_filter(dirac, 1.0)
            f[:, :, 0, 0] = gf
            f[:, :, 1, 1] = gf
            f[:, :, 2, 2] = gf
            f = tf.constant(f)
            def conv(x, f):
                '''
                    m = torch.nn.AvgPool2d(2)
                    m(torch.nn.functional.conv2d(diff_0, f, padding=3))
    
                Torch uses NCHW
                TF uses NHWC
                '''
                padding = [
                    [0,0], # Minibatch
                    [3,3],
                    [3,3],
                    [0,0]
                ]
                y = tf.nn.conv2d(x, f, strides=1, padding=padding)
                y = tf.nn.avg_pool2d(y, ksize=2, strides=2, padding='VALID')
                return y
    
            r = 256
            # NOTE: `perm` must change according to the `data_format`
            diff_0 = tf.transpose(tf.reshape(img - target, (1, r, r, 3)), perm=[0,2,1,3])
            diff_1 = conv(diff_0, f)
            diff_2 = conv(diff_1, f)
            diff_3 = conv(diff_2, f)
            diff_4 = conv(diff_3, f)
            diff_5 = conv(diff_4, f)
            
            loss = tf.reduce_sum(tf.square(diff_0)) / (r*r) + \
                   tf.reduce_sum(tf.square(diff_1)) / ((r/2)*(r/2)) + \
                   tf.reduce_sum(tf.square(diff_2)) / ((r/4)*(r/4)) + \
                   tf.reduce_sum(tf.square(diff_3)) / ((r/8)*(r/8)) + \
                   tf.reduce_sum(tf.square(diff_4)) / ((r/16)*(r/16)) + \
                   tf.reduce_sum(tf.square(diff_5)) / ((r/32)*(r/32))
    print('>>> LOSS:', loss)

    grads = tape.gradient(
        loss, 
        [diffuse_reflectance, specular_reflectance, roughness, cam_translation]
    )

    print(grads)

    print('diffuse_reflectance.grad:', grads[0].numpy())
    print('specular_reflectance.grad:', grads[1].numpy())
    print('roughness.grad:', grads[2].numpy())
    print('cam_translation.grad:', grads[3].numpy())
    grads[2] = tf.clip_by_norm(grads[2], 10)
    grads[3] = tf.clip_by_norm(grads[3], 10)

    optimizer.apply_gradients(
        zip(grads, [diffuse_reflectance, specular_reflectance, roughness, cam_translation]))
    print(">>> AFTER CLIPPING")
    print('diffuse_reflectance.grad:', grads[0].numpy())
    print('specular_reflectance.grad:', grads[1].numpy())
    print('roughness.grad:', grads[2].numpy())
    print('cam_translation.grad:', grads[3].numpy())
    
    print('diffuse_reflectance:', diffuse_reflectance.numpy())
    print('specular_reflectance:', specular_reflectance.numpy())
    print('roughness:', roughness.numpy())
    print('cam_translation:', cam_translation.numpy())

    # Linearly reduce the learning rate
    # lr = 1e-2 * float(num_iteration - t) / float(num_iteration)
    # optimizer._lr = lr

scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 1024,
    max_bounces = 2)
img = pyredner.render(num_iteration + 2, *scene_args)
pyredner.imwrite(img, 'results/test_teapot_reflectance/final.exr')
pyredner.imwrite(img, 'results/test_teapot_reflectance/final.png')
pyredner.imwrite(tf.abs(target - img), 'results/test_teapot_reflectance/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_teapot_reflectance/iter_%d.png", "-vb", "20M",
    "results/test_teapot_reflectance/out.mp4"])
