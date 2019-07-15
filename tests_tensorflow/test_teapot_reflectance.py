import numpy as np
import scipy.ndimage as ndimage
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tfe = tf.contrib.eager
import pyrednertensorflow as pyredner


import pdb

# Optimize for material parameters and camera pose

# Use GPU if available
pyredner.set_use_gpu(False)

# Load the scene from a Mitsuba scene file
scene = pyredner.load_mitsuba('scenes/teapot.xml')

# The last material is the teapot material, set it to the target
scene.materials[-1].diffuse_reflectance = \
    pyredner.Texture(tfe.Variable([0.3, 0.2, 0.2], ))
scene.materials[-1].specular_reflectance = \
    pyredner.Texture(tfe.Variable([0.6, 0.6, 0.6], ))
scene.materials[-1].roughness = \
    pyredner.Texture(tfe.Variable([0.05], ))
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
cam_translation = tfe.Variable([-0.2, 0.2, -0.2], trainable=True)
diffuse_reflectance = tfe.Variable([0.3, 0.3, 0.3],
    trainable=True)
specular_reflectance = tfe.Variable([0.5, 0.5, 0.5],
    trainable=True)
roughness = tfe.Variable([0.2],
    trainable=True)
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
optimizer = tf.train.AdamOptimizer(lr)
# optimizer = torch.optim.Adam([diffuse_reflectance,
#                               specular_reflectance,
#                               roughness,
#                               cam_translation], lr=lr)
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
        scene_args = pyredner.serialize_scene(
            scene = scene,
            num_samples = 4,
            max_bounces = 2)
        img = pyredner.render(t+1, *scene_args)
        pyredner.imwrite(img, 'results/test_teapot_reflectance/iter_{}.png'.format(t))

        # NOTE: Loss in reflectance is bit special!
        diff = img - target
        # dirac = np.zeros([7,7], dtype = np.float32)
        dirac = tf.constant(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1.0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ],
            dtype=tf.float32
        )
        gf = ndimage.filters.gaussian_filter(dirac, 1.0)
        
        f = tf.concat([
            tf.reshape(
                tf.concat(
                    [tf.reshape(gf, [1, 7,7]),
                    tf.constant(np.zeros([1,7,7], dtype=np.float32), dtype=tf.float32),
                    tf.constant(np.zeros([1,7,7], dtype=np.float32), dtype=tf.float32)],
                    axis=0
                ), 
                [1,3,7,7]
            ),
            tf.reshape(
                tf.concat([
                    tf.constant(np.zeros([1,7,7], dtype=np.float32), dtype=tf.float32),
                    tf.reshape(gf, [1, 7,7]),
                    tf.constant(np.zeros([1,7,7], dtype=np.float32), dtype=tf.float32)],
                    axis=0
                ), 
                [1,3,7,7]
            ),
            tf.reshape(
                tf.concat([
                    tf.constant(np.zeros([1,7,7], dtype=np.float32), dtype=tf.float32),
                    tf.constant(np.zeros([1,7,7], dtype=np.float32), dtype=tf.float32),
                    tf.reshape(gf, [1, 7,7])],
                    axis=0
                ), 
                [1,3,7,7
            ])
            ],
            axis=0
        )
        # padding = [[0,0],[3,3],[3,3],[0,0]]
        # tf.nn.conv2d(diff_0, ff, padding=padding, data_format='NHWC')
        def conv(x, filter):
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
            # y = tf.nn.conv2d(x, filter, padding=padding, data_format='NCHW')
            y = tf.nn.conv2d(x, filter, padding=padding, data_format='NHWC')
            y = tf.nn.avg_pool2d(y, ksize=2, strides=2, padding='VALID', data_format='NHWC')
            return y

        r = 256
        # NOTE: `perm` must change according to the `data_format`
        diff_0 = tf.transpose(tf.reshape(img - target, (1, r, r, 3)), perm=[0,2,1,3])  
        f = tf.transpose(f, perm=[2,3,0,1])
        # pdb.set_trace()
        diff_1 = conv(diff_0, f)
        diff_2 = conv(diff_1, f)
        diff_3 = conv(diff_2, f)
        diff_4 = conv(diff_3, f)
        diff_5 = conv(diff_4, f)
        
        loss = tf.reduce_sum(tf.pow(diff_0, 2)) / (r*r) + \
            tf.reduce_sum(tf.pow(diff_1, 2)) / ((r/2)*(r/2)) + \
            tf.reduce_sum(tf.pow(diff_2, 2)) / ((r/4)*(r/4)) + \
            tf.reduce_sum(tf.pow(diff_3, 2)) / ((r/8)*(r/8)) + \
            tf.reduce_sum(tf.pow(diff_4, 2)) / ((r/16)*(r/16)) + \
            tf.reduce_sum(tf.pow(diff_5, 2)) / ((r/32)*(r/32))


    print('>>> LOSS:', loss)

    grads = tape.gradient(
        loss, 
        [diffuse_reflectance, specular_reflectance, roughness, cam_translation]
    )

    print('diffuse_reflectance.grad:', grads[0].numpy())
    print('specular_reflectance.grad:', grads[1].numpy())
    print('roughness.grad:', grads[2].numpy())
    print('cam_translation.grad:', grads[3].numpy())
    # pdb.set_trace()
    # grads, _ = tf.clip_by_global_norm(grads, 10.0)
    grads[2] = tf.clip_by_norm(grads[2], 10)
    grads[3] = tf.clip_by_norm(grads[3], 10)

    optimizer.apply_gradients(
        zip(
            grads, 
            [diffuse_reflectance, specular_reflectance, roughness, cam_translation])
    )
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
pyredner.imwrite(tf.abs(target - img).cpu(), 'results/test_teapot_reflectance/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_teapot_reflectance/iter_%d.png", "-vb", "20M",
    "results/test_teapot_reflectance/out.mp4"])
