# Tensorflow by default allocates all GPU memory, leaving very little for rendering.
# We set the environment variable TF_FORCE_GPU_ALLOW_GROWTH to true to enforce on demand
# memory allocation to reduce page faults.
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pyredner_tensorflow as pyredner
import numpy as np

# Optimize texels of a textured patch

# Perlin noise code taken from Stackoverflow
# https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy
def perlin(x, y, seed=0):
    # permutation table
    np.random.seed(seed)
    p = np.arange(256, dtype=np.int32)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()
    # coordinates of the top-left
    xi = x.astype(np.int32)
    yi = y.astype(np.int32)
    # internal coordinates
    xf = (x - xi).astype(np.float32)
    yf = (y - yi).astype(np.float32)
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi]+yi],xf,yf)
    n01 = gradient(p[p[xi]+yi+1],xf,yf-1)
    n11 = gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
    n10 = gradient(p[p[xi+1]+yi],xf-1,yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    return lerp(x1, x2, v)

def lerp(a,b,x):
    return a + x * (b-a)

def fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h,x,y):
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]], dtype=np.float32)
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y

lin = np.linspace(0, 5, 256, endpoint=False, dtype=np.float32)
x, y = np.meshgrid(lin, lin)
diffuse = perlin(x, y, seed=0)
diffuse = (diffuse - np.min(diffuse) + 1e-3) / (np.max(diffuse) - np.min(diffuse))
diffuse = tf.convert_to_tensor(np.tile(np.reshape(diffuse, (256, 256, 1)), (1, 1, 3)), dtype=tf.float32)
specular = perlin(x, y, seed=1)
specular = (specular - np.min(specular) + 1e-3) / (np.max(specular) - np.min(specular))
specular = tf.convert_to_tensor(np.tile(np.reshape(specular, (256, 256, 1)), (1, 1, 3)), dtype=tf.float32)
roughness = perlin(x, y, seed=2)
roughness = (roughness - np.min(roughness) + 1e-3) / (np.max(roughness) - np.min(roughness))
roughness = tf.convert_to_tensor(np.reshape(roughness, (256, 256, 1)), dtype=tf.float32)

# Use GPU if available
pyredner.set_use_gpu(tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None))

# Set up the scene using Pytorch tensor
with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
    position = tf.Variable([0.0, 0.0, -5.0], dtype=tf.float32)
    look_at = tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32)
    up = tf.Variable([0.0, 1.0, 0.0], dtype=tf.float32)
    fov = tf.Variable([45.0], dtype=tf.float32)
    clip_near = 1e-2
    resolution = (256, 256)
    cam = pyredner.Camera(position = position,
                          look_at = look_at,
                          up = up,
                          fov = fov,
                          clip_near = clip_near,
                          resolution = resolution)

mat_perlin = pyredner.Material(
    diffuse_reflectance = diffuse,
    specular_reflectance = specular,
    roughness = roughness)
with tf.device(pyredner.get_device_name()):
    mat_black = pyredner.Material(
        diffuse_reflectance = tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32))
    materials = [mat_perlin, mat_black]
    vertices = tf.Variable([[-1.5,-1.5,0.0], [-1.5,1.5,0.0], [1.5,-1.5,0.0], [1.5,1.5,0.0]],
        dtype=tf.float32)
    indices = tf.constant([[0, 1, 2], [1, 3, 2]], dtype=tf.int32)
    uvs = tf.Variable([[0.05, 0.05], [0.05, 0.95], [0.95, 0.05], [0.95, 0.95]],
        dtype=tf.float32)
    shape_plane = pyredner.Shape(vertices, indices, 0, uvs)
    light_vertices = tf.Variable([[-1.0,-1.0,-7.0],[1.0,-1.0,-7.0],[-1.0,1.0,-7.0],[1.0,1.0,-7.0]],
        dtype=tf.float32)
    light_indices = tf.constant([[0,1,2],[1,3,2]], dtype=tf.int32)
    shape_light = pyredner.Shape(light_vertices, light_indices, 1)
shapes = [shape_plane, shape_light]
with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
    light_intensity = tf.Variable([20.0, 20.0, 20.0], dtype=tf.float32)
# The first argument is the shape id of the light
light = pyredner.AreaLight(1, light_intensity)
area_lights = [light]
scene = pyredner.Scene(cam, shapes, materials, area_lights)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)

# Render our target
img = pyredner.render(0, *scene_args)
pyredner.imwrite(img, 'results/test_svbrdf/target.exr')
pyredner.imwrite(img, 'results/test_svbrdf/target.png')
target = pyredner.imread('results/test_svbrdf/target.exr')

# Our initial guess is three gray textures 
with tf.device(pyredner.get_device_name()):
    diffuse_tex = tf.Variable(
        tf.ones((256, 256, 3), dtype=np.float32) * 0.5,
        trainable=True)
    specular_tex = tf.Variable(
        tf.ones((256, 256, 3), dtype=np.float32) * 0.5,
        trainable=True)
    roughness_tex = tf.Variable(
        tf.ones((256, 256, 1), dtype=np.float32) * 0.5,
        trainable=True)
mat_perlin.diffuse_reflectance = pyredner.Texture(diffuse_tex)
mat_perlin.specular_reflectance = pyredner.Texture(specular_tex)
mat_perlin.roughness = pyredner.Texture(roughness_tex)
scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
# Render the initial guess
img = pyredner.render(1, *scene_args)
pyredner.imwrite(img, 'results/test_svbrdf/init.png')
diff = tf.abs(target - img)
pyredner.imwrite(diff, 'results/test_svbrdf/init_diff.png')

# Optimize for triangle vertices
optimizer = tf.compat.v1.train.AdamOptimizer(1e-2)
for t in range(200):
    print('iteration:', t)
    
    with tf.GradientTape() as tape:
        # Forward pass: render the image
        # Need to rerun the mipmap generation for autodiff to flow through
        mat_perlin.diffuse_reflectance = pyredner.Texture(diffuse_tex)
        mat_perlin.specular_reflectance = pyredner.Texture(specular_tex)
        mat_perlin.roughness = pyredner.Texture(roughness_tex)
        scene_args = pyredner.serialize_scene(
            scene = scene,
            num_samples = 4,
            max_bounces = 1)
        img = pyredner.render(t+1, *scene_args)
        pyredner.imwrite(img, 'results/test_svbrdf/iter_{}.png'.format(t))
        loss = tf.reduce_sum(tf.square(img - target))
    print('loss:', loss)

    grads = tape.gradient(loss, [diffuse_tex, specular_tex, roughness_tex])
    optimizer.apply_gradients(zip(grads, [diffuse_tex, specular_tex, roughness_tex]))
    # Project textures back to valid range
    diffuse_tex.assign(tf.maximum(diffuse_tex, 0))
    specular_tex.assign(tf.maximum(specular_tex, 0))
    roughness_tex.assign(tf.maximum(roughness_tex, 1e-5))

scene_args = pyredner.serialize_scene(
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
img = pyredner.render(202, *scene_args)
pyredner.imwrite(img, 'results/test_svbrdf/final.exr')
pyredner.imwrite(img, 'results/test_svbrdf/final.png')
pyredner.imwrite(tf.abs(target - img), 'results/test_svbrdf/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_svbrdf/iter_%d.png", "-vb", "20M",
    "results/test_svbrdf/out.mp4"])
