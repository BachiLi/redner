from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pyredner
import numpy as np
import torch

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
diffuse = torch.from_numpy(np.tile(np.reshape(diffuse, (256, 256, 1)), (1, 1, 3)))
specular = perlin(x, y, seed=1)
specular = (specular - np.min(specular) + 1e-3) / (np.max(specular) - np.min(specular))
specular = torch.from_numpy(np.tile(np.reshape(specular, (256, 256, 1)), (1, 1, 3)))
roughness = perlin(x, y, seed=2)
roughness = (roughness - np.min(roughness) + 1e-3) / (np.max(roughness) - np.min(roughness))
roughness = torch.from_numpy(np.reshape(roughness, (256, 256, 1)))

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

# Set up the scene using Pytorch tensor
position = torch.tensor([0.0, 0.0, -5.0])
look_at = torch.tensor([0.0, 0.0, 0.0])
up = torch.tensor([0.0, 1.0, 0.0])
fov = torch.tensor([45.0])
clip_near = 1e-2

resolution = (256, 256)
cam = pyredner.Camera(position = position,
                     look_at = look_at,
                     up = up,
                     fov = fov,
                     clip_near = clip_near,
                     resolution = resolution)
if pyredner.get_use_gpu():
    diffuse = diffuse.cuda(device = pyredner.get_device())
    specular = specular.cuda(device = pyredner.get_device())
    roughness = roughness.cuda(device = pyredner.get_device())
mat_perlin = pyredner.Material(\
    diffuse_reflectance = diffuse,
    specular_reflectance = specular,
    roughness = roughness)
mat_black = pyredner.Material(\
    diffuse_reflectance = torch.tensor([0.0, 0.0, 0.0], device = pyredner.get_device()))
materials = [mat_perlin, mat_black]
vertices = torch.tensor([[-1.5,-1.5,0.0], [-1.5,1.5,0.0], [1.5,-1.5,0.0], [1.5,1.5,0.0]],
                        device = pyredner.get_device())
indices = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype = torch.int32,
                       device = pyredner.get_device())
uvs = torch.tensor([[0.05, 0.05], [0.05, 0.95], [0.95, 0.05], [0.95, 0.95]],
				   device = pyredner.get_device())
shape_plane = pyredner.Shape(vertices = vertices,
                             indices = indices,
                             uvs = uvs,
                             material_id = 0)
light_vertices = torch.tensor([[-1.0,-1.0,-7.0],[1.0,-1.0,-7.0],[-1.0,1.0,-7.0],[1.0,1.0,-7.0]],
                              device = pyredner.get_device())
light_indices = torch.tensor([[0,1,2],[1,3,2]], dtype = torch.int32, device = pyredner.get_device())
shape_light = pyredner.Shape(light_vertices, light_indices, 1)
shapes = [shape_plane, shape_light]
light_intensity = torch.tensor([20.0, 20.0, 20.0])
# The first argument is the shape id of the light
light = pyredner.AreaLight(1, light_intensity)
area_lights = [light]
scene = pyredner.Scene(cam, shapes, materials, area_lights)
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)

# Alias of the render function
render = pyredner.RenderFunction.apply
# Render our target
img = render(0, *args)
pyredner.imwrite(img.cpu(), 'results/test_svbrdf/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_svbrdf/target.png')
target = pyredner.imread('results/test_svbrdf/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

# Our initial guess is three gray textures 
diffuse_tex = torch.tensor(\
    np.ones((256, 256, 3), dtype=np.float32) * 0.5,
    requires_grad = True,
    device = pyredner.get_device())
specular_tex = torch.tensor(\
    np.ones((256, 256, 3), dtype=np.float32) * 0.5,
    requires_grad = True,
    device = pyredner.get_device())
roughness_tex = torch.tensor(\
    np.ones((256, 256, 1), dtype=np.float32) * 0.5,
    requires_grad = True,
    device = pyredner.get_device())
mat_perlin.diffuse_reflectance = pyredner.Texture(diffuse_tex)
mat_perlin.specular_reflectance = pyredner.Texture(specular_tex)
mat_perlin.roughness = pyredner.Texture(roughness_tex)
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
# Render the initial guess
img = render(1, *args)
pyredner.imwrite(img.cpu(), 'results/test_svbrdf/init.png')
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/test_svbrdf/init_diff.png')

# Optimize for triangle vertices
optimizer = torch.optim.Adam([diffuse_tex, specular_tex, roughness_tex], lr=1e-2)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image
    # Need to rerun the mipmap generation for autodiff to flow through
    mat_perlin.diffuse_reflectance = pyredner.Texture(diffuse_tex)
    mat_perlin.specular_reflectance = pyredner.Texture(specular_tex)
    mat_perlin.roughness = pyredner.Texture(roughness_tex)
    args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4,
        max_bounces = 1)
    img = render(t+1, *args)
    pyredner.imwrite(img.cpu(), 'results/test_svbrdf/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    loss.backward()
    optimizer.step()
    diffuse_tex.data = diffuse_tex.data.clamp(0, 1)
    specular_tex.data = specular_tex.data.clamp(0, 1)
    roughness_tex.data = roughness_tex.data.clamp(1e-5, 1)

args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1)
img = render(202, *args)
pyredner.imwrite(img.cpu(), 'results/test_svbrdf/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_svbrdf/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/test_svbrdf/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_svbrdf/iter_%d.png", "-vb", "20M",
    "results/test_svbrdf/out.mp4"])
