import pyredner
import numpy as np
import torch
import scipy.ndimage.filters
import urllib.request
import os
import zipfile
from shutil import copyfile

if not os.path.isdir('scenes/living-room-3'):
    print('Scene file not found, downloading')
    filedata = urllib.request.urlretrieve('https://benedikt-bitterli.me/resources/mitsuba/living-room-3.zip', 'living-room-3.zip')
    print('Unzipping living-room-3.zip')
    zip_ref = zipfile.ZipFile('living-room-3.zip', 'r')
    zip_ref.extractall('scenes/')
    print('Copying scene file')
    copyfile('scenes/living-room-3-scene.xml', 'scenes/living-room-3/scene.xml')
    print('Removing zip file')
    os.remove('living-room-3.zip')

# Optimize for camera pose

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

# Load the scene from a Mitsuba scene file
scene = pyredner.load_mitsuba('scenes/living-room-3/scene.xml')
print('scene loaded')

max_bounces = 6
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = max_bounces)

render = pyredner.RenderFunction.apply
# Render our target. The first argument is the seed for RNG in the renderer.
img = render(0, *args)
pyredner.imwrite(img.cpu(), 'results/test_living_room/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_living_room/target.png')
target = pyredner.imread('results/test_living_room/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

scene.camera.look_at = torch.tensor([-0.556408, 0.951295, -3.98066], requires_grad=True)
scene.camera.position = torch.tensor([0.00419251, 0.973707, -4.80844], requires_grad=True)
scene.camera.up = torch.tensor([-0.00920347, 0.999741, 0.020835], requires_grad=True)

args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = max_bounces)

img = render(1, *args)
pyredner.imwrite(img.cpu(), 'results/test_living_room/init.exr')
pyredner.imwrite(img.cpu(), 'results/test_living_room/init.png')
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/test_living_room/init_diff.png')

optimizer = torch.optim.Adam([scene.camera.position, scene.camera.look_at, scene.camera.up],
    lr = 5e-3, betas=(0.5, 0.9))
iter_count = 0
for t in range(600):
    print('iteration:', t)
    optimizer.zero_grad()
    scene.camera = pyredner.Camera(position   = scene.camera.position,
                                   look_at    = scene.camera.look_at,
                                   up         = scene.camera.up,
                                   fov        = scene.camera.fov,
                                   clip_near  = scene.camera.clip_near,
                                   resolution = scene.camera.resolution)
    args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4,
        max_bounces = max_bounces)

    img = render(t + 1, *args)
    # Save the intermediate render.
    pyredner.imwrite(img.cpu(), 'results/test_living_room/iter_{}.png'.format(t))

    diff = img - target
    dirac = np.zeros([7,7], dtype = np.float32)
    dirac[3,3] = 1.0
    dirac = torch.from_numpy(dirac)
    f = np.zeros([3, 3, 7, 7], dtype = np.float32)
    gf = scipy.ndimage.filters.gaussian_filter(dirac, 1.0)
    f[0, 0, :, :] = gf
    f[1, 1, :, :] = gf
    f[2, 2, :, :] = gf
    f = torch.from_numpy(f)
    if pyredner.get_use_gpu():
        f = f.cuda(device = pyredner.get_device())
    m = torch.nn.AvgPool2d(2)

    r = 256
    diff_0 = (img - target).view(1, r, r, 3).permute(0, 3, 2, 1)
    diff_1 = m(torch.nn.functional.conv2d(diff_0, f, padding=3))
    diff_2 = m(torch.nn.functional.conv2d(diff_1, f, padding=3))
    diff_3 = m(torch.nn.functional.conv2d(diff_2, f, padding=3))
    diff_4 = m(torch.nn.functional.conv2d(diff_3, f, padding=3))
    loss = diff_0.pow(2).sum() / (r*r) + \
           diff_1.pow(2).sum() / ((r/2)*(r/2)) + \
           diff_2.pow(2).sum() / ((r/4)*(r/4)) + \
           diff_3.pow(2).sum() / ((r/8)*(r/8)) + \
           diff_4.pow(2).sum() / ((r/16)*(r/16))
    print('loss:', loss.item())
    print('cam.look_at:', scene.camera.look_at)
    print('cam.position:', scene.camera.position)
    print('cam.up:', scene.camera.up)

    loss.backward()
    optimizer.step()

    print('cam.look_at.grad:', scene.camera.look_at.grad)
    print('cam.position.grad:', scene.camera.position.grad)
    print('cam.up.grad:', scene.camera.up.grad)

args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 4,
    max_bounces = max_bounces)
img = render(602, *args)
pyredner.imwrite(img.cpu(), 'results/test_living_room/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_living_room/final.png')
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/test_living_room/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_living_room/iter_%d.png", "-vb", "20M", "results/test_living_room/out.mp4"])
