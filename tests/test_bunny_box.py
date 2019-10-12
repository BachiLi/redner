import scipy.ndimage.filters
import pyredner
import numpy as np
import torch

pyredner.set_use_gpu(torch.cuda.is_available())

scene = pyredner.load_mitsuba('scenes/bunny_box.xml')

scene.shapes[-1].vertices += torch.tensor([0, 0.01, 0], device = pyredner.get_device())

args=pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 4,
    max_bounces = 6)
render = pyredner.RenderFunction.apply
# Render our target. The first argument is the seed for RNG in the renderer.
img = render(0, *args)
pyredner.imwrite(img.cpu(), 'results/test_bunny_box/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_bunny_box/target.png')
target = pyredner.imread('results/test_bunny_box/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

bunny_vertices = scene.shapes[-1].vertices.clone()
bunny_translation = torch.tensor([0.1,0.4,0.1], device = pyredner.get_device(), requires_grad=True)
bunny_rotation = torch.tensor([-0.2,0.1,-0.1], device = pyredner.get_device(), requires_grad=True)
bunny_rotation_matrix = pyredner.gen_rotate_matrix(bunny_rotation)

scene.shapes[-1].vertices = \
    (bunny_vertices-torch.mean(bunny_vertices, 0))@torch.t(bunny_rotation_matrix) + \
    torch.mean(bunny_vertices, 0) + bunny_translation
args=pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4,
        max_bounces = 6)
img = render(1, *args)
pyredner.imwrite(img.cpu(), 'results/test_bunny_box/init.exr')
pyredner.imwrite(img.cpu(), 'results/test_bunny_box/init.png')

optimizer = torch.optim.Adam([bunny_translation, bunny_rotation], lr=1e-2)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image
    bunny_rotation_matrix = pyredner.gen_rotate_matrix(bunny_rotation)

    scene.shapes[-1].vertices = \
        (bunny_vertices-torch.mean(bunny_vertices, 0))@torch.t(bunny_rotation_matrix) + \
        torch.mean(bunny_vertices, 0) + bunny_translation
    args=pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4,
        max_bounces = 6)
    img = render(t+1, *args)
    pyredner.imwrite(img.cpu(), 'results/test_bunny_box/iter_{}.png'.format(t))

    dirac = np.zeros([7,7], dtype=np.float32)
    dirac[3,3] = 1.0
    dirac = torch.from_numpy(dirac)
    f = np.zeros([3, 3, 7, 7], dtype=np.float32)
    gf = scipy.ndimage.filters.gaussian_filter(dirac, 1.0)
    f[0, 0, :, :] = gf
    f[1, 1, :, :] = gf
    f[2, 2, :, :] = gf
    f = torch.from_numpy(f)
    if pyredner.get_use_gpu():
        f = f.cuda(device = pyredner.get_device())
    m = torch.nn.AvgPool2d(2)

    res = 256
    diff_0 = (img - target).view(1, res, res, 3).permute(0, 3, 2, 1)
    diff_1 = m(torch.nn.functional.conv2d(diff_0, f, padding=3)) # 128 x 128
    diff_2 = m(torch.nn.functional.conv2d(diff_1, f, padding=3)) # 64 x 64
    diff_3 = m(torch.nn.functional.conv2d(diff_2, f, padding=3)) # 32 x 32
    diff_4 = m(torch.nn.functional.conv2d(diff_3, f, padding=3)) # 16 x 16
    loss = diff_0.pow(2).sum() / (res*res) + \
           diff_1.pow(2).sum() / ((res/2)*(res/2)) + \
           diff_2.pow(2).sum() / ((res/4)*(res/4)) + \
           diff_3.pow(2).sum() / ((res/8)*(res/8)) + \
           diff_4.pow(2).sum() / ((res/16)*(res/16))
    print('loss:', loss.item())

    loss.backward()
    print('bunny_translation.grad:', bunny_translation.grad)
    print('bunny_rotation.grad:', bunny_rotation.grad)

    optimizer.step()
    print('bunny_translation:', bunny_translation)
    print('bunny_rotation:', bunny_rotation)

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/bunny_box/iter_%d.png", "-vb", "20M",
    "results/bunny_box/out.mp4"])
