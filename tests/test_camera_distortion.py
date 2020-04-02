import pyredner
import torch

pyredner.set_use_gpu(torch.cuda.is_available())

position = torch.tensor([1.0, 0.0, -3.0])
look_at = torch.tensor([1.0, 0.0, 0.0])
up = torch.tensor([0.0, 1.0, 0.0])
fov = torch.tensor([45.0])
clip_near = 1e-2

# randomly generate distortion parameters
torch.manual_seed(1234)
target_distort_params = (torch.rand(8) - 0.5) * 0.05
resolution = (256, 256)
cam = pyredner.Camera(position = position,
                      look_at = look_at,
                      up = up,
                      fov = fov,
                      clip_near = clip_near,
                      resolution = resolution,
                      distortion_params = target_distort_params)

checkerboard_texture = pyredner.imread('scenes/teapot.png')
if pyredner.get_use_gpu():
    checkerboard_texture = checkerboard_texture.cuda(device = pyredner.get_device())

mat_checkerboard = pyredner.Material(\
    diffuse_reflectance = checkerboard_texture)
mat_black = pyredner.Material(\
    diffuse_reflectance = torch.tensor([0.0, 0.0, 0.0], device = pyredner.get_device()))

plane = pyredner.Object(vertices = torch.tensor([[-1.0,-1.0, 0.0],
                                                 [-1.0, 1.0, 0.0],
                                                 [ 1.0,-1.0, 0.0],
                                                 [ 1.0, 1.0, 0.0]],
                                                 device = pyredner.get_device()),
                        indices = torch.tensor([[0, 1, 2],
                                                [1, 3, 2]],
                                               dtype = torch.int32,
                                               device = pyredner.get_device()),
                        uvs = torch.tensor([[0.05, 0.05],
                                            [0.05, 0.95],
                                            [0.95, 0.05],
                                            [0.95, 0.95]], device = pyredner.get_device()),
                        material = mat_checkerboard)
scene = pyredner.Scene(camera=cam, objects=[plane])
img = pyredner.render_albedo(scene=scene)
pyredner.imwrite(img.cpu(), 'results/test_camera_distortion/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_camera_distortion/target.png')
# Read the target image we just saved.
target = pyredner.imread('results/test_camera_distortion/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

cam.distortion_params = torch.zeros(8, requires_grad = True)
scene = pyredner.Scene(camera=cam, objects=[plane])
img = pyredner.render_albedo(scene=scene)
pyredner.imwrite(img.cpu(), 'results/test_camera_distortion/init.exr')
pyredner.imwrite(img.cpu(), 'results/test_camera_distortion/init.png')

# Optimize for triangle vertices.
optimizer = torch.optim.Adam([cam.distortion_params], lr=1e-3)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    scene = pyredner.Scene(camera=cam, objects=[plane])
    img = pyredner.render_albedo(scene=scene)
    pyredner.imwrite(img.cpu(), 'results/test_camera_distortion/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    loss.backward()
    print('grad:', cam.distortion_params.grad)

    optimizer.step()
    print('distortion_params:', cam.distortion_params)

img = pyredner.render_albedo(scene=scene)
pyredner.imwrite(img.cpu(), 'results/test_camera_distortion/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_camera_distortion/final.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_camera_distortion/iter_%d.png", "-vb", "20M",
    "results/test_camera_distortion/out.mp4"])
