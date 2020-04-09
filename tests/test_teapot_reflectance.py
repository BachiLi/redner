import pyredner
import numpy as np
import torch
import scipy
import scipy.ndimage

# Optimize for material parameters and camera pose

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

# Load the scene from a Mitsuba scene file
scene = pyredner.load_mitsuba('scenes/teapot.xml')

# The last material is the teapot material, set it to the target
scene.materials[-1].diffuse_reflectance = \
    pyredner.Texture(torch.tensor([0.3, 0.2, 0.2], device = pyredner.get_device()))
scene.materials[-1].specular_reflectance = \
    pyredner.Texture(torch.tensor([0.6, 0.6, 0.6], device = pyredner.get_device()))
scene.materials[-1].roughness = \
    pyredner.Texture(torch.tensor([0.05], device = pyredner.get_device()))
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 1024,
    max_bounces = 2)

# Alias of the render function
render = pyredner.RenderFunction.apply
# Render our target. The first argument is the seed for RNG in the renderer.
img = render(0, *args)
pyredner.imwrite(img.cpu(), 'results/test_teapot_reflectance/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_teapot_reflectance/target.png')
target = pyredner.imread('results/test_teapot_reflectance/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

# Perturb the scene, this is our initial guess
cam = scene.camera
cam_position = cam.position
cam_translation = torch.tensor([-0.2, 0.2, -0.2], requires_grad = True)
diffuse_reflectance = torch.tensor([0.3, 0.3, 0.3],
    device = pyredner.get_device(), requires_grad = True)
specular_reflectance = torch.tensor([0.5, 0.5, 0.5],
    device = pyredner.get_device(), requires_grad = True)
roughness = torch.tensor([0.2],
    device = pyredner.get_device(), requires_grad = True)
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
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 1024,
    max_bounces = 2)
# Render the initial guess
img = render(1, *args)
pyredner.imwrite(img.cpu(), 'results/test_teapot_reflectance/init.png')
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/test_teapot_reflectance/init_diff.png')

lr_base = 1e-2
lr = lr_base
optimizer = torch.optim.Adam([diffuse_reflectance,
                              specular_reflectance,
                              roughness,
                              cam_translation], lr=lr)
num_iteration = 400
for t in range(num_iteration):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image
    # need to rerun Camera constructor for autodiff 
    scene.camera = pyredner.Camera(position   = cam_position + cam_translation,
                                   look_at    = cam.look_at + cam_translation,
                                   up         = cam.up,
                                   fov        = cam.fov,
                                   clip_near  = cam.clip_near,
                                   resolution = cam.resolution,
                                   fisheye    = False)
    args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4,
        max_bounces = 2)
    img = render(t+1, *args)
    pyredner.imwrite(img.cpu(), 'results/test_teapot_reflectance/iter_{}.png'.format(t))
    #loss = (img - target).pow(2).sum()

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
    diff_5 = m(torch.nn.functional.conv2d(diff_4, f, padding=3))
    loss = diff_0.pow(2).sum() / (r*r) + \
           diff_1.pow(2).sum() / ((r/2)*(r/2)) + \
           diff_2.pow(2).sum() / ((r/4)*(r/4)) + \
           diff_3.pow(2).sum() / ((r/8)*(r/8)) + \
           diff_4.pow(2).sum() / ((r/16)*(r/16)) + \
           diff_5.pow(2).sum() / ((r/32)*(r/32))

    print('loss:', loss.item())

    loss.backward()
    print('diffuse_reflectance.grad:', diffuse_reflectance.grad)
    print('specular_reflectance.grad:', specular_reflectance.grad)
    print('roughness.grad:', roughness.grad)
    print('cam_translation.grad:', cam_translation.grad)

    # HACK: gradient clipping to deal with outlier gradients
    torch.nn.utils.clip_grad_norm_(roughness, 10)
    torch.nn.utils.clip_grad_norm_(cam_translation, 10)

    optimizer.step()

    print('diffuse_reflectance:', diffuse_reflectance)
    print('specular_reflectance:', specular_reflectance)
    print('roughness:', roughness)
    print('cam_translation:', cam_translation)

    # Linearly reduce the learning rate
    #lr = lr_base * float(num_iteration - t) / float(num_iteration)
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = lr

args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 1024,
    max_bounces = 2)
img = render(num_iteration + 2, *args)
pyredner.imwrite(img.cpu(), 'results/test_teapot_reflectance/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_teapot_reflectance/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/test_teapot_reflectance/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_teapot_reflectance/iter_%d.png", "-vb", "20M",
    "results/test_teapot_reflectance/out.mp4"])
