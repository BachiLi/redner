import pyredner
import numpy as np
import torch

# Optimize for a textured plane in a specular reflection

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

# Load the scene from a Mitsuba scene file
scene = pyredner.load_mitsuba('scenes/teapot_specular.xml')

# The last material is the teapot material, set it to a specular material
scene.materials[-1].diffuse_reflectance = \
    pyredner.Texture(torch.tensor([0.15, 0.2, 0.15], device = pyredner.get_device()))
scene.materials[-1].specular_reflectance = \
    pyredner.Texture(torch.tensor([0.8, 0.8, 0.8], device = pyredner.get_device()))
scene.materials[-1].roughness = \
    pyredner.Texture(torch.tensor([0.0001], device = pyredner.get_device()))
args=pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 2)

render = pyredner.RenderFunction.apply
# Render our target. The first argument is the seed for RNG in the renderer.
img = render(0, *args)
pyredner.imwrite(img.cpu(), 'results/test_teapot_specular/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_teapot_specular/target.png')
target = pyredner.imread('results/test_teapot_specular/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

# Perturb the scene, this is our initial guess
# We perturb the last shape, which is the SIGGRAPH logo
ref_pos = scene.shapes[-1].vertices
translation = torch.tensor([20.0, 0.0, 2.0], device = pyredner.get_device(), requires_grad=True)
scene.shapes[-1].vertices = ref_pos + translation
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 2)
# Render the initial guess
img = render(1, *args)
pyredner.imwrite(img.cpu(), 'results/test_teapot_specular/init.png')
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/test_teapot_specular/init_diff.png')

lr = 0.5
optimizer = torch.optim.Adam([translation], lr=lr)
num_iteration = 400
for t in range(num_iteration):
    print('iteration:', t)
    optimizer.zero_grad()

    scene.shapes[-1].vertices = ref_pos + translation
    args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4,
        max_bounces = 2)
    img = render(t+1, *args)
    pyredner.imwrite(img.cpu(), 'results/test_teapot_specular/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    loss.backward()
    print('translation.grad:', translation.grad)

    torch.nn.utils.clip_grad_norm_(translation, 10)

    optimizer.step()
    print('translation:', translation)

    # Linearly reduce the learning rate
    lr = 0.5 * float(num_iteration - t) / float(num_iteration)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

scene.shapes[-1].vertices = ref_pos + translation
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 2)
img = render(num_iteration + 2, *args)
pyredner.imwrite(img.cpu(), 'results/test_teapot_specular/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_teapot_specular/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/test_teapot_specular/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_teapot_specular/iter_%d.png", "-vb", "20M",
    "results/test_teapot_specular/out.mp4"])
