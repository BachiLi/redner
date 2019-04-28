import pyredner
import redner
import torch
import scipy
import scipy.ndimage
import numpy as np

# Test Quasi Monte Carlo rendering.
# We optimize for the materials of a Cornell box scene

scene = pyredner.load_mitsuba('scenes/cbox/cbox.xml')
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 5, # Set max_bounces = 5 for global illumination
    sampler_type = redner.SamplerType.sobol) # Use Sobol sampler
render = pyredner.RenderFunction.apply
img = render(0, *scene_args)
pyredner.imwrite(img.cpu(), 'results/test_qmc/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_qmc/target.png')
target = pyredner.imread('results/test_qmc/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda()

# Now let's generate an initial guess by perturbing the reference.
# Let's set all the diffuse color to gray by manipulating material.diffuse_reflectance.
# We also store all the material variables to optimize in a list.
material_vars = []
for mi, m in enumerate(scene.materials):
    var = torch.tensor([0.5, 0.5, 0.5],
                       device = pyredner.get_device(),
                       requires_grad = True)
    material_vars.append(var)
    m.diffuse_reflectance = pyredner.Texture(var)

# Serialize the scene and render the initial guess
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 5,
    sampler_type = redner.SamplerType.sobol)
img = render(1, *scene_args)
pyredner.imwrite(img.cpu(), 'results/test_qmc/init.png')

# Optimize for parameters.
optimizer = torch.optim.Adam(material_vars, lr=1e-2)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: serialize the scene and render the image
    # Need to redefine the camera
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4,
        max_bounces = 5,
        sampler_type = redner.SamplerType.sobol)
    # Important to use a different seed every iteration, otherwise the result
    # would be biased.
    img = render(t+1, *scene_args)
    pyredner.imwrite(img.cpu(), 'results/test_qmc/iter_{}.png'.format(t))

    # Compute the loss function.
    # We clamp the difference between -1 and 1 to prevent
    # light source from being dominating the loss function
    loss = (img - target).clamp(-1.0, 1.0).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()

    # Take a gradient descent step.
    optimizer.step()

    # Important: the material parameters has hard constraints: the
    # reflectance and roughness cannot be negative. We enforce them here
    # by projecting the values to the boundaries.
    for var in material_vars:
        var.data = var.data.clamp(1e-5, 1.0)
        print(var)

# Render the final result.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 5,
    sampler_type = redner.SamplerType.sobol)
img = render(202, *scene_args)
# Save the images and differences.
pyredner.imwrite(img.cpu(), 'results/test_qmc/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_qmc/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/test_qmc/final_diff.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_qmc/iter_%d.png", "-vb", "20M",
    "results/test_qmc/out.mp4"])
