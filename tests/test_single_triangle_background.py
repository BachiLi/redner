import pyredner
import redner
import numpy as np
import torch
import skimage.transform

# Optimize three vertices of a single triangle, with a SIGGRAPH logo background

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

# Set up the pyredner scene for rendering:

# Setup camera
cam = pyredner.Camera(position = torch.tensor([0.0, 0.0, -5.0]),
                      look_at = torch.tensor([0.0, 0.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([45.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      fisheye = False)

# Setup materials
mat_grey = pyredner.Material(\
    diffuse_reflectance = \
        torch.tensor([0.5, 0.5, 0.5], device = pyredner.get_device()))
# The material list of the scene
materials = [mat_grey]

# Setup geometries
shape_triangle = pyredner.Shape(\
    vertices = torch.tensor([[-1.7, 1.0, 0.0], [1.0, 1.0, 0.0], [-0.5, -1.0, 0.0]],
        device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2]], dtype = torch.int32,
        device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)
# Setup light source shape
shape_light = pyredner.Shape(\
    vertices = torch.tensor([[-1.0, -1.0, -7.0],
                             [ 1.0, -1.0, -7.0],
                             [-1.0,  1.0, -7.0],
                             [ 1.0,  1.0, -7.0]], device = pyredner.get_device()),
    indices = torch.tensor([[0, 1, 2],[1, 3, 2]],
        dtype = torch.int32, device = pyredner.get_device()),
    uvs = None,
    normals = None,
    material_id = 0)
# The shape list of the scene
shapes = [shape_triangle, shape_light]

# Setup light source
light = pyredner.AreaLight(shape_id = 1, 
                           intensity = torch.tensor([20.0,20.0,20.0]))
area_lights = [light]

# Construct the scene
scene = pyredner.Scene(cam, shapes, materials, area_lights)
# Serialize the scene
# Here we specify the output channels as "radiance" and "alpha"
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1,
    channels = [redner.channels.radiance, redner.channels.alpha])

# Render the scene as our target image.
render = pyredner.RenderFunction.apply
# Render. The first argument is the seed for RNG in the renderer.
img = render(0, *scene_args)
# Since we specified alpha as output channel, img has 4 channels now
# We blend the image with a background image
background = pyredner.imread('scenes/textures/siggraph.jpg')
background = torch.from_numpy(skimage.transform.resize(background.numpy(), (256, 256, 3)))
if pyredner.get_use_gpu():
    background = background.cuda(device = pyredner.get_device())
background = background.type_as(img)
img = img[:, :, :3] * img[:, :, 3:4] + background * (1 - img[:, :, 3:4])

# Save the images.
# The output image is in the GPU memory if you are using GPU.
pyredner.imwrite(img.cpu(), 'results/test_single_triangle_background/target.exr')
pyredner.imwrite(img.cpu(), 'results/test_single_triangle_background/target.png')
# Read the target image we just saved.
target = pyredner.imread('results/test_single_triangle_background/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

# Perturb the scene, this is our initial guess.
shape_triangle.vertices = torch.tensor(\
    [[-2.0,1.5,0.3], [0.9,1.2,-0.3], [-0.4,-1.4,0.2]],
    device = pyredner.get_device(),
    requires_grad = True)
# We need to serialize the scene again to get the new arguments.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1,
    channels = [redner.channels.radiance, redner.channels.alpha])
# Render the initial guess.
img = render(1, *scene_args)
# Blend the image with a background image
img = img[:, :, :3] * img[:, :, 3:4] + background * (1 - img[:, :, 3:4])
# Save the images.
pyredner.imwrite(img.cpu(), 'results/test_single_triangle_background/init.png')
# Compute the difference and save the images.
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/test_single_triangle_background/init_diff.png')

# Optimize for triangle vertices.
optimizer = torch.optim.Adam([shape_triangle.vertices], lr=5e-2)
# Run 300 Adam iterations.
for t in range(300):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image.
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4, # We use less samples in the Adam loop.
        max_bounces = 1,
        channels = [redner.channels.radiance, redner.channels.alpha])
    # Important to use a different seed every iteration, otherwise the result
    # would be biased.
    img = render(t+1, *scene_args)
    # Blend the image with a background image
    img = img[:, :, :3] * img[:, :, 3:4] + background * (1 - img[:, :, 3:4])
    # Save the intermediate render.
    pyredner.imwrite(img.cpu(), 'results/test_single_triangle_background/iter_{}.png'.format(t))
    # Compute the loss function. Here it is L2.
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    # Backpropagate the gradients.
    loss.backward()
    # Print the gradients of the three vertices.
    print('grad:', shape_triangle.vertices.grad)

    # Take a gradient descent step.
    optimizer.step()
    # Print the current three vertices.
    print('vertices:', shape_triangle.vertices)

# Render the final result.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 1,
    channels = [redner.channels.radiance, redner.channels.alpha])
img = render(302, *scene_args)
# Blend the image with a background image
img = img[:, :, :3] * img[:, :, 3:4] + background * (1 - img[:, :, 3:4])
# Save the images and differences.
pyredner.imwrite(img.cpu(), 'results/test_single_triangle_background/final.exr')
pyredner.imwrite(img.cpu(), 'results/test_single_triangle_background/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/test_single_triangle_background/final_diff.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/test_single_triangle_background/iter_%d.png", "-vb", "20M",
    "results/test_single_triangle_background/out.mp4"])
