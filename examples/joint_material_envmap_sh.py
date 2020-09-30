import torch
import math
import numpy as np
import pyredner

# This is an example for setting up a spherical harmonics parameterized environment map
# and jointly optimize for the spherical harmonics coefficients and object materials

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

# Setup camera
cam = pyredner.Camera(position = torch.tensor([0.0, 0.0, -5.0]),
                      look_at = torch.tensor([0.0, 0.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([45.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      fisheye = False)

# Setup material
mat_grey = pyredner.Material(\
    diffuse_reflectance = \
        torch.tensor([0.4, 0.4, 0.4], device = pyredner.get_device()),
    specular_reflectance = \
        torch.tensor([0.5, 0.5, 0.5], device = pyredner.get_device()),
    roughness = \
        torch.tensor([0.02], device = pyredner.get_device()))
materials = [mat_grey]

# Setup scene geometry: we use the utility function "generate_sphere" to generate a sphere
# triangle mesh
vertices, indices, uvs, normals = pyredner.generate_sphere(128, 64)
shape_sphere = pyredner.Shape(\
    vertices = vertices,
    indices = indices,
    uvs = uvs,
    normals = normals,
    material_id = 0)
shapes = [shape_sphere]

# Setup lighting: the scene is lit by a single environment map, parameterized by 3rd-order
# spherical harmonics coefficients.
# First we setup the target coefficients for r, g, b,
# taken from https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf
# Last 7 coefficients are randomly determined
coeffs = torch.tensor([[ 0.79,
                         0.39, -0.35, -0.34,
                        -0.11, -0.26, -0.16,  0.56,  0.21,
                         0.10,  0.05, -0.20, -0.03, -0.10, -0.30, -0.01], # coeffs for red
                       [ 0.44,
                         0.35, -0.18, -0.06,
                        -0.05, -0.22, -0.09,  0.21, -0.05,
                         0.03,  0.07, -0.01, -0.09, -0.06,  0.03, 0.05], # coeffs for green
                       [ 0.54,
                         0.60, -0.27,  0.01,
                        -0.12, -0.47, -0.15,  0.14, -0.30,
                         0.10,  0.04,  0.08, -0.10, -0.02, -0.07, 0.06]], # coeffs for blue
                       device = pyredner.get_device())
# Deringing: directly using the original coefficients creates aliasing due to truncation,
#            which results in negative values in the environment map when we rasterize the coefficients.
#            Our solution is to multiply (convolve in sptial domain) the coefficients with a low pass
#            filter.
#            Following the recommendation in https://www.ppsloan.org/publications/shdering.pdf
#            We use sinc^4 filter with a window size of 6
def deringing(coeffs, window):
    deringed_coeffs = torch.zeros_like(coeffs)
    deringed_coeffs[:, 0] += coeffs[:, 0]
    deringed_coeffs[:, 1:1 + 3] += \
        coeffs[:, 1:1 + 3] * math.pow(math.sin(math.pi * 1.0 / window) / (math.pi * 1.0 / window), 4.0)
    deringed_coeffs[:, 4:4 + 5] += \
        coeffs[:, 4:4 + 5] * math.pow(math.sin(math.pi * 2.0 / window) / (math.pi * 2.0 / window), 4.0)
    deringed_coeffs[:, 9:9 + 7] += \
        coeffs[:, 9:9 + 7] * math.pow(math.sin(math.pi * 3.0 / window) / (math.pi * 3.0 / window), 4.0)
    return deringed_coeffs
deringed_coeffs = deringing(coeffs, 6.0)
res = (128, 128)
# We call the utility function SH_reconstruct to rasterize the coefficients into an envmap
envmap = pyredner.SH_reconstruct(deringed_coeffs, res)
# Save the target envmap
pyredner.imwrite(envmap.cpu(), 'results/joint_material_envmap_sh/target_envmap.exr')
# Convert the PyTorch tensor into pyredner compatible envmap
envmap = pyredner.EnvironmentMap(envmap)
# Setup the scene
scene = pyredner.Scene(camera = cam,
                       shapes = shapes,
                       materials = materials,
                       envmap = envmap)
# Serialize the scene
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 1)
# Render the target
render = pyredner.RenderFunction.apply
img = render(0, *scene_args)
# Save the target image
pyredner.imwrite(img.cpu(), 'results/joint_material_envmap_sh/target.exr')
pyredner.imwrite(img.cpu(), 'results/joint_material_envmap_sh/target.png')
# Read the target image back
target = pyredner.imread('results/joint_material_envmap_sh/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda()

# Reset the coefficients to some constant color, repeat the same process as in target envmap
coeffs = torch.tensor([[ 0.5,
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # coeffs for red
                       [ 0.5,
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # coeffs for green
                       [ 0.5,
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], # coeffs for blue
                       device = pyredner.get_device(),
                       requires_grad = True)
deringed_coeffs = deringing(coeffs, 6.0)
envmap = pyredner.SH_reconstruct(deringed_coeffs, res)
pyredner.imwrite(envmap.cpu(), 'results/joint_material_envmap_sh/init_envmap.exr')
envmap = pyredner.EnvironmentMap(envmap)
# Also reset the material since we want to do joint estimation
# We use an intermediate "param" variable and then take absolute value of it to avoid negative values
# A better way to resolve this is to use projective SGD
diffuse_reflectance_param = \
    torch.tensor([0.3, 0.3, 0.3], device = pyredner.get_device(), requires_grad = True)
specular_reflectance_param = \
    torch.tensor([0.3, 0.3, 0.3], device = pyredner.get_device(), requires_grad = True)
roughness_param = torch.tensor([0.3], device = pyredner.get_device(), requires_grad = True)
diffuse_reflectance = diffuse_reflectance_param.abs()
specular_reflectance = specular_reflectance_param.abs()
roughness = roughness_param.abs() 
mat_grey = pyredner.Material(\
    diffuse_reflectance = diffuse_reflectance,
    specular_reflectance = specular_reflectance,
    roughness = roughness)
materials = [mat_grey]
scene = pyredner.Scene(camera = cam,
                       shapes = shapes,
                       materials = materials,
                       envmap = envmap)
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 1)
img = render(1, *scene_args)
pyredner.imwrite(img.cpu(), 'results/joint_material_envmap_sh/init.png')
diff = torch.abs(target - img)
pyredner.imwrite(diff.cpu(), 'results/joint_material_envmap_sh/init_diff.png')

# Finally we can start the Adam iteration
optimizer = torch.optim.Adam(\
    [coeffs, diffuse_reflectance_param, specular_reflectance_param, roughness_param], lr=3e-2)
for t in range(400):
    print('iteration:', t)
    optimizer.zero_grad()
    # Repeat the envmap generation & material for the gradients
    deringed_coeffs = deringing(coeffs, 6.0)
    envmap = pyredner.SH_reconstruct(deringed_coeffs, res)
    pyredner.imwrite(envmap.cpu(), 'results/joint_material_envmap_sh/envmap_{}.exr'.format(t))
    envmap = pyredner.EnvironmentMap(envmap)
    diffuse_reflectance = diffuse_reflectance_param.abs()
    specular_reflectance = specular_reflectance_param.abs()
    roughness = roughness_param.abs() # avoid going below zero
    materials[0] = pyredner.Material(\
        diffuse_reflectance = diffuse_reflectance,
        specular_reflectance = specular_reflectance,
        roughness = roughness)
    scene = pyredner.Scene(camera = cam,
                           shapes = shapes,
                           materials = materials,
                           envmap = envmap)
    scene_args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 4,
        max_bounces = 1)
    img = render(t+1, *scene_args)
    pyredner.imwrite(img.cpu(), 'results/joint_material_envmap_sh/iter_{}.png'.format(t))
    loss = torch.pow(img - target, 2).sum()
    print('loss:', loss.item())

    loss.backward()
    optimizer.step()

    # Print the gradients of the coefficients, material parameters
    print('coeffs.grad:', coeffs.grad)
    print('diffuse_reflectance_param.grad:', diffuse_reflectance_param.grad)
    print('specular_reflectance_param.grad:', specular_reflectance_param.grad)
    print('roughness_param.grad:', roughness_param.grad)
    # Print the current parameters
    print('coeffs:', coeffs)
    print('diffuse_reflectance:', diffuse_reflectance)
    print('specular_reflectance:', specular_reflectance)
    print('roughness:', roughness)

scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 1)
img = render(202, *scene_args)
pyredner.imwrite(img.cpu(), 'results/joint_material_envmap_sh/final.exr')
pyredner.imwrite(img.cpu(), 'results/joint_material_envmap_sh/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/joint_material_envmap_sh/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/joint_material_envmap_sh/iter_%d.png", "-vb", "20M",
    "results/joint_material_envmap_sh/out.mp4"])
