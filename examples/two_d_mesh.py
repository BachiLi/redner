import pyredner
import numpy as np
import torch
import redner

# Optimize vertices of 2D meshes

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

# Setup camera: We place the camera at (0, 0, -1), with look vector
#               (0, 0, 1). We also use an orthographic camera just to
#               make the projection more "2D": the depth is only used
#               for determining the order of the meshes.
cam = pyredner.Camera(position = torch.tensor([0.0, 0.0, -1.0]),
                      look_at = torch.tensor([0.0, 0.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([45.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      camera_type = redner.CameraType.orthographic)

# The materials: 
mat_quad = pyredner.Material(\
    diffuse_reflectance = torch.tensor([0.75, 0.75, 0.25],
    device = pyredner.get_device()))
mat_tri = pyredner.Material(\
    diffuse_reflectance = torch.tensor([0.9, 0.35, 0.35],
    device = pyredner.get_device()))
materials = [mat_quad, mat_tri]

# We'll have a quad and a triangle as our meshes.
# First we define the 2D coordinates. The size of the screen is
# from -1.0 to 1.0. Y is pointing up.
quad_vertices_2d = torch.tensor(\
    [[-0.3, 0.5], [0.2, 0.6], [-0.5, -0.3], [0.5, -0.4]],
    device = pyredner.get_device())
tri_vertices_2d = torch.tensor(\
    [[-0.6, 0.3], [0.4, 0.5], [-0.1, -0.2]],
    device = pyredner.get_device())
# We need to pad the depth coordinates for these vertices
# We'll assign depth = 1 for the quad, depth = 0 for the triangle,
# so the triangle will block the quad.
quad_vertices = torch.cat((quad_vertices_2d,
    torch.ones(quad_vertices_2d.shape[0], 1, device = pyredner.get_device())), dim=1).contiguous()
tri_vertices = torch.cat((tri_vertices_2d,
    torch.zeros(tri_vertices_2d.shape[0], 1, device = pyredner.get_device())), dim=1).contiguous()
quad_indices = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype = torch.int32, device = pyredner.get_device())
tri_indices = torch.tensor([[0, 1, 2]], dtype = torch.int32, device = pyredner.get_device())
shape_quad = pyredner.Shape(\
    vertices = quad_vertices,
    indices = quad_indices,
    material_id = 0)
shape_tri = pyredner.Shape(\
    vertices = tri_vertices,
    indices = tri_indices,
    material_id = 1)
shapes = [shape_quad, shape_tri]

# Setup the scene. We don't need lights.
scene = pyredner.Scene(camera = cam,
                       shapes = shapes,
                       materials = materials)
# We output the shape id, so that we can shape it later
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    # Set max bounces to 0, we don't need lighting.
    max_bounces = 0,
    # Use the diffuse color as the output
    channels = [redner.channels.diffuse_reflectance])

# Render the scene as our target image.
render = pyredner.RenderFunction.apply
img = render(0, *args)
pyredner.imwrite(img.cpu(), 'results/two_d_mesh/target.exr')
pyredner.imwrite(img.cpu(), 'results/two_d_mesh/target.png')
target = pyredner.imread('results/two_d_mesh/target.exr')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

# Perturb the scene, this is our initial guess
quad_vertices_2d = torch.tensor(\
    [[-0.5, 0.3], [0.3, 0.4], [-0.7, -0.2], [0.4, -0.3]],
    device = pyredner.get_device(),
    requires_grad = True)
tri_vertices_2d = torch.tensor(\
    [[-0.5, 0.4], [0.4, 0.6], [-0.0, -0.3]],
    device = pyredner.get_device(),
    requires_grad = True)
# Need to redo the concatenation
shape_quad.vertices = torch.cat((quad_vertices_2d,
    torch.ones(quad_vertices_2d.shape[0], 1, device = pyredner.get_device())), dim=1).contiguous()
shape_tri.vertices = torch.cat((tri_vertices_2d,
    torch.zeros(tri_vertices_2d.shape[0], 1, device = pyredner.get_device())), dim=1).contiguous()
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    # Set max bounces to 0, we don't need lighting.
    max_bounces = 0,
    # Use the diffuse color as the output
    channels = [redner.channels.diffuse_reflectance])
# Render the initial guess.
render = pyredner.RenderFunction.apply
img = render(0, *args)
pyredner.imwrite(img.cpu(), 'results/two_d_mesh/init.exr')
pyredner.imwrite(img.cpu(), 'results/two_d_mesh/init.png')

# Optimize for mesh vertices
optimizer = torch.optim.Adam([quad_vertices_2d, tri_vertices_2d], lr=4e-2)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    # Forward pass: render the image
    # Need to redo the concatenation
    shape_quad.vertices = torch.cat((quad_vertices_2d,
        torch.ones(quad_vertices_2d.shape[0], 1, device = pyredner.get_device())), dim=1).contiguous()
    shape_tri.vertices = torch.cat((tri_vertices_2d,
        torch.zeros(tri_vertices_2d.shape[0], 1, device = pyredner.get_device())), dim=1).contiguous()
    args = pyredner.RenderFunction.serialize_scene(\
        scene = scene,
        num_samples = 1,
        max_bounces = 0,
        channels = [redner.channels.diffuse_reflectance])
    img = render(t+1, *args)
    pyredner.imwrite(img.cpu(), 'results/two_d_mesh/iter_{}.png'.format(t))
    loss = (img - target).pow(2).sum()
    print('loss:', loss.item())

    loss.backward()
    print('quad_vertices_2d.grad:', quad_vertices_2d.grad)
    print('tri_vertices_2d.grad:', tri_vertices_2d.grad)

    optimizer.step()
    print('quad_vertices_2d:', quad_vertices_2d)
    print('tri_vertices_2d:', tri_vertices_2d)

shape_quad.vertices = torch.cat((quad_vertices_2d,
    torch.ones(quad_vertices_2d.shape[0], 1, device = pyredner.get_device())), dim=1).contiguous()
shape_tri.vertices = torch.cat((tri_vertices_2d,
    torch.zeros(tri_vertices_2d.shape[0], 1, device = pyredner.get_device())), dim=1).contiguous()
args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 0,
    channels = [redner.channels.diffuse_reflectance])
img = render(t+1, *args)

pyredner.imwrite(img.cpu(), 'results/two_d_mesh/final.exr')
pyredner.imwrite(img.cpu(), 'results/two_d_mesh/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/two_d_mesh/final_diff.png')

from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/two_d_mesh/iter_%d.png", "-vb", "20M",
    "results/two_d_mesh/out.mp4"])
