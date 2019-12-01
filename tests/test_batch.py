import torch
import math
import pyredner
import redner

pyredner.set_use_gpu(torch.cuda.is_available())



class BatchRenderFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, seed, *args):
        batch_dims = args[0]
        args_old_format = args[1:]
        chunk_len = int(len(args_old_format)/batch_dims)
        h, w = args_old_format[11]
        result = torch.zeros(\
            batch_dims, h, w, 9, device = pyredner.get_device(), requires_grad=True)
        for k in range(0, batch_dims):
            sub_args = args_old_format[k*chunk_len:(k+1)*chunk_len]
            result[k, :, :, :] = pyredner.RenderFunction.forward(ctx, seed, *sub_args)
        return result

    @staticmethod
    def backward(ctx, grad_img):
        #None gradient for seed and batch_dims
        ret_list = (None, None,)
        batch_dims = grad_img.shape[0]
        for k in range(0, batch_dims):
            #[1:] cuz original backward function returns None grad for seed input, but we manage that ourselves
            ret_list = ret_list + pyredner.RenderFunction.backward(ctx, grad_img[k,:,:,:])[1:]
        return ret_list


batch_render = BatchRenderFunction.apply

# Load from the teapot Wavefront object file
material_map, mesh_list, light_map = pyredner.load_obj('../tutorials/teapot.obj')
# Compute shading normal
for _, mesh in mesh_list:
    mesh.normals = pyredner.compute_vertex_normal(mesh.vertices, mesh.indices)

# Setup camera
cam = pyredner.Camera(position = torch.tensor([0.0, 30.0, 200.0]),
                      look_at = torch.tensor([0.0, 30.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      fov = torch.tensor([45.0]), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (256, 256),
                      fisheye = False)

#
mesh = mesh_list[0][1]
shapes = [pyredner.Shape(\
        vertices = mesh.vertices,
        indices = mesh.indices,
        uvs = mesh.uvs,
        normals = mesh.normals,
        material_id = 0)]


tex_path='../tutorials/teapot.png'
tex_tensor = pyredner.imread(tex_path)
if pyredner.get_use_gpu():
    tex_tensor = tex_tensor.cuda(device = pyredner.get_device())


diffuse_reflectance = tex_tensor
materials = [pyredner.Material(diffuse_reflectance=diffuse_reflectance)]


# Construct the scene.
# Don't setup any light sources, only use primary visibility.
scene = pyredner.Scene(cam, shapes, materials, area_lights = [], envmap = None)

# TEST1: render (test forward function)

scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 16,
    max_bounces = 0,
    channels = [redner.channels.position,
                redner.channels.shading_normal,
                redner.channels.diffuse_reflectance])
scene_args = [2] + 2*scene_args
g_buffer = batch_render(0, *scene_args)

img1 = g_buffer[0,:,:,6:9]
pyredner.imwrite(img1.cpu(), 'results/test_multichannels/test1.png')
img2 = g_buffer[1,:,:,6:9]
pyredner.imwrite(img2.cpu(), 'results/test_multichannels/test2.png')

# TEST2: convergence (test backward function)
target = pyredner.imread('results/test_multichannels/test1.png')
if pyredner.get_use_gpu():
    target = target.cuda(device = pyredner.get_device())

batch_dims = 2
diffuse_reflectance = torch.zeros(\
    batch_dims, 128, 128, 3, device = pyredner.get_device(), requires_grad=True)

scenes = [scene, scene]

optimizer = torch.optim.Adam([diffuse_reflectance], lr=1e-2)
for t in range(200):
    print('iteration:', t)
    optimizer.zero_grad()
    scene_args_batch = [batch_dims]
    for k in range(0, batch_dims):
        scenes[k].materials[0].diffuse_reflectance = pyredner.Texture(diffuse_reflectance[k,:,:,:])
        scene_args = pyredner.RenderFunction.serialize_scene(\
            scene = scenes[k],
            num_samples = 16,
            max_bounces = 0,
            channels = [redner.channels.position,
                    redner.channels.shading_normal,
                    redner.channels.diffuse_reflectance])
        scene_args_batch = scene_args_batch + scene_args
    g_buffer = batch_render(t, *scene_args_batch)

    img1 = g_buffer[0,:,:,6:9]
    img2 = g_buffer[1,:,:,6:9]
    loss = (img1 - target).pow(2).sum() + (img2 - target).pow(2).sum()
    print('loss:', loss.item())
    loss.backward()
    optimizer.step()

pyredner.imwrite(diffuse_reflectance[0, :, :, :].cpu(), 'results/test_multichannels/testtex1.png')
pyredner.imwrite(diffuse_reflectance[1, :, :, :].cpu(), 'results/test_multichannels/testtex2.png')
