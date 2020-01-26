import pyredner
import torch

# Test the sample pixel center flag

pyredner.set_use_gpu(torch.cuda.is_available())
objects = pyredner.load_obj('scenes/teapot.obj', return_objects=True)
camera = pyredner.automatic_camera_placement(objects, resolution=(128, 128))
scene = pyredner.Scene(camera = camera, objects = objects)
img = pyredner.render_albedo(scene, sample_pixel_center = True)
pyredner.imwrite(img.cpu(), 'results/test_sample_pixel_center/img_no_aa.exr')
img = pyredner.render_albedo(scene, sample_pixel_center = False)
pyredner.imwrite(img.cpu(), 'results/test_sample_pixel_center/img_with_aa.exr')