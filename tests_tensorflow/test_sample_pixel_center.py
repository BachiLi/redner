import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
tf.compat.v1.enable_eager_execution() # redner only supports eager mode
import pyredner_tensorflow as pyredner

# Test the sample pixel center flag

objects = pyredner.load_obj('scenes/teapot.obj', return_objects=True)
camera = pyredner.automatic_camera_placement(objects, resolution=(128, 128))
scene = pyredner.Scene(camera = camera, objects = objects)
img = pyredner.render_albedo(scene, sample_pixel_center = True)
pyredner.imwrite(img.cpu(), 'results/test_sample_pixel_center/img_no_aa.exr')
img = pyredner.render_albedo(scene, sample_pixel_center = False)
pyredner.imwrite(img.cpu(), 'results/test_sample_pixel_center/img_with_aa.exr')
