import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pyredner_tensorflow as pyredner
import redner
import numpy as np

pyredner.set_use_gpu(tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None))

box_v = tf.constant([
    [1,0,1], [-1,0,1], [-1,0,-1], [1,0,-1], [1,2,1], [-1,2,1], [-1,2,-1], [1,2,-1],
], dtype=tf.float32)

box_i = tf.constant([
    [0,1,4],
    [1,2,5],
#     [2,3,6],
    [3,0,7],
    
    [1,5,4],
    [2,6,5],
#     [3,7,6],
    [0,4,7],
    
    [2,1,0],
    [3,2,0],
    [4,5,6],
    [4,6,7],
], dtype=tf.int32)

ground_i = tf.constant([[2,1,0],[3,2,0]], dtype=tf.int32)

sc = .2
light_v = [0,1.99,0] + sc * tf.constant([
    [1,0,1], [-1,0,1], [-1,0,-1], [1,0,-1],
], dtype=tf.float32)
light_i = tf.constant([[0,1,2],[2,3,0]], dtype=tf.int32)


### basedir = '/data/ShapeNetCore.v2/02958343/ec9f938ad52f98abbda72093f9b5aa73'

use_materials = True
matstr = 'withmat' if use_materials else 'nomat'

material_map, mesh_list, light_map = pyredner.load_obj('model_normalized.obj')

def get_cam(cam_rad, cam_th, cam_y=0., look_at=[0.0, 0.0, 0.0]):
    pos = [cam_rad * np.sin(cam_th), cam_y, cam_rad * np.cos(cam_th)]
    cam = pyredner.Camera(position = tf.Variable(pos, dtype=tf.float32, use_resource=True),
                      look_at = tf.Variable(look_at, dtype=tf.float32, use_resource=True),
                      up = tf.Variable([0.0, 1.0, 0.0], dtype=tf.float32, use_resource=True),
                      fov = tf.Variable([45.0], dtype=tf.float32, use_resource=True), # in degree
                      clip_near = 1e-2, # needs to > 0
                      resolution = (512,512),
                      fisheye = False)
    return cam

cam_rad = -1-1/np.tan(45.*.5/180*np.pi) - .05
cam_th = 0. / 180 * np.pi
cam = get_cam(cam_rad, cam_th, cam_y=1, look_at=[0,1,0])

shape_light = pyredner.Shape(light_v, light_i, material_id = 0)
area_lights = [pyredner.AreaLight(shape_id = 1, intensity = [20.,20.,20.])]


shape_box = pyredner.Shape(box_v, box_i, material_id=0)

materials = [pyredner.Material(tf.constant([1,1,1.]))]

material_id_map = {}
for key, value in material_map.items():
    material_id_map[key] = len(materials)
    materials.append(value)

shapes = [shape_box, shape_light]

# M transforms vertices to fit inside cornell box
M = np.reshape([1.0176800583091991e-16, 0.0, -1.661997661722135, -0.002306598864749619, 
                0.0, 1.661997661722135, 0.0, 0.1513695297999608, 
                1.661997661722135, 0.0, 1.0176800583091991e-16, 0.00023089923337398828, 
                0.0, 0.0, 0.0, 1.0], (4,4))
for mtl_name, mesh in mesh_list:
    if mesh.vertices.shape[0] > 0:
        v = mesh.vertices @ M[:3,:3].T + M[:3, -1]
        shapes.append(pyredner.Shape(v, mesh.indices, mesh.uvs, mesh.normals,
            material_id = material_id_map[mtl_name] if use_materials else 0))

scene = pyredner.Scene(cam, shapes, materials, area_lights, None)
scene_args = pyredner.serialize_scene(scene, 256, 3)
img = pyredner.render(0, *scene_args)
pyredner.imwrite(img, f'cornell_box_{matstr}.png')