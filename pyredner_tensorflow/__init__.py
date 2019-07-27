import numpy as np
import tensorflow as tf
from .device import *
from .camera import *
from .shape import *
from .material import *
from .texture import *
from .area_light import *
from .envmap import *
from .scene import *
from .render_tensorflow import *
from .image import *
from .load_obj import load_obj
from .load_mitsuba import load_mitsuba
from .transform import gen_rotate_matrix
from .utils import *
from .channels import *

import os.path
import redner
from tensorflow.python.framework import ops

__data_ptr_module = tf.load_op_library(os.path.join(os.path.dirname(redner.__file__), 'libredner_tf_data_ptr.so'))

DEBUG = False
IS_UNIT_TEST = False

def data_ptr(tensor):    
    addr_as_uint64 = __data_ptr_module.data_ptr(tensor)
    return int(addr_as_uint64)

def write_tensor(path, tensor, height, width):
    with open(path, 'w') as f:
        for i in range(height):
            for j in range(width):
                f.write(f'{tensor[i,j]} ')
            f.write('\n')

def pretty_debug_print(grads, vars, iter_num=-1):
    from pprint import pprint
    print("/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\/\\")
    if iter_num > -1:
        print("Iteration: ", iter_num)
    print(">>> GRADIENTS:")
    if (isinstance(grads, dict)):
        for k, v in grads.items():
            print(k, v.shape, v.numpy())
    elif (isinstance(grads, list)):
        for k in grads:
            print(k)
    print("\n>>> VARIABLES:")
    for v in vars:
        print(v.name, v.shape, v.numpy())

def get_render_args(seed, scene_args):
    return [tf.constant(seed)] + scene_args
