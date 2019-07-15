from typing import List, Set, Dict, Tuple, Optional, Callable, Union
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

from tensorflow.python.framework import ops

__data_ptr_module = tf.load_op_library('/data_ptr.so')
__scatter_add_module = tf.load_op_library('/pytorch_scatter_add.so')

is_tensor = tf.contrib.framework.is_tensor

DEBUG = False
IS_UNIT_TEST = False

@ops.RegisterGradient("PytorchScatterAdd")
def _pytorch_scatter_add_grad(op, grad):
    """The gradients for `zero_out`.

    Args:
    op: The `zero_out` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `zero_out` op.

    Returns:
    Gradients with respect to the input of `zero_out`.
    """
    return [grad, None, None]

def zeros_like(
        tensor: Union[np.ndarray, tf.Tensor, tf.Variable], 
        is_var=False, 
        dtype=tf.float32) -> Union[tf.Tensor, tf.Variable]:
    zeros = np.zeros_like(tensor)
    if is_var:
        return tf.Variable(zeros, dtype=dtype)
    else:
        return tf.constant(zeros, dtype=dtype)

def data_ptr(tensor):
    
    addr_as_uint64 = __data_ptr_module.data_ptr(tensor)
    # import pdb;pdb.set_trace()
    return int(addr_as_uint64)


def scatter_add(ref, indices, updates):
    output = __scatter_add_module.pytorch_scatter_add(ref, indices, updates)
    return output



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
