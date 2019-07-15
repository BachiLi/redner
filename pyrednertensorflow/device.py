import tensorflow as tf

use_gpu = False
# use_gpu = tf.test.is_gpu_available(
#     cuda_only=False,
#     min_cuda_compute_capability=None
# )

def set_use_gpu(v):
    global use_gpu
    use_gpu = v

def get_use_gpu():
    global use_gpu
    return use_gpu

# def get_device():
#     global use_gpu
#     return torch.device('cuda') if use_gpu else torch.device('cpu')
