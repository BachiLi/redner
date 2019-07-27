import tensorflow as tf

use_gpu = tf.test.is_gpu_available(
    cuda_only=True,
    min_cuda_compute_capability=None
)
cpu_device_id = 0
gpu_device_id = 0

def get_device_name():
    global use_gpu
    global cpu_device_id
    global gpu_device_id
    return '/device:gpu:' + str(gpu_device_id) if use_gpu else '/device:cpu:' + str(cpu_device_id)

def set_use_gpu(v):
    global use_gpu
    use_gpu = v

def get_use_gpu():
    global use_gpu
    return use_gpu

def set_cpu_device_id(did):
    global cpu_device_id
    cpu_device_id = did

def get_cpu_device_id():
    global cpu_device_id
    return cpu_device_id

def set_gpu_device_id(did):
    global gpu_device_id
    gpu_device_id = did

def get_gpu_device_id():
    global gpu_device_id
    return gpu_device_id
