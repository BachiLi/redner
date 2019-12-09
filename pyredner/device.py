import torch

use_gpu = torch.cuda.is_available()
device = torch.device('cuda') if use_gpu else torch.device('cpu')

def set_use_gpu(v: bool):
    """
        Set whether to use CUDA or not.
    """
    global use_gpu
    global device
    use_gpu = v
    if not use_gpu:
        device = torch.device('cpu')
    else:
        assert(torch.cuda.is_available())
        device = torch.device('cuda')

def get_use_gpu():
    """
        Get whether we are using CUDA or not.
    """
    global use_gpu
    return use_gpu

def set_device(d: torch.device):
    """
        Set the torch device we are using.
    """
    global device
    global use_gpu
    device = d
    use_gpu = device.type == 'cuda'

def get_device():
    """
        Get the torch device we are using.
    """
    global device
    return device

