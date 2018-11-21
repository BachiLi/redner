import torch
import numpy as np
import pyredner
import torch
import enum

class Texture:
    def __init__(self,
                 texels,
                 uv_scale = torch.tensor([1.0, 1.0])):
        self.texels = texels
        self.uv_scale = uv_scale
