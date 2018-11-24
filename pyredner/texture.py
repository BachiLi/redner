import torch
import numpy as np
import pyredner
import torch
import enum
import math

class Texture:
    def __init__(self,
                 texels,
                 uv_scale = torch.tensor([1.0, 1.0])):
        if len(texels.shape) >= 2:
            # Build a mipmap for texels
            width = max(texels.shape[0], texels.shape[1])
            num_levels = math.ceil(math.log(width, 2) + 1)
            mipmap = texels.unsqueeze(0).expand(num_levels, *texels.shape)
            if len(mipmap.shape) == 3:
                mipmap.unsqueeze_(-1)
            num_channels = mipmap.shape[-1]
            box_filter = torch.ones(num_channels, 1, 2, 2,
                device = texels.device) / 4.0
            # Convert from NHWC to NCHW
            mipmap = mipmap.permute(0, 3, 1, 2)
            for l in range(1, num_levels):
                prev_lvl = mipmap[l-1:l, ...]
                dilation_size = 2 ** (l - 1)
                # Pad for circular boundary condition
                # This is slow. The hope is at some point PyTorch will support
                # circular boundary condition for conv2d
                prev_lvl = torch.cat([prev_lvl, prev_lvl[:,:,0:dilation_size]], dim=2)
                prev_lvl = torch.cat([prev_lvl, prev_lvl[:,:,:,0:dilation_size]], dim=3)
                current_lvl = torch.nn.functional.conv2d(\
                    prev_lvl, box_filter,
                    dilation = dilation_size, groups = num_channels)
                mipmap[l:l+1, ...] = current_lvl
            # Convert from NCHW to NHWC
            mipmap = mipmap.permute(0, 2, 3, 1)
            texels = mipmap.contiguous()

        self.texels = texels
        self.uv_scale = uv_scale
