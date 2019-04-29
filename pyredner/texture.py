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
        self.texels = texels
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

            # Convert from HWC to NCHW
            base_level = texels.unsqueeze(0).permute(0, 3, 1, 2)
            mipmap = [base_level]
            prev_lvl = base_level
            for l in range(1, num_levels):
                dilation_size = 2 ** (l - 1)
                # Pad for circular boundary condition
                # This is slow. The hope is at some point PyTorch will support
                # circular boundary condition for conv2d
                desired_height = prev_lvl.shape[2] + dilation_size
                while prev_lvl.shape[2] < desired_height:
                    prev_lvl = torch.cat([prev_lvl, prev_lvl[:,:,0:(desired_height - prev_lvl.shape[2])]], dim=2)
                desired_width = prev_lvl.shape[3] + dilation_size
                while prev_lvl.shape[3] < desired_width:
                    prev_lvl = torch.cat([prev_lvl, prev_lvl[:,:,:,0:dilation_size]], dim=3)
                current_lvl = torch.nn.functional.conv2d(\
                    prev_lvl, box_filter,
                    dilation = dilation_size,
                    groups = num_channels)
                mipmap.append(current_lvl)
                prev_lvl = current_lvl

            mipmap = torch.cat(mipmap, 0)
            # Convert from NCHW to NHWC
            mipmap = mipmap.permute(0, 2, 3, 1)
            texels = mipmap.contiguous()

        self.mipmap = texels
        self.uv_scale = uv_scale

    def state_dict(self):
        return {
            'texels': self.texels,
            'mipmap': self.mipmap,
            'uv_scale': self.uv_scale
        }
    @classmethod
    def load_state_dict(cls, state_dict):
        out = cls.__new__(Texture)
        out.texels = state_dict['texels']
        out.mipmap = state_dict['mipmap']
        out.uv_scale = state_dict['uv_scale'].to(torch.device('cpu'))
        return out
