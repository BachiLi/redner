import torch
import numpy as np
import pyredner
import torch
import enum
import math
from typing import Optional

class Texture:
    """
        Representing a texture and its mipmap.

        Args
        ====
        texels: torch.Tensor
            a float32 tensor with size C or [height, width, C]
        uv_scale: Optional[torch.Tensor]
            scale the uv coordinates when mapping the texture
            a float32 tensor with size 2
    """

    def __init__(self,
                 texels: torch.Tensor,
                 uv_scale: Optional[torch.Tensor] = None):
        if uv_scale is None:
            uv_scale = torch.tensor([1.0, 1.0], device = pyredner.get_device())
        assert(texels.dtype == torch.float32)
        assert(uv_scale.dtype == torch.float32)
        assert(uv_scale.is_contiguous())
        self._texels = texels
        self.uv_scale = uv_scale
        self.generate_mipmap()

    def generate_mipmap(self):
        texels = self._texels
        if len(texels.shape) >= 2:
            # Build a mipmap for texels
            width = max(texels.shape[0], texels.shape[1])
            num_levels = min(math.ceil(math.log(width, 2) + 1), 8)
            num_channels = texels.shape[2]
            box_filter = torch.ones(num_channels, 1, 2, 2,
                device = texels.device) / 4.0

            # Convert from HWC to NCHW
            mipmap = [texels.contiguous()]
            base_level = texels.unsqueeze(0).permute(0, 3, 1, 2)
            prev_lvl = base_level
            for l in range(1, num_levels):
                # Pad for circular boundary condition
                current_lvl = torch.nn.functional.pad(\
                    input = prev_lvl,
                    pad = (0, 1, 0, 1),
                    mode = 'circular')
                # Convolve with a box filter
                current_lvl = torch.nn.functional.conv2d(\
                    current_lvl, box_filter,
                    groups = num_channels)
                # Downsample
                next_size = (max(current_lvl.shape[2] // 2, 1),
                             max(current_lvl.shape[3] // 2, 1))
                current_lvl = torch.nn.functional.interpolate(\
                    current_lvl, size = next_size, mode = 'area')
                # NCHW -> CHW -> HWC
                mipmap.append(current_lvl.squeeze(0).permute(1, 2, 0).contiguous())
                prev_lvl = current_lvl
        else:
            mipmap = [texels]

        self.mipmap = mipmap

    @property
    def texels(self):
        return self._texels

    @texels.setter
    def texels(self, value):
        self._texels = value
        self.generate_mipmap()

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
