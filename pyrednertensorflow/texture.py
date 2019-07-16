import tensorflow as tf
import numpy as np
import pyrednertensorflow as pyredner
import enum
import math
# import pdb

class Texture:
    def __init__(self,
                 texels,
                 uv_scale = tf.constant([1.0, 1.0])):
        self.texels = texels   # torch.Size([32, 64, 3])
        if len(texels.shape) >= 2:
            # Build a mipmap for texels
            width = max(texels.shape[0], texels.shape[1]).value # Without value, it will have a type `Dimension`
            num_levels = math.ceil(math.log(width, 2) + 1)      # 7
            mipmap = tf.broadcast_to(texels, [num_levels, *texels.shape]) # 7 x 32 x 64 x 3
            """
            FIXME: Convert unsqueeze
            """
            if len(mipmap.shape) == 3:
                # mipmap.unsqueeze_(-1)
                mipmap = tf.expand_dims(mipmap, axis=-1)
            num_channels = mipmap.shape[-1]
            """NOTE: conv2d kernel axes

            torch: (outchannels,   in_channels / groups, kH,          kW)
            tf:    [filter_height, filter_width,         in_channels, out_channels]
            """
            # box_filter = torch.ones(num_channels, 1, 2, 2,
                # device = texels.device) / 4.0
            box_filter = tf.convert_to_tensor(np.ones([2, 2, num_channels, 1]), dtype=tf.float32) / 4.0

            # (PyTorch)Convert from HWC to NCHW 
            # base_level = texels.unsqueeze(0).permute(0, 3, 1, 2)
            # (TF) [batch, in_height, in_width, in_channels], i.e. NHWC 
            base_level = tf.transpose(tf.expand_dims(texels, axis=0), perm=[0, 3, 1, 2 ])

            #######################################################
            # NOTE: TEST purpose
            # if pyredner.DEBUG:
            #     self.width = width
            #     self.num_levels = num_levels
            #     self.mipmap_after_broadcast = mipmap
            #     self.num_channels = num_channels
            #     self.box_filter = box_filter
            #     self.base_level = base_level
            #######################################################
            

            mipmap = [base_level]
            prev_lvl = base_level
            for l in range(1, num_levels):
                dilation_size = 2 ** (l - 1)
                # Pad for circular boundary condition
                # This is slow. The hope is at some point PyTorch will support
                # circular boundary condition for conv2d

                # PyTorch version:
                desired_height = prev_lvl.shape[2] + dilation_size
                while prev_lvl.shape[2] < desired_height:
                    prev_lvl = tf.concat(
                        [
                            prev_lvl, 
                            prev_lvl[:,:,0:(desired_height - prev_lvl.shape[2])]
                        ], 2)

                desired_width = prev_lvl.shape[3] + dilation_size
                while prev_lvl.shape[3] < desired_width:
                    prev_lvl = tf.concat(
                        [prev_lvl, prev_lvl[:,:,:,0:dilation_size]], 
                        3)


                """NOTE: Torch conv data_format is NCHW. In Tensorflow, GPU supports
                NCHW but CPU supports only NHWC. Hence, we need to convert between
                NCHW and NHwC when we use CPU.
                """
                """NOTE: Current libxsmm and customized CPU implementations do 
                not yet support dilation rates larger than 1, i.e. we cannot use
                TF Conv2DCustomBackpropInputOp

                https://github.com/tensorflow/tensorflow/blob/7bc1c3c37ce4e591012f4325ab7a25ae387773c7/tensorflow/core/kernels/conv_grad_input_ops.cc#L300
                """
                # if pyredner.DEBUG:
                #     pyredner.imwrite(
                #         tf.squeeze(tf.transpose(prev_lvl, perm=[2,3,1,0])), 
                #         f'results/texture-tf/prev_lvl_before_{l}.png')
                    
                # if pyredner.use_gpu:
                    
                #     current_lvl = tf.nn.depthwise_conv2d(
                #         prev_lvl,
                #         box_filter,  # [filter_height, filter_width, in_channels, out_channels]
                #         dilations=[dilation_size,dilation_size],
                #         strides=[1,1,1,1],
                #         padding="VALID",   # No padding
                #         data_format="NCHW"
                #     )

                # else:
                prev_lvl = tf.transpose(prev_lvl, perm=[0,2,3,1])
                current_lvl = tf.nn.depthwise_conv2d(
                    prev_lvl,
                    box_filter,  # [filter_height, filter_width, in_channels, out_channels]
                    dilations=[dilation_size, dilation_size],
                    strides=[1,1,1,1],
                    padding="VALID",   # No padding
                    data_format="NHWC"
                )
                current_lvl = tf.transpose(current_lvl, [0,3,1,2])

                
                mipmap.append(current_lvl)
                prev_lvl = current_lvl
                # if pyredner.DEBUG:
                #     pyredner.imwrite(
                #         tf.transpose(tf.squeeze(prev_lvl), perm=[1,2,0]), 
                #         f'results/texture-tf/prev_lvl_{l}.png')
                #     self.level = prev_lvl

            mipmap = tf.concat(mipmap, 0)
            # Convert from NCHW to NHWC
            mipmap = tf.transpose(mipmap, perm=[0, 2, 3, 1])

            # Hard-copy a tensor
            # texels = mipmap.contiguous()
            # texels = tf.identity(mipmap)
            texels = mipmap

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
        out.uv_scale = state_dict['uv_scale'].numpy()
        return out
