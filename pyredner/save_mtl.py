import pyredner
from typing import Union
import os
import torch

def save_mtl(m: pyredner.Material,
             filename: str):
    if filename[-4:] != '.mtl':
        filename = filename + '.mtl'
    path = os.path.dirname(filename)

    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'w') as f:
        f.write('newmtl mtl_1\n')

        if m.diffuse_reflectance is not None:
            texels = m.diffuse_reflectance.texels
            if len(texels.size()) == 1:
                f.write('Kd {} {} {}\n'.format(texels[0], texels[1], texels[2]))
            else:
                f.write('map_Kd Kd_texels.png\n')
                pyredner.imwrite(texels.data.cpu(), path + '/Kd_texels.png')
            #pyredner.imwrite(torch.randn(10, 10).abs(), path + '/tt.png')
