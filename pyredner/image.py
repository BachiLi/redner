import numpy as np
import skimage
import skimage.io
import torch
import os
import imageio

def imwrite(img: torch.Tensor,
            filename: str,
            gamma: float = 2.2,
            normalize: bool = False):
    """
        write img to filename

        Args
        ====
        img: torch.Tensor
            with size [height, width, channel]
        filename: str

        gamma: float
            if the image is not an OpenEXR file, apply gamma correction
        normalize:
            normalize img to the range [0, 1] before writing
    """

    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)
    
    img = img.data.numpy()
    if normalize:
        img_rng = np.max(img) - np.min(img)
        if img_rng > 0:
            img = (img - np.min(img)) / img_rng
    if filename[-4:] == '.exr':
        imageio.plugins.freeimage.download()
        imageio.imwrite(filename, img)
    else:
        skimage.io.imsave(filename,
            (np.power(np.clip(img, 0.0, 1.0), 1.0/gamma) * 255).astype(np.uint8))

def imread(filename: str,
           gamma: float = 2.2):
    """
        read img from filename

        Args
        ====
        filename: str

        gamma: float
            if the image is not an OpenEXR file, apply gamma correction

        Returns
        =======
        torch.Tensor
            a float32 tensor with size [height, width, channel]
    """

    if (filename[-4:] == '.exr'):
        imageio.plugins.freeimage.download()
        return torch.from_numpy(imageio.imread(filename).astype(np.float32))
    else:
        im = skimage.io.imread(filename)
        if im.ndim == 2:
            im = np.stack([im, im, im], axis=-1)
        elif im.shape[2] == 4:
            im = im[:, :, :3]
        return torch.from_numpy(np.power(\
            skimage.img_as_float(im).astype(np.float32), gamma))
