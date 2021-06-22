"""
- Multiscale IoU -

Licensed under the MIT License. (see LICENSE for details)
Written by Azim Ahmadzadeh (Georgia State University)
"""

from PIL import Image
import numpy as np


def load_mask(path: str, threshold: int = 125):
    """
    loads an image as grayscale and convert it to a binary mask. If `threshold` is not
    given, every pixel with gray tone equal to or greater than 125 would be set to 1,]
    and below that would be set to 0.

    :param path: path to the image.
    :param threshold: if given will be used to set the binary mapping.
    :return:
    """

    mask = Image.open(path).convert('L')
    mask = np.asarray(mask, dtype=np.int)
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0
    mask[mask == 255] = 1
    return mask

