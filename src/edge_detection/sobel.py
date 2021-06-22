"""
 - Multiscale IoU -

Licensed under the MIT License. (see LICENSE for details)
Written by Azim Ahmadzadeh (Georgia State University)
"""


from scipy import ndimage
import numpy as np


def get_edges(mask):
    """
    This is a simple implementation of Sobel Edge Detection algorithm that
    is based on the Sobel filter from scipy.ndimage module.

    :param mask: a binary mask of an object whose edges are of interest.
    :return: a binary mask of 1's as edges and 0's as background.
    """
    sx = ndimage.sobel(mask, axis=0, mode='constant')
    sy = ndimage.sobel(mask, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    sob[sob > 0] = 1
    sob[sob <= 0] = 0
    return sob