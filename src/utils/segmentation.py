"""
- Multiscale IoU -

Licensed under the MIT License. (see LICENSE for details)
Written by Azim Ahmadzadeh (Georgia State University)
"""


import numpy as np
import copy
import matplotlib.pyplot as plt
# from PIL import Image


class GridBasedSegmentation:
    """
    A simple class that does segmentation over a given image. In other words, it reduces the
    image-size by segmentizing the image into small cells and replacing each cell with 1 (if any
    non-zero value was present in that cell) or 0 (otherwise). To make the image size divisible
    by the given batch size, it does zero padding both horizontally and vertically.
    """

    def __init__(self, img, batch_size): #batch_size: box_size
        self.img = copy.deepcopy(img)
        self.batch_size = int(batch_size)
        self._image_h = img.shape[0]
        self._image_w = img.shape[1]
        self._padding = [0, 0, 0, 0]  # top, bottom, left, right

    def _pad_with(self, vector, pad_width, iaxis, kwargs):
        """
        pads a 2-D array.

        This implementation is inspired from 'https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html'.

        - Usage::
            a = np.arange(6).reshape((2, 3))
            np.pad(a, ((1,2),(3,0)), pad_with, padder=0)  # ((top, bottom), (left, right))

        :param vector:
        :param pad_width:
        :param iaxis:
        :param kwargs:
        :return:
        """
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        if pad_width[1] != 0:
            vector[-pad_width[1]:] = pad_value

    def shrink_by_segmentation(self, image):
        """
        divides the `image` into partitions using a gird-based segmentation (with each cell
        having sides of the size equal to `self.batch_size`), then replaces each cell with 1 if
        there is any non-zero value within that cell, and with 0, otherwise. The result will be a
        binary, 2D array, whose size is roughly equal to the size of the original image divide by
        `self.batch_size`. To make the image-width and image-height divisible by
        `self.batch_size` it adds some horizontal and/or vertical padding with zeros.

        :param image: the input 2-D array representing the image to be segmentized.
        :return: a 2D array of ones and zeros.
        """
        h = image.shape[0]
        w = image.shape[1]
        out = image.reshape(h // self.batch_size,
                            self.batch_size,
                            -1,
                            self.batch_size).swapaxes(1, 2).reshape(-1,
                                                                    self.batch_size,
                                                                    self.batch_size)
        s = np.einsum('ijk->i', out)
        s = s.reshape(h // self.batch_size, w // self.batch_size)
        s[s > 0] = 1
        return s

    def _pad_image(self):
        """
        Prepares the image in case any of its dimensions is not divisible by the class
        batch_size. It does so by added zero padding.

        :return:
        """
        img_padded = self.img

        # if vertical padding is needed, add rows to 'top'
        if self._image_h % self.batch_size != 0:
            pad_height = self.batch_size - (self._image_h % self.batch_size)
            self._padding[0] = pad_height
            img_padded = np.pad(self.img,
                                ((pad_height, 0), (0, 0)),
                                self._pad_with, padder=0)
        # if horizontal padding is needed, add rows to 'left'
        if self._image_w % self.batch_size != 0:
            pad_width = self.batch_size - (self._image_w % self.batch_size)
            self._padding[2] = pad_width
            img_padded = np.pad(img_padded,
                                ((0, 0), (pad_width, 0)),
                                self._pad_with, padder=0)
        return img_padded

    def segmentize(self):
        """
        this methods wraps all the steps needs in box-counting: it pads the image if needed,

        :return:
        """
        img_padded = self._pad_image()
        img_chunked = self.shrink_by_segmentation(img_padded)
        return img_chunked

