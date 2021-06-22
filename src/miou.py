
"""
- Multiscale IoU -

Licensed under the MIT License. (see LICENSE for details)
Written by Azim Ahmadzadeh (Georgia State University)
"""


import numpy as np
import copy
from scipy.stats import linregress
from metric import *
from edge_detection import sobel
from utils.segmentation import GridBasedSegmentation


class MIoU(IMetric):
    """
    This is the main class that implements MIoU.
    """
    def __init__(self, boxsizes, edge_only):
        """
        Class constructor.

        :param boxsizes: a list or 1-D array of the side lengths to be used as the grid
        cell. The values of this collection determines the resolutions based on which
        MIoU is computed.

        :param edge_only: the default value is `True`. Usage of `False` is only experimental.
        """
        self.edge_only = edge_only
        self.boxsizes = np.array(boxsizes)
        self.aiou = 0
        self.box_counts_intersect = []
        self.box_counts_m1 = []
        self.ratios = []
        self.slope = 0
        self.area = 0

    def measure(self, mask1, mask2, *args, **kwargs):
        """
        computes the metric for the two given regions. Use 'miou/utils/mask_loader.py'
        to properly load masks as binary arrays for `mask1` and `mask2`.

        :param mask1: mask segmentation of the reference (ground-truth) object.
        :param mask2: mask segmentation of the target (detected) object.
        """
        m1 = copy.deepcopy(mask1)
        m2 = copy.deepcopy(mask2)

        self.box_counts_intersect = []
        self.box_counts_m1 = []
        self.ratios = []

        if self.edge_only:
            m1 = sobel.get_edges(m1)
            m2 = sobel.get_edges(m2)

        for boxsize in self.boxsizes:
            gbs = GridBasedSegmentation(m1, boxsize)
            s1 = gbs.segmentize()
            gbs = GridBasedSegmentation(m2, boxsize)
            s2 = gbs.segmentize()

            # Box counting on m1 -------------------
            self.box_counts_m1.append(np.sum(s1) + 1)
            # --------------------------------------

            # Box counting on intersect(m1, m2) ----
            intersect = np.logical_and(s1, s2)
            self.box_counts_intersect.append(np.sum(intersect) + 1)
            # --------------------------------------

            self.ratios.append(self.box_counts_intersect[-1] / self.box_counts_m1[-1]) #calculate r metircs

        normalized_boxsizes = normalize_boxsizes(self.boxsizes)
        self.slope, _, _, _, _ = linregress(normalized_boxsizes, self.ratios)
        self.aiou = np.abs(self.slope)

        self.area = integral_trapezoidal(self.ratios, dx=1/(len(self.boxsizes)-1))
        return self.area


def normalize_boxsizes(boxsizes):
    """
    normalizes the values of `boxsizes` using a 0-1 transformation.

    :param boxsizes: a list or 1-D array of box sizes.
    :return: A 1-D array of transformed values.
    """
    from sklearn.preprocessing import MinMaxScaler

    if isinstance(boxsizes, list):
        b = np.array(boxsizes).reshape((-1, 1))  # 1D to 2D
    else:
        b = copy.deepcopy(boxsizes).reshape((-1, 1))  # 1D to 2D

    bb = MinMaxScaler().fit_transform(b)  # 0-1 normalize
    return bb.flatten()  # flatten and return


def integral_simpson(y, dx):
    """
    calculates area under the curves using Simpson's method.

    :param y: values of the curve on the Y axis.
    :param dx: spacing (for binning) on the X axis.

    :return: the area under the curve.
    """
    from scipy.integrate import simps
    area = simps(y, dx=dx)
    return area


def integral_trapezoidal(y, dx):
    """
    Calculates area under the curves using Trapezoids.

    :param y: values of the curve on the Y axis.
    :param dx: spacing (for binning) on the X axis.

    :return: the area under the curve.
    """
    from numpy import trapz
    area = trapz(y, dx=dx)
    return area
