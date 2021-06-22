"""
- Multiscale IoU -

Licensed under the MIT License. (see LICENSE for details)
Written by Azim Ahmadzadeh (Georgia State University)
"""

import abc
import numpy as np


class IMetric():
    @abc.abstractmethod
    def measure(self, mask1, mask2, *args, **kwargs):
        pass
