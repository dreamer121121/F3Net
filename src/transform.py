import cv2
import torch

__all__ = ('Normalize','Resize','ToTensor')

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image


class ToTensor(object):
    def __call__(self, image):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        return image
