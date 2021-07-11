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
        (w, h, c) = image.shape
        if w >= h:
            r = 800/image.shape[0]
            image = cv2.resize(image, dsize=(800, h * r), interpolation=cv2.INTER_LINEAR)
        elif h > w:
            r = 800/image.shape[1]
            image = cv2.resize(image, dsize=(w * r, 800), interpolation=cv2.INTER_LINEAR)
        return image

class ToTensor(object):
    def __call__(self, image):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        return image
