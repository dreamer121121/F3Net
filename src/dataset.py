#!/usr/bin/python3
#coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image):
        image = (image - self.mean)/self.std
        return image


class RandomCrop(object):
    def __call__(self, image):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw #cunzai bug quedingyixia zai shen mo qingkaungxia jiang caihou mask.shape() h*w =0
        #tmp_mask = mask[p0:p1,p2:p3]
        #if self.cal_shape(tmp_mask):
        #    return image[p0:p1,p2:p3, :], mask
        #else:
        return image[p0:p1,p2:p3, :]

    #def cal_shape(array):
    #    h, w = array.shape
    #    if h * w == 0:
    #        return True


class RandomFlip(object):
    def __call__(self, image):
        if np.random.randint(2)==0:
            return image[:,::-1,:]
        else:
            return image


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask,mode='train'):
        if mode == 'train':
            image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image
        else:
            image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image


class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        return image


class Rotate(object):
    def __call__(self, image):
        angle = [90, 180, 270][np.random.randint(0, 3)]

        return self.rotate(image, angle)

    def rotate(self, image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)

        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[138.0536, 130.8389, 121.07198]]])
        self.std    = np.array([[[ 71.3735, 69.87299, 73.78777]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None

#f = open('error_sample.txt','w')
########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg        = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize     = Resize(352, 352)
        self.totensor   = ToTensor()
        self.rotate     = Rotate()
        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())

    def __getitem__(self, idx):
        name  = self.samples[idx]
        self.name = name
        #print('-----img  name ----',self.cfg.datapath+'/image/'+name+'.jpg')
        image = cv2.imread(self.cfg.datapath+'/image/'+name+'.jpg')[:,:,::-1].astype(np.float32)

        image = self.normalize(image)
        image = self.randomcrop(image)
        image = self.rotate(image)
        image = self.randomflip(image)
        return image

    def collate(self, batch):
        size = [224, 256, 288, 320, 352, 384, 416][np.random.randint(0, 7)]
        image = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            try:
                image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            except:
                print("name: ", self.name)

        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        return image

    def __len__(self):
        return len(self.samples)


########################### Testing Script ###########################
if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    cfg  = Config(mode='train', datapath='../data/DUTS')
    data = Data(cfg)
    for i in range(1000):
        image, mask = data[i]
        image       = image*cfg.std + cfg.mean
        plt.subplot(121)
        plt.imshow(np.uint8(image))
        plt.subplot(122)
        plt.imshow(mask)
        input()
