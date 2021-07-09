#!/usr/bin/python3
#coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image, mask):
        image = (image - self.mean)/self.std
        mask /= 255
        return image, mask


class RandomCrop(object):
    def __call__(self, image, mask):
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
        return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3]

    #def cal_shape(array):
    #    h, w = array.shape
    #    if h * w == 0:
    #        return True


class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2)==0:
            return image[:,::-1,:], mask[:, ::-1]
        else:
            return image, mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask,mode='train'):
        if mode == 'train':
            image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image, mask
        else:
            image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image,mask
class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        return image, mask


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
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
        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())

    def __getitem__(self, idx):
        name  = self.samples[idx]
        self.name = name
        #print('-----img  name ----',self.cfg.datapath+'/image/'+name+'.jpg')
        image = cv2.imread(self.cfg.datapath+'/image/'+name+'.jpg')[:,:,::-1].astype(np.float32)
        mask  = cv2.imread(self.cfg.datapath+'/mask/' +name+'.png', 0).astype(np.float32)


        # if len(image.shape) != 3
        # # print("img shape: ",image.shape)
        # # print("mask shape:",mask.shape)
        # # import sys
        # # sys.exit(0)

        shape = mask.shape

        if self.cfg.mode=='train':
            background, unknown, foreground = self.generate_trimap(mask)
            target = np.stack((mask, unknown, foreground), axis=2)
            image, target = self.normalize(image, target)
            image, target = self.randomcrop(image, target)
            image, target = self.randomflip(image, target)

            return image, target

        else:
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask, self.cfg.mode)
            image, mask = self.totensor(image, mask)
            return image, mask, shape, name

    def generate_trimap(self, mask):
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dilated = cv2.dilate(mask, dilation_kernel, iterations=5)

        erosion_kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask, erosion_kernel, iterations=3)

        background = np.zeros(mask.shape, dtype=np.uint8)
        background[dilated < 128] = 255

        unknown = np.zeros(mask.shape, dtype=np.uint8)
        unknown.fill(255)
        unknown[eroded > 128] = 0
        unknown[dilated < 128] = 0

        foreground = np.zeros(mask.shape, dtype=np.uint8)
        foreground[eroded > 128] = 255

        return background, unknown, foreground


    def collate(self, batch):
        size = [224, 256, 288, 320, 352, 384, 416][np.random.randint(0, 7)]
        image, target = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            try:
                image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
                target[i]  = cv2.resize(target[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            except:
                print("name: ",self.name)
                print("maks.shape: ",target[i].shape)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        target  = torch.from_numpy(np.stack(target, axis=0)).permute(0, 3, 1, 2)
        return image, target

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
