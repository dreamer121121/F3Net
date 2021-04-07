#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from net  import F3Net


class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=path, snapshot='./out/model-100', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def show(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                print('image.shape: ',image.shape)
                import sys
                sys.exit(0)
                image, mask = image.cuda().float(), mask.cuda().float()
                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image)
                out = out2u

                plt.subplot(221)
                plt.imshow(np.uint8(image[0].permute(1,2,0).cpu().numpy()*self.cfg.std + self.cfg.mean))
                plt.subplot(222)
                plt.imshow(mask[0].cpu().numpy())
                plt.subplot(223)
                plt.imshow(out[0, 0].cpu().numpy())
                plt.subplot(224)
                plt.imshow(torch.sigmoid(out[0, 0]).cpu().numpy())
                plt.show()
                input()
    
    def save(self):
        with torch.no_grad():
            import datetime
            #start = datetime.datetime.now()
            cnt = 1
            total = datetime.datetime(1999,1,1)
            for image, mask, shape, name in self.loader:
                print(np.where(mask>0.5))
                import sys
                sys.exit(0)
                #image.shape (1,3,352,352)
                #shape: init img shape ,which is for pre_mask to match the size of init img

                image = image.cuda().float()
                start = datetime.datetime.now()
                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image, shape)
                total += datetime.datetime.now()-start
                print("inference time: ",(total-datetime.datetime(1999,1,1))/cnt)
                out   = out2u
                pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                head  = '../eval/maps/F3Net/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))
                #print("inference time: ",(datetime.datetime.now()-start)/cnt)
                cnt +=1

    def save_fig(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                #image.shape (1,3,352,352)
                #shape: init img shape ,which is for pre_mask to match the size of init img

                image = image.cuda().float()
                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image, shape)
                out   = out2u
                pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                pred = np.round(pred)

                user_image = cv2.resize(image,shape)

                res = np.zeros(shape=(shape[0], pred.shape[1], 3))
                res[:, :, 0] = pred[:, :]
                res[:, :, 1] = pred[:, :]
                res[:, :, 2] = pred[:, :]

                outimg = np.where(res > 127, user_image, 255)

                head  = '../eval/results/F3Net/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)

                cv2.imwrite(head+'/'+name[0]+'.png', outimg)

if __name__=='__main__':
    #for path in ['../data/ECSSD', '../data/PASCAL-S', '../data/DUTS', '../data/HKU-IS', '../data/DUT-OMRON']:
     for path in ['../data/DUTS']:
        print("path:",path)
        t = Test(dataset, F3Net, path)
        t.save()
        # t.show()
