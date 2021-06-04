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
from transform import *


class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=path, snapshot='./out/model_best.pth.tar', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
        self.path = path

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
<<<<<<< Updated upstream
            total = datetime.datetime(1999,1,1)
=======
            total = datetime.datetime(1999, 1, 1)
            print('--type(self.loader--)', type(self.loader))

>>>>>>> Stashed changes
            for image, mask, shape, name in self.loader:
                # print(np.where(mask>0.5))
                # import sys
                # sys.exit(0)
                #image.shape (1,3,352,352)
                #shape: init img shape ,which is for pre_mask to match the size of init img
<<<<<<< Updated upstream
                image = image.cuda().float()
=======
                print(image)
                image = image.to('cuda:1').float()
>>>>>>> Stashed changes
                start = datetime.datetime.now()
                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image, shape)
                total += datetime.datetime.now()-start
                print("inference time: ", (total-datetime.datetime(1999, 1, 1))/cnt)
                out   = out2u
                pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                head  = '../eval/maps/F3Net/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))
                #print("inference time: ",(datetime.datetime.now()-start)/cnt)
                cnt +=1

    def save_fig(self):

        normalize = Normalize(mean=self.cfg.mean, std=self.cfg.std)
        resize = Resize(352, 352)
        totensor = ToTensor()

        fr = open(self.path+'/test.txt','r')

        file_list = fr.readlines()

        fr.close()

<<<<<<< Updated upstream
=======
        import datetime
        cnt = 1
        total = datetime.datetime(1999, 1, 1)
        import segmentation_refinement as refine
        refiner = refine.Refiner(device='cuda:1')
>>>>>>> Stashed changes
        for name in file_list:

            name = name.replace('\n','')
            user_image = cv2.imread(self.path+'/image/'+name+'.jpg')
            input_data = user_image[:,:,::-1].astype(np.float32)
            shape = [torch.tensor([int(input_data.shape[0])]),torch.tensor([int(input_data.shape[1])])]

            input_data = normalize(input_data)
            input_data = resize(input_data)
            input_data = totensor(input_data)
            input_data = input_data[np.newaxis,:,:,:]

<<<<<<< Updated upstream
            image = input_data.cuda().float()

=======
            image = input_data.to('cuda:1').float()
>>>>>>> Stashed changes
            out1u, out2u, out2r, out3r, out4r, out5r = self.net(image, shape)
            out = out2u

<<<<<<< Updated upstream
            pred = (torch.sigmoid(out[0, 0]) * 255).cpu().detach().numpy()
=======
            mask = (torch.sigmoid(out2u[0, 0]) * 255).cpu().numpy()
            # if args.crf:
            #     Q = self.dense_crf(user_image.astype(np.uint8), pred.cpu().numpy())
            #     print('--Q--', Q)
            #     cv2.imwrite('./crf_test.png',np.round(Q*255))
            #     import sys
            #     sys.exit(0)
            if args.erd:
                # ret, img_thr = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
                mask = np.where(mask>178, 255, mask)
                mask = np.where(mask<76, 0, mask)
                kernal = np.ones((20, 20), np.uint8)
                mask = cv2.erode(mask, kernal, iterations=1)
                # print(mask.shape)
                # print(user_image.shape)
                # mask = refiner.refine(user_image, mask, fast=True, L=900)

            alpha = np.ones((w, h, 1))*255
            user_image = np.dstack((user_image, alpha))

            outimg = np.multiply(user_image, mask[:, :, np.newaxis]/255)

            del mask, out1u, out2u, out2r, out3r, out4r, out5r
>>>>>>> Stashed changes


            mask = np.zeros(shape=(pred.shape[0],pred.shape[1],3))
            pred = np.round(pred) #network output

            mask[:, :, 0] = pred[:, :]
            mask[:, :, 1] = pred[:, :]
            mask[:, :, 2] = pred[:, :]

            outimg = np.where(mask > 127, user_image, 0)

            alpha = np.zeros((outimg.shape[0], outimg.shape[1])).astype(int)

            for w in range(outimg.shape[0]):
                for h in range(outimg.shape[1]):
                    if all(outimg[w][h] == [0, 0, 0]):
                        alpha[w][h] = 0
                    else:
                        alpha[w][h] = 255  # 看看能否优化速度

            outimg = np.dstack([outimg, alpha])

            head  = '../eval/result/F3Net/'+ self.cfg.datapath.split('/')[-1]

            if not os.path.exists(head):
                os.makedirs(head)

            cv2.imwrite(head+'/'+name+'.png', outimg)


if __name__=='__main__':
    #for path in ['../data/ECSSD', '../data/PASCAL-S', '../data/DUTS', '../data/HKU-IS', '../data/DUT-OMRON']:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        type=str,
                        default='mask')
    parser.add_argument('--dataset',
                        type=str,
                        )
    args = parser.parse_args()
    # for path in ['../data/allbody_base']:
    #     print("path:",path)
    t = Test(dataset, F3Net, args.dataset)
    if args.mode == 'mask':
        t.save()
    elif args.mode == 'fig':
        t.save_fig()
    # t.show()
