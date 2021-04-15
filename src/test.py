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

import pydensecrf.densecrf as dcrf


class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=path, snapshot=args.model, mode='test')
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
            total = datetime.datetime(1999,1,1)

            for user_img,image, mask, shape, name in self.loader:
                #image.shape (1,3,352,352)
                #shape: init img shape ,which is for pre_mask to match the size of init img

                image = image.cuda().float()
                start = datetime.datetime.now()
                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image, shape)
                total += datetime.datetime.now()-start
                print("inference time: ",(total-datetime.datetime(1999,1,1))/cnt)
                out   = out2u
                pred  = (torch.sigmoid(out[0,0])*255).cpu().numpy()

                Q = None
                if args.crf:
                    Q = self.dense_crf(user_img.numpy().astype(np.uint8),pred)
                head  = '../eval/maps_crf/F3Net/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))
                # import sys
                # sys.exit(0)
                #print("inference time: ",(datetime.datetime.now()-start)/cnt)
                cnt +=1

    @torch.no_grad()
    def save_fig(self):

        normalize = Normalize(mean=self.cfg.mean, std=self.cfg.std)
        resize = Resize(352, 352)
        totensor = ToTensor()

        fr = open(self.path+'/test.txt','r')

        file_list = fr.readlines()

        fr.close()

        import datetime
        cnt = 1
        total = datetime.datetime(1999, 1, 1)
        for name in file_list:

            name = name.replace('\n','')
            user_image = cv2.imread(self.path+'/image/'+name+'.jpg')

            start = datetime.datetime.now()
            input_data = user_image[:,:,::-1].astype(np.float32)
            shape = [torch.tensor([int(input_data.shape[0])]),torch.tensor([int(input_data.shape[1])])]

            input_data = normalize(input_data)
            input_data = resize(input_data)
            input_data = totensor(input_data)
            input_data = input_data[np.newaxis,:,:,:]

            image = input_data.cuda().float()
            # user_image = torch.from_numpy(user_image).cuda().float()
            # alpha = torch.ones(user_image.size()[0],user_image.size()[1],1).cuda()*255
            # user_image = torch.cat((user_image,alpha),dim=2)


            out1u, out2u, out2r, out3r, out4r, out5r = self.net(image, shape)
            out = out2u

            pred = (torch.sigmoid(out[0, 0]))
            if args.crf:
                Q = self.dense_crf(user_image.astype(np.uint8), pred.cpu().numpy())
                print('--Q--', Q)
                cv2.imwrite('./crf_test.png',np.round(Q*255))
                import sys
                sys.exit(0)

            mask = pred.unsqueeze(dim=2)

            outimg = (mask*user_image).detach().cpu().numpy()

            # for w in range(outimg.shape[0]):
            #     for h in range(outimg.shape[1]):
            #         if all(outimg[w][h] == [0, 0, 0]):
            #             alpha[w][h] = 0
            #         else:
            #             alpha[w][h] = 255  # 看看能否优化速度
            #
            # outimg = np.dstack([outimg, alpha])

            head  = '../eval/results/F3Net/'+ self.cfg.datapath.split('/')[-1]

            if not os.path.exists(head):
                os.makedirs(head)

            cv2.imwrite(head+'/'+name+'.png', np.round(outimg))
            total += datetime.datetime.now() - start
            print("inference time: ", (total - datetime.datetime(1999, 1, 1)) / cnt)
            cnt += 1


    def dense_crf(slef,img, output_probs):  # img为输入的图像，output_probs是经过网络预测后得到的结果
        h = output_probs.shape[0]  # 高度
        w = output_probs.shape[1]  # 宽度

        output_probs = np.expand_dims(output_probs, 0)
        output_probs = np.append(1 - output_probs, output_probs, axis=0)

        d = dcrf.DenseCRF2D(w, h, 2)  # NLABELS=2两类标注，车和不是车
        U = -np.log(output_probs)  # 得到一元势
        U = U.reshape((2, -1))  # NLABELS=2两类标注
        U = np.ascontiguousarray(U)  # 返回一个地址连续的数组
        img = np.ascontiguousarray(img)

        d.setUnaryEnergy(U)  # 设置一元势

        img = img.squeeze()
        d.addPairwiseGaussian(sxy=20, compat=3)  # 设置二元势中高斯情况的值
        d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)  # 设置二元势众双边情况的值

        Q = d.inference(5)  # 迭代5次推理
        Q = np.argmax(np.array(Q), axis=0).reshape((h, w))  # 得列中最大值的索引结果

        return Q

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
    parser.add_argument('--model',
                        type=str)
    parser.add_argument('--crf',
                        action='store_true')
    args = parser.parse_args()
    # for path in ['../data/allbody_base']:
    #     print("path:",path)
    t = Test(dataset, F3Net, '../data/'+args.dataset)
    if args.mode == 'mask':
        t.save()
    elif args.mode == 'fig':
        t.save_fig()
    # t.show()
