#!/usr/bin/python3
#coding=utf-8

import sys
import os
import datetime

sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from net  import F3Net
import shutil
import argparse

from saliency_metrics import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm


log_stream = open('train.log','a')

global_step = 0
best_mae = float('inf')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval',
                        action='store_true',
                        help='whether to eval a model')
    parser.add_argument('--resume',
                        type=str,
                        help='resume from pretrained model to fintune')
    args = parser.parse_args()

    return args


def save_checkpoints(state, is_best, epoch,cfg):
    torch.save(state, '%s/%s_%dcheckpoint.pth.tar' % (os.path.join(cfg.savepath), 'model',epoch))
    if is_best:
        shutil.copyfile('%s/%s_%dcheckpoint.pth.tar' % (os.path.join(cfg.savepath), 'model',epoch),'%s/%s_best.pth.tar' % (os.path.join(cfg.savepath), 'model'))


def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def main(Dataset,Network):

    ##parse args
    args = parse_args()

    train_cfg = Dataset.Config(datapath='../data/DUTS/', savepath='./out', snapshot=args.resume, mode='train', batch=32,
                            lr=0.05, momen=0.9, decay=5e-4, epochs=32)

    eval_cfg =  Dataset.Config(datapath='../data/DUTS/', mode='test',eval_freq=1)

    train_data = Dataset.Data(train_cfg)

    eval_data = Dataset.Data(eval_cfg)

    train_dataloader = DataLoader(train_data,collate_fn=train_data.collate, batch_size=train_cfg.batch, shuffle=True, num_workers=16)
    eval_dataloader =  DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=16)

    net    = Network(train_cfg)
    net.train(True)

    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=train_cfg.lr, momentum=train_cfg.momen, weight_decay=train_cfg.decay, nesterov=True)
    sw             = SummaryWriter(train_cfg.savepath)

    net = nn.DataParallel(net,device_ids=[0,1,2,3])
    net.cuda()

    if args.eval:
        evaluate(net,eval_dataloader)

    for epoch in range(train_cfg.epochs):
        log_stream.write('='*30+'Epoch: '+str(epoch+1)+'='*30+'\n')
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(train_cfg.epochs+1)*2-1))*train_cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(train_cfg.epochs+1)*2-1))*train_cfg.lr

        train(net,optimizer,train_dataloader,sw,epoch,train_cfg)

        if (epoch + 1) % eval_cfg.eval_freq == 0 or epoch == train_cfg.epochs - 1:

            mae = evaluate(net,eval_dataloader)

            global best_mae
            is_best = mae < best_mae
            best_mae = min(mae,best_mae)

            save_checkpoints({
                'epoch':epoch+1,
                'state_dict':net.state_dict(),
                'best_mae':best_mae,
            }, is_best,epoch,train_cfg)

            log_stream.write('Valid MAE: {:.4f} \n'.format(mae))
            log_stream.flush()


def evaluate(net,loader):
    mae = cal_mae()

    net.eval()

    with torch.no_grad():
        for image, mask, shape, name in loader:
            image = image.cuda().float()
            out1u, out2u, out2r, out3r, out4r, out5r = net(image, shape)
            out = out2u
            pred = (torch.sigmoid(out[0, 0])).cpu().numpy()

            mask = np.asarray(mask,np.float32)
            mask /= (mask.max()+1e-8)
            
            mask[mask > 0.5] = 1
            mask[mask != 1] = 0
            
            pred = np.array(pred)
            if pred.max() == pred.min():
                pred = pred/255
            else:
                pred = (pred - pred.min()) / (pred.max() - pred.min())
            mae.update(pred,mask)
        Mae = mae.show()

    return Mae

def train(net,optimizer,loader,sw,epoch,cfg):
    net.train()
    for step, (image, mask) in enumerate(loader):
        image, mask = image.cuda().float(), mask.cuda().float()

        image,mask_a,mask_b,lam = mixup_data(image,mask,alpha=1.0,use_cuda=True)
        # print(image.shape) #(32,3,320,320)
        # print(mask.shape) #(32,1,320,320)
        # import sys
        # sys.exit(0)
        out1u, out2u, out2r, out3r, out4r, out5r = net(image)
        loss1u = mixup_criterion(structure_loss, out1u, mask_a, mask_b, lam)
        loss2u = mixup_criterion(structure_loss, out2u, mask_a, mask_b, lam)

        loss2r = mixup_criterion(structure_loss, out2r, mask_a, mask_b, lam)
        loss3r = mixup_criterion(structure_loss, out3r, mask_a, mask_b, lam)
        loss4r = mixup_criterion(structure_loss, out4r, mask_a, mask_b, lam)
        loss5r = mixup_criterion(structure_loss, out5r, mask_a, mask_b, lam)
        loss   = (loss1u+loss2u)/2+loss2r/2+loss3r/4+loss4r/8+loss5r/16

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## log
        global global_step
        global_step += 1
        sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
        sw.add_scalars('loss', {'loss1u':loss1u.item(), 'loss2u':loss2u.item(), 'loss2r':loss2r.item(), 'loss3r':loss3r.item(), 'loss4r':loss4r.item(), 'loss5r':loss5r.item()}, global_step=global_step)
        if step%10 == 0:
            log_stream.write('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f \n'%(datetime.datetime.now(), global_step, epoch+1, cfg.epochs, optimizer.param_groups[0]['lr'], loss.item()))
            log_stream.flush()


if __name__=='__main__':
    main(dataset, F3Net)
