#!/usr/bin/python3
# coding=utf-8

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
from net import F3Net
import shutil
import argparse

from saliency_metrics import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm
from .smoothing import gaussian_blur

log_stream = open('train_cortor.log', 'a')

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
    parser.add_argument('--dataset',
                        type=str,
                        help='dataset used for train')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32)
    parser.add_argument('--epochs',
                        type=int,
                        default=32)
    parser.add_argument('--start',
                        type=int,
                        default=0)
    parser.add_argument('--lr',
                        type=float,
                        default=0.05)
    parser.add_argument('--decay',
                        type=float,
                        default=5e-4)
    args = parser.parse_args()

    return args


def save_checkpoints(state, is_best, epoch, cfg):
    torch.save(state, '%s/%s_%dcheckpoint.pth.tar' % (os.path.join(cfg.savepath), 'model', epoch))
    if is_best:
        shutil.copyfile('%s/%s_%dcheckpoint.pth.tar' % (os.path.join(cfg.savepath), 'model', epoch),
                        '%s/%s_best.pth.tar' % (os.path.join(cfg.savepath), 'model'))



def structure_loss(pred, target):

    def masked_l1_loss(y_hat, y, mask):
        loss = F.l1_loss(y_hat, y, reduction='none')
        loss = (loss * mask.float()).sum()
        non_zero_elements = mask.sum()
        return loss / non_zero_elements

    shape = target.shape
    mask = target[:, 0] # target:N, W, H
    smoothed_mask = gaussian_blur(
        mask.unsqueeze(dim=1), (9, 9), (2.5, 2.5)).squeeze(dim=1)
    unknown_mask = target[:, 1]

    l1_mask = torch.ones(mask.shape)
    l1_details_mask = torch.zeros(mask.shape)

    for idx in range(shape[0]):
        l1_details_mask[idx] = unknown_mask[idx]

    loss = 0
    loss += 2 * masked_l1_loss(F.sigmoid(pred), mask, l1_mask)
    # this loss should give some learning signals to focus on unknown areas
    loss += 3 * masked_l1_loss(F.sigmoid(pred), mask, l1_details_mask)
    # i'm not quite sure if this loss gives the right incentive, the idea
    # is to blur the segmentation mask a bit to reduce background bleeding
    # caused by bad labels, preliminary results seem to be quite ok.
    loss += F.mse_loss(F.sigmoid(pred), smoothed_mask)

    return loss



def main(Dataset, Network):
    ##parse args
    args = parse_args()

    train_cfg = Dataset.Config(datapath='../data/' + args.dataset, savepath='./out_cortor', snapshot=args.resume,
                               mode='train',
                               batch=args.batch_size,
                               lr=args.lr, momen=0.9, decay=args.decay, epochs=args.epochs, start=args.start)

    eval_cfg = Dataset.Config(datapath='../data/' + args.dataset, mode='test', eval_freq=1)

    train_data = Dataset.Data(train_cfg)

    eval_data = Dataset.Data(eval_cfg)

    train_dataloader = DataLoader(train_data, collate_fn=train_data.collate, batch_size=train_cfg.batch, shuffle=True,
                                  num_workers=16)
    eval_dataloader = DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=16)

    net = Network(train_cfg)
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
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=train_cfg.lr, momentum=train_cfg.momen,
                                weight_decay=train_cfg.decay, nesterov=True)
    sw = SummaryWriter(train_cfg.savepath)

    net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    net.cuda()

    if args.eval:
        evaluate(net, eval_dataloader)

    for epoch in range(train_cfg.start, train_cfg.epochs):
        log_stream.write('=' * 30 + 'Epoch: ' + str(epoch + 1) + '=' * 30 + '\n')
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (train_cfg.epochs + 1) * 2 - 1)) * train_cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (train_cfg.epochs + 1) * 2 - 1)) * train_cfg.lr

        train(net, optimizer, train_dataloader, sw, epoch, train_cfg)

        if (epoch + 1) % eval_cfg.eval_freq == 0 or epoch == train_cfg.epochs - 1:
            mae, loss = evaluate(net, eval_dataloader)

            global best_mae
            is_best = mae < best_mae
            best_mae = min(mae, best_mae)

            save_checkpoints({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_mae': best_mae,
            }, is_best, epoch, train_cfg)

            log_stream.write('Valid MAE: {:.4f} Loss {:.4f}\n'.format(mae, loss))
            log_stream.flush()


def evaluate(net, loader):
    Loss = 0.0
    mae = cal_mae()
    net.eval()

    with torch.no_grad():
        for image, mask, shape, name in loader:
            image = image.cuda().float()
            mask_1 = torch.unsqueeze(mask, dim=0).cuda().float()
            out1u, out2u, out2r, out3r, out4r, out5r = net(image, shape)
            loss1u = structure_loss(out1u, mask_1)
            loss2u = structure_loss(out2u, mask_1)

            loss2r = structure_loss(out2r, mask_1)
            loss3r = structure_loss(out3r, mask_1)
            loss4r = structure_loss(out4r, mask_1)
            loss5r = structure_loss(out5r, mask_1)
            loss = (loss1u + loss2u) / 2 + loss2r / 2 + loss3r / 4 + loss4r / 8 + loss5r / 16

            out = out2u
            pred = (torch.sigmoid(out[0, 0])).cpu().numpy()

            mask = np.asarray(mask, np.float32)
            mask /= (mask.max() + 1e-8)

            mask[mask > 0.5] = 1
            mask[mask != 1] = 0

            pred = np.array(pred)
            if pred.max() == pred.min():
                pred = pred / 255
            else:
                pred = (pred - pred.min()) / (pred.max() - pred.min())
            mae.update(pred, mask)
            Loss += loss
        Mae = mae.show()

    return Mae, Loss / len(loader)


def train(net, optimizer, loader, sw, epoch, cfg):
    net.train()
    for step, (image, target) in enumerate(loader):
        image, target = image.cuda().float(), target.cuda().float()
        # print(image.shape) #(32,3,320,320)
        # print(mask.shape) #(32,1,320,320)
        # import sys
        # sys.exit(0)
        out1u, out2u, out2r, out3r, out4r, out5r = net(image)
        loss1u = structure_loss(out1u, target)
        loss2u = structure_loss(out2u, target)

        loss2r = structure_loss(out2r, target)
        loss3r = structure_loss(out3r, target)
        loss4r = structure_loss(out4r, target)
        loss5r = structure_loss(out5r, target)
        loss = (loss1u + loss2u) / 2 + loss2r / 2 + loss3r / 4 + loss4r / 8 + loss5r / 16

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## log
        global global_step
        global_step += 1
        sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
        sw.add_scalar('loss', loss.item(), global_step=global_step)
        if step % 10 == 0:
            log_stream.write('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f \n' % (
                datetime.datetime.now(), global_step, epoch + 1, cfg.epochs, optimizer.param_groups[0]['lr'],
                loss.item()))
            log_stream.flush()


if __name__ == '__main__':
    main(dataset, F3Net)
