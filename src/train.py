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

import cv2
from saliency_metrics import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm

log_stream = open('train.log', 'a')

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


def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    image_numpy = image_tensor.cpu().float().detach().numpy()
    # print(image_numpy.shape)
    # blank_image = np.zeros((image_tensor.shape[1],image_tensor.shape[2],image_tensor.shape[0]), np.uint8)

    blank_image = image_numpy.reshape(image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[0])

    return blank_image


def im2tensor(image_numpy):
    image_tensor = torch.Tensor(image_numpy)
    (N, W, H, C) = image_tensor.size()
    res = image_tensor.view((N, C, W, H))

    return res


def generate_trimap(mask):
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

    # cv2.imshow('111', foreground)
    # cv2.imshow('222', unknown)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # import sys
    # sys.exit(0)

    return unknown / 255


def structure_loss(pred, mask):
    # wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    # wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))
    #

    # add cortor loss
    N, C, W, H = mask.shape

    mc_matrix = np.zeros([N, W, H, C], dtype=np.float)
    for i in range(N):
        f_mask = mask[i, :, :, :]
        f_mask_img = tensor2im(f_mask)
        transition = generate_trimap(f_mask_img*255)
        if transition.sum() == 0:
            transition = np.ones((W, H, C))
        mc_matrix[i, :, :, :] = transition[:, :, :]

    W = im2tensor(mc_matrix).cuda()
    loss_alpha = torch.sqrt(
        torch.square((mask - torch.sigmoid(pred)) * W) + torch.square(torch.Tensor([1e-6]).cuda())).sum(
        dim=(2, 3)) / W.sum(dim=(2, 3))

    print('--loss_alpha--', loss_alpha.mean())
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wiou + wbce + loss_alpha).mean()


def main(Dataset, Network):
    ##parse args
    args = parse_args()

    train_cfg = Dataset.Config(datapath='../data/' + args.dataset, savepath='./out', snapshot=args.resume, mode='train',
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
    for step, (image, mask) in enumerate(loader):
        image, mask = image.cuda().float(), mask.cuda().float()
        # print(image.shape) #(32,3,320,320)
        # print(mask.shape) #(32,1,320,320)
        # import sys
        # sys.exit(0)
        out1u, out2u, out2r, out3r, out4r, out5r = net(image)
        loss1u = structure_loss(out1u, mask)
        loss2u = structure_loss(out2u, mask)

        loss2r = structure_loss(out2r, mask)
        loss3r = structure_loss(out3r, mask)
        loss4r = structure_loss(out4r, mask)
        loss5r = structure_loss(out5r, mask)
        loss = (loss1u + loss2u) / 2 + loss2r / 2 + loss3r / 4 + loss4r / 8 + loss5r / 16

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## log
        global global_step
        global_step += 1
        sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
        sw.add_scalars('loss', {'loss1u': loss1u.item(), 'loss2u': loss2u.item(), 'loss2r': loss2r.item(),
                                'loss3r': loss3r.item(), 'loss4r': loss4r.item(), 'loss5r': loss5r.item()},
                       global_step=global_step)
        if step % 10 == 0:
            log_stream.write('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f \n' % (
                datetime.datetime.now(), global_step, epoch + 1, cfg.epochs, optimizer.param_groups[0]['lr'],
                loss.item()))
            log_stream.flush()


if __name__ == '__main__':
    main(dataset, F3Net)
