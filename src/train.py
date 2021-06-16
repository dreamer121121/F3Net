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
from net import ResNet
import shutil
import argparse

from saliency_metrics import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm

log_stream = open('train_bkbone.log', 'a')

global_step = 0
best_accuracy = 0


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


def criterion(pred, target):
    metric = nn.CrossEntropyLoss()

    return metric(pred, target)


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


def main(Dataset, Network):
    ##parse args
    args = parse_args()

    train_cfg = Dataset.Config(datapath='../../class_data/', savepath='./out_bkbone', snapshot=args.resume,
                               mode='train',
                               batch=args.batch_size,
                               lr=args.lr, momen=0.9, decay=args.decay, epochs=args.epochs, start=args.start)

    eval_cfg = Dataset.Config(datapath='../../class_data_eval/', mode='test', eval_freq=1)

    train_data = Dataset.Data(train_cfg)

    eval_data = Dataset.Data(eval_cfg)

    train_dataloader = DataLoader(train_data, collate_fn=train_data.collate, batch_size=train_cfg.batch, shuffle=True,
                                  num_workers=0)
    eval_dataloader = DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=0)

    net = Network(num_classes=5)
    net.train(True)

    ## parameter
    optimizer = torch.optim.SGD(params=net.parameters(), lr=train_cfg.lr, momentum=train_cfg.momen,
                                weight_decay=train_cfg.decay, nesterov=True)
    sw = SummaryWriter(train_cfg.savepath)

    # net = nn.DataParallel(net, device_ids=[0])
    net = net.cuda()

    if args.eval:
        evaluate(net, eval_dataloader)

    for epoch in range(train_cfg.start, train_cfg.epochs):
        log_stream.write('=' * 30 + 'Epoch: ' + str(epoch + 1) + '=' * 30 + '\n')
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (train_cfg.epochs + 1) * 2 - 1)) * train_cfg.lr

        train(net, optimizer, train_dataloader, sw, epoch, train_cfg)

        if (epoch + 1) % eval_cfg.eval_freq == 0 or epoch == train_cfg.epochs - 1:
            acc = evaluate(net, eval_dataloader)

            global best_accuracy
            is_best = acc > best_accuracy
            best_accuracy = max(acc, best_accuracy)

            save_checkpoints({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_acc': best_accuracy,
            }, is_best, epoch, train_cfg)

            log_stream.write('acc {:.4f}\n'.format(acc))
            log_stream.flush()


def train(net, optimizer, loader, sw, epoch, cfg):
    net.train()
    for step, (image, label) in enumerate(loader):
        image, label = image.cuda().float(), label.cuda()
        print(label.size())

        out = net(image)
        loss = criterion(out, label)

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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(net, loader):
    top1 = AverageMeter()
    net.eval()
    with torch.no_grad():
        for step, (image, label) in enumerate(loader):
            print('--image.shape--', image.size())
            print('--label.shape--', label.size())
            image, label = image.cuda().float(), label.cuda()

            out = net(image)
            prec1, prec5 = accuracy(out, label, topk=(1, 5))
            top1.update(prec1[0], image.size(0))

    return top1.avg


if __name__ == '__main__':
    main(dataset, ResNet)
