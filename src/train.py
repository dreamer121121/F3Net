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
from saliency_metrics import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm, cal_iou

# from smoothing import gaussian_blur

import torch.nn.functional as F

log_stream = open('train.log', 'a')

global_step = 0
best_mae = float('inf')
best_iou = 0.0

gauss_filter = torch.tensor([[1., 4., 6., 4., 1.],
                             [4., 16., 24., 16., 4.],
                             [6., 24., 36., 24., 6.],
                             [4., 16., 24., 16., 4.],
                             [1., 4., 6., 4., 1.]]).cuda()

gauss_filter /= 256.
gauss_filter = gauss_filter.repeat(1, 1, 1, 1)
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


def composite_loss(image, pred, mask):

    return F.l1_loss(image * torch.sigmoid(pred), image * mask) + F.l1_loss(image * (1-torch.sigmoid(pred)), image * (1-mask))


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


def regression_loss(logit, target, loss_type='l1', weight=None):
    """
    Alpha reconstruction loss
    :param logit:
    :param target:
    :param loss_type: "l1" or "l2"
    :param weight: tensor with shape [N,1,H,W] weights for each pixel,
    :return:
    """
    if weight is None:
        if loss_type == 'l1':
            return F.l1_loss(logit, target)
        elif loss_type == 'l2':
            return F.mse_loss(logit, target)
        else:
            raise NotImplementedError("NotImplemented loss type {}".format(loss_type))
    else:
        if loss_type == 'l1':
            return F.l1_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
        elif loss_type == 'l2':
            return F.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
        else:
            raise NotImplementedError("NotImplemented loss type {}".format(loss_type))


def lap_loss(logit, target, gauss_filter, loss_type='l1', weight=None):
    '''
    Based on FBA Matting implementation:
    https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
    '''

    def conv_gauss(x, kernel):
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        x = F.conv2d(x, kernel, groups=x.shape[1])
        return x

    def downsample(x):
        return x[:, :, ::2, ::2]

    def upsample(x, kernel):
        N, C, H, W = x.shape
        cc = torch.cat([x, torch.zeros(N, C, H, W).cuda()], dim=3)
        cc = cc.view(N, C, H * 2, W)
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(N, C, W, H * 2).cuda()], dim=3)
        cc = cc.view(N, C, W * 2, H * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return conv_gauss(x_up, kernel=4 * gauss_filter)

    def lap_pyramid(x, kernel, max_levels=3):
        current = x
        pyr = []
        for level in range(max_levels):
            filtered = conv_gauss(current, kernel)
            down = downsample(filtered)
            up = upsample(down, kernel)
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr

    def weight_pyramid(x, max_levels=3):
        current = x
        pyr = []
        for level in range(max_levels):
            down = downsample(current)
            pyr.append(current)
            current = down
        return pyr

    pyr_logit = lap_pyramid(x=logit, kernel=gauss_filter, max_levels=5)
    pyr_target = lap_pyramid(x=target, kernel=gauss_filter, max_levels=5)
    if weight is not None:
        pyr_weight = weight_pyramid(x=weight, max_levels=5)
        return sum(regression_loss(A[0], A[1], loss_type=loss_type, weight=A[2]) * (2 ** i) for i, A in
                   enumerate(zip(pyr_logit, pyr_target, pyr_weight)))
    else:
        return sum(regression_loss(A[0], A[1], loss_type=loss_type, weight=None) * (2 ** i) for i, A in
                   enumerate(zip(pyr_logit, pyr_target)))


def structure_loss(image, pred, mask, sw=None):
    # add cortor loss
    N, C, W, H = mask.shape
    mc_matrix = np.zeros([N, W, H, C], dtype=np.float)

    #cal smooth loss
    # smoothed_mask = gaussian_blur(mask, (9, 9), (2.5, 2.5))
    # smooth_loss = 5 * F.mse_loss(torch.sigmoid(pred), smoothed_mask)

    for i in range(N):
        f_mask = mask[i, :, :, :]
        f_mask_img = tensor2im(f_mask)
        # e_y = cv2.erode(f_mask_img, kernel=kernal, iterations=1)
        # # d_y = cv2.dilate(f_mask_img, kernel=kernal, iterations=1)
        # m_c = cv2.GaussianBlur(5 * (f_mask_img[:, :, 0] - e_y), (5, 5), 0)
        # m_c = m_c[:, :, np.newaxis]
        transition = generate_trimap(f_mask_img * 255)
        if transition.sum() == 0:
            transition = np.ones((W, H, C))
        mc_matrix[i, :, :, :] =  transition[:, :, :]

    # L = np.ones((N, W, H, C))
    W = im2tensor(mc_matrix).cuda()
    loss_alpha = torch.sqrt(
        torch.square((mask - torch.sigmoid(pred)) * W) + torch.square(torch.Tensor([1e-6]).cuda())).sum(
        dim=(2, 3)) / W.sum(dim=(2, 3))

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    #focal loss
    # focus = FocalLoss()
    # focal_loss = focus(pred, mask)

    #==================composite loss====================
    comp_loss = composite_loss(image, pred, mask)

    if sw:
        sw.add_scalar('scalar/alpha_loss', loss_alpha.mean().item(), global_step=global_step)
        sw.add_scalar('scalar/comp loss', comp_loss.item(), global_step=global_step)
        sw.add_scalar('scalar/wiou loss', wiou.mean().item(), global_step=global_step)
        sw.add_scalar('scalar/wbce loss', wbce.mean().item(), global_step=global_step)

    return (wiou + wbce + loss_alpha).mean() + comp_loss


def structure_loss_2(pred, mask, sw=None):
    # add cortor loss
    N, C, W, H = mask.shape
    mc_matrix = np.zeros([N, W, H, C], dtype=np.float)

    #cal smooth loss
    # smoothed_mask = gaussian_blur(mask, (9, 9), (2.5, 2.5))
    # smooth_loss = 5 * F.mse_loss(torch.sigmoid(pred), smoothed_mask)

    for i in range(N):
        f_mask = mask[i, :, :, :]
        f_mask_img = tensor2im(f_mask)
        # e_y = cv2.erode(f_mask_img, kernel=kernal, iterations=1)
        # # d_y = cv2.dilate(f_mask_img, kernel=kernal, iterations=1)
        # m_c = cv2.GaussianBlur(5 * (f_mask_img[:, :, 0] - e_y), (5, 5), 0)
        # m_c = m_c[:, :, np.newaxis]
        transition = generate_trimap(f_mask_img * 255)
        if transition.sum() == 0:
            transition = np.ones((W, H, C))
        mc_matrix[i, :, :, :] =  transition[:, :, :]

    # L = np.ones((N, W, H, C))
    W = im2tensor(mc_matrix).cuda()
    loss_alpha = torch.sqrt(
        torch.square((mask - torch.sigmoid(pred)) * W) + torch.square(torch.Tensor([1e-6]).cuda())).sum(
        dim=(2, 3)) / W.sum(dim=(2, 3))

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    #focal loss
    # focus = FocalLoss()
    # focal_loss = focus(pred, mask)

    #==================composite loss====================

    return (wiou + wbce + loss_alpha).mean()


def main(Dataset, Network):
    ##parse args
    args = parse_args()

    train_cfg = Dataset.Config(datapath='../data/' + args.dataset, savepath='./out', snapshot=args.resume, mode='train',
                               batch=args.batch_size,
                               lr=args.lr, momen=0.9, decay=args.decay, epochs=args.epochs, start=args.start)

    eval_cfg = Dataset.Config(datapath='../data/' + args.dataset, mode='test', batch=1, eval_freq=5)

    train_data = Dataset.Data(train_cfg)

    eval_data = Dataset.Data(eval_cfg)

    train_dataloader = DataLoader(train_data, collate_fn=train_data.collate, batch_size=train_cfg.batch, shuffle=True,
                                  num_workers=16, pin_memory=True)
    eval_dataloader = DataLoader(eval_data, batch_size=eval_cfg.batch, shuffle=False, num_workers=16, pin_memory=True)

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

    global global_step
    if args.resume:
        global_step = torch.load(args.resume)['step']
    print(global_step)

    if args.eval:
        evaluate(net, eval_dataloader)

    for epoch in range(train_cfg.start, train_cfg.epochs):
        log_stream.write('=' * 30 + 'Epoch: ' + str(epoch + 1) + '=' * 30 + '\n')
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (train_cfg.epochs + 1) * 2 - 1)) * train_cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (train_cfg.epochs + 1) * 2 - 1)) * train_cfg.lr

        train(net, optimizer, train_dataloader, sw, epoch, train_cfg)

        if (epoch + 1) % eval_cfg.eval_freq == 0 or epoch == train_cfg.epochs - 1:
            mae, iou, loss = evaluate(net, eval_dataloader)

            global best_iou
            is_best = iou > best_iou
            best_iou = max(iou, best_iou)

            save_checkpoints({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_iou': best_iou,
                'step': global_step
            }, is_best, epoch, train_cfg)

            log_stream.write('Valid MAE: {:.4f} Valid IOU: {:.4f} Valid Loss: {:.4f}\n'.format(mae, iou, loss))
            log_stream.flush()
            sw.add_scalar('eval loss', loss, global_step=epoch+1)


def evaluate(net, loader):
    Loss = 0.0
    mae = cal_mae()
    iou = cal_iou()
    net.eval()

    with torch.no_grad():
        for image, mask, shape, name in loader:
            image = image.cuda().float()
            mask_1 = mask.unsqueeze(dim=1).cuda().float()
            print('--mask--', mask_1.size())
            out1u, out2u, out2r, out3r, out4r, out5r = net(image, shape)
            loss1u = structure_loss_2(out1u, mask_1)
            loss2u = structure_loss_2(out2u, mask_1)

            loss2r = structure_loss_2(out2r, mask_1)
            loss3r = structure_loss_2(out3r, mask_1)
            loss4r = structure_loss_2(out4r, mask_1)
            loss5r = structure_loss_2(out5r, mask_1)
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
            iou.update(pred, mask)
            print('---eval loss---', loss)
            Loss += loss
        Mae = mae.show()
        Iou = iou.show()
        print('---iou---', Iou)

    return Mae, Iou, Loss/len(loader)


def train(net, optimizer, loader, sw, epoch, cfg):
    net.train()
    for step, (image, mask) in enumerate(loader):
        image, mask = image.cuda().float(), mask.cuda().float()
        # print(image.shape) #(32,3,320,320)
        # print(mask.shape) #(32,1,320,320)
        # import sys
        # sys.exit(0)
        out1u, out2u, out2r, out3r, out4r, out5r = net(image)
        loss1u = structure_loss(image, out1u, mask)
        loss2u = structure_loss(image, out2u, mask)

        loss2r = structure_loss(image, out2r, mask, sw=sw)
        loss3r = structure_loss(image, out3r, mask, )
        loss4r = structure_loss(image, out4r, mask, )
        loss5r = structure_loss(image, out5r, mask, )
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
