#!/usr/bin/python3
#coding=utf-8

import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from net  import F3Net
#from apex import amp
#from apex.parallel import convert_syncbn_model
# from apex.parallel import DistributedDataParallel as DDP
# from torch.nn.parallel import DistributedDataParallel as DDP


log_stream = open('train.log','a')
# torch.cuda.set_device(4)
# torch.distributed.init_process_group(backend='nccl') #must init the process_group if use the distributed.parallel

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='../data/train_data_no_opv6/', savepath='./out', mode='train', batch=32,
                            lr=0.05, momen=0.9, decay=5e-4, epoch=100,snapshot='./out/model-8')
    data   = Dataset.Data(cfg)
    #train_sampler = torch.utils.data.distributed.DistributedSampler(data)
    loader = DataLoader(data,collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=16)
    ## network
    net    = Network(cfg)
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
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    #net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    #resume = True
    #if resume:
    #    checkpoints = torch.load("./out/model-100")
    #    net.load_state_dict(checkpoints)
        #optimizer.load_state_dict(torch.load(''))

    #torch.distributed.init_process_group('nccl',init_method='file:///home/.../my_file',world_size=1,rank=0)
    net = nn.DataParallel(net,device_ids=[0,1,2,3])
    net.cuda()

    #
    # resume = False
    # if resume:
    #     checkpoints = torch.load("./out/model-100")
    #     net.load_state_dict(checkpoints)
    #     #optimizer.load_state_dict(torch.load(''))
    #

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr
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
            loss   = (loss1u+loss2u)/2+loss2r/2+loss3r/4+loss4r/8+loss5r/16

            optimizer.zero_grad()
#            with amp.scale_loss(loss, optimizer) as scale_loss:
#                scale_loss.backward()
            loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss1u':loss1u.item(), 'loss2u':loss2u.item(), 'loss2r':loss2r.item(), 'loss3r':loss3r.item(), 'loss4r':loss4r.item(), 'loss5r':loss5r.item()}, global_step=global_step)
            if step%10 == 0:
                log_stream.write('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f \n'%(datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item()))
                log_stream.flush()
        
        if epoch>5:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))
            torch.save(optimizer.state_dict(),cfg.savepath+'/opt-'+str(epoch+1))


if __name__=='__main__':
    train(dataset, F3Net)
