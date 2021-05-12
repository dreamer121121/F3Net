#!/usr/bin/python3
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# from train import log_stream


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.ModuleList):
            pass
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('../res/resnet50-19c8e357.pth'), strict=False)


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

    def initialize(self):
        weight_init(self)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)

    def initialize(self):
        weight_init(self)


class DSConv5x5(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv5x5, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=5, s=stride, p=2*dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)


class VAMM_backbone(nn.Module):
    def __init__(self, pretrained=None):
        super(VAMM_backbone, self).__init__()
        self.layer1 = nn.Sequential(
                convbnrelu(3, 16, k=3, s=2, p=1),
                VAMM(16, dilation_level=[1,2,3])
                )
        self.layer2 = nn.Sequential(
                DSConv3x3(16, 32, stride=2),
                VAMM(32, dilation_level=[1,2,3])
                )
        self.layer3 = nn.Sequential(
                DSConv3x3(32, 64, stride=2),
                VAMM(64, dilation_level=[1,2,3]),
                VAMM(64, dilation_level=[1,2,3]),
                VAMM(64, dilation_level=[1,2,3])
                )
        self.layer4 = nn.Sequential(
                DSConv3x3(64, 96, stride=2),
                VAMM(96, dilation_level=[1,2,3]),
                VAMM(96, dilation_level=[1,2,3]),
                VAMM(96, dilation_level=[1,2,3]),
                VAMM(96, dilation_level=[1,2,3]),
                VAMM(96, dilation_level=[1,2,3]),
                VAMM(96, dilation_level=[1,2,3])
                )
        self.layer5 = nn.Sequential(
                DSConv3x3(96, 128, stride=2),
                VAMM(128, dilation_level=[1,2]),
                VAMM(128, dilation_level=[1,2]),
                VAMM(128, dilation_level=[1,2])
                )
        #
        # if pretrained is not None:
        #     self.load_state_dict(torch.load(pretrained))
        #     print('Pretrained model loaded!')

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('../res/SAMNet_backbone_pretrain.pth'), strict=False)


class VAMM(nn.Module):
    def __init__(self, channel, dilation_level=[1,2,4,8], reduce_factor=4):
        super(VAMM, self).__init__()
        self.planes = channel
        self.dilation_level = dilation_level
        self.conv = DSConv3x3(channel, channel, stride=1)
        self.branches = nn.ModuleList([
                DSConv3x3(channel, channel, stride=1, dilation=d) for d in dilation_level
                ])
        ### ChannelGate
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = convbnrelu(channel, channel, 1, 1, 0, bn=True, relu=True)
        self.fc2 = nn.Conv2d(channel, (len(self.dilation_level) + 1) * channel, 1, 1, 0, bias=False)
        self.fuse = convbnrelu(channel, channel, k=1, s=1, p=0, relu=False)
        ### SpatialGate
        self.convs = nn.Sequential(
                convbnrelu(channel, channel // reduce_factor, 1, 1, 0, bn=True, relu=True),
                DSConv3x3(channel // reduce_factor, channel // reduce_factor, stride=1, dilation=2),
                DSConv3x3(channel // reduce_factor, channel // reduce_factor, stride=1, dilation=4),
                nn.Conv2d(channel // reduce_factor, 1, 1, 1, 0, bias=False)
                )

    def forward(self, x):
        conv = self.conv(x)
        brs = [branch(conv) for branch in self.branches]
        brs.append(conv)
        gather = sum(brs)

        ### ChannelGate
        d = self.gap(gather)
        d = self.fc2(self.fc1(d))
        d = torch.unsqueeze(d, dim=1).view(-1, len(self.dilation_level) + 1, self.planes, 1, 1)

        ### SpatialGate
        s = self.convs(gather).unsqueeze(1)

        ### Fuse two gates
        f = d * s
        f = F.softmax(f, dim=1)

        return self.fuse(sum([brs[i] * f[:, i, ...] for i in range(len(self.dilation_level) + 1)]))	+ x

    def initialize(self):
        weight_init(self)


class CFM(nn.Module):
    def __init__(self):
        super(CFM, self).__init__()
        self.conv1h = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn1h   = nn.BatchNorm2d(16)
        self.conv2h = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(16)
        self.conv3h = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn3h   = nn.BatchNorm2d(16)
        self.conv4h = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn4h   = nn.BatchNorm2d(16)

        self.conv1v = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn1v   = nn.BatchNorm2d(16)
        self.conv2v = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn2v   = nn.BatchNorm2d(16)
        self.conv3v = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn3v   = nn.BatchNorm2d(16)
        self.conv4v = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn4v   = nn.BatchNorm2d(16)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = F.relu(self.bn1h(self.conv1h(left )), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down )), inplace=True)
        out2v = F.relu(self.bn2v(self.conv2v(out1v)), inplace=True)
        fuse  = out2h*out2v
        out3h = F.relu(self.bn3h(self.conv3h(fuse )), inplace=True)+out1h
        out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse )), inplace=True)+out1v
        out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True)
        return out4h, out4v

    def initialize(self):
        weight_init(self)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cfm45  = CFM()
        self.cfm34  = CFM()
        self.cfm23  = CFM()

    def forward(self, out2h, out3h, out4h, out5v, fback=None):
        if fback is not None:
            refine5      = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4      = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3      = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2      = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')
            out5v        = out5v+refine5
            out4h, out4v = self.cfm45(out4h+refine4, out5v)
            out3h, out3v = self.cfm34(out3h+refine3, out4v)
            out2h, pred  = self.cfm23(out2h+refine2, out3v)
        else:
            out4h, out4v = self.cfm45(out4h, out5v)
            out3h, out3v = self.cfm34(out3h, out4v)
            out2h, pred  = self.cfm23(out2h, out3v)
        return out2h, out3h, out4h, out5v, pred

    def initialize(self):
        weight_init(self)


class F3Net(nn.Module):
    def __init__(self, cfg):
        super(F3Net, self).__init__()
        self.cfg      = cfg
        # self.bkbone   = ResNet()
        self.bkbone = VAMM_backbone('../res/SAMNet_backbone_pretrain.pth')
        # '../res/SAMNet_backbone_pretrain.pth'
        # self.squeeze5 = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # self.squeeze4 = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # self.squeeze3 = nn.Sequential(nn.Conv2d( 512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # self.squeeze2 = nn.Sequential(nn.Conv2d( 256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.squeeze5 = nn.Sequential(nn.Conv2d(128, 16, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(96, 16, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d(64, 16, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(32, 16, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))

        self.decoder1 = Decoder()
        self.decoder2 = Decoder()
        self.linearp1 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.linearp2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

        self.linearr2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

        self.initialize()

    def forward(self, x, shape=None):
        out2h, out3h, out4h, out5v        = self.bkbone(x)
        out2h, out3h, out4h, out5v        = self.squeeze2(out2h), self.squeeze3(out3h), self.squeeze4(out4h), self.squeeze5(out5v)
        out2h, out3h, out4h, out5v, pred1 = self.decoder1(out2h, out3h, out4h, out5v)
        out2h, out3h, out4h, out5v, pred2 = self.decoder2(out2h, out3h, out4h, out5v, pred1)

        shape = x.size()[2:] if shape is None else shape
        pred1 = F.interpolate(self.linearp1(pred1), size=shape, mode='bilinear')
        pred2 = F.interpolate(self.linearp2(pred2), size=shape, mode='bilinear')

        out2h = F.interpolate(self.linearr2(out2h), size=shape, mode='bilinear')
        out3h = F.interpolate(self.linearr3(out3h), size=shape, mode='bilinear')
        out4h = F.interpolate(self.linearr4(out4h), size=shape, mode='bilinear')
        out5h = F.interpolate(self.linearr5(out5v), size=shape, mode='bilinear')
        return pred1, pred2, out2h, out3h, out4h, out5h

    def initialize(self):
        if self.cfg.snapshot: #finetune
            if os.path.isfile(self.cfg.snapshot):
                print('=> loading checkpoint {} \n'.format(self.cfg.snapshot))
            checkpoints = torch.load(self.cfg.snapshot)['state_dict']
            #checkpoints = torch.load(self.cfg.snapshot)
            fielter_checkpoints = dict()
            for k,v in checkpoints.items():
                if "module" in k:
                    fielter_checkpoints[k[7:]] = v
                else:
                    fielter_checkpoints[k] = v
            self.load_state_dict(fielter_checkpoints)
        else:
            weight_init(self)


if __name__ == '__main__':
    cfg = None
    model = F3Net(cfg)
    Input = torch.randn((1, 3, 352, 352))
    # out = model(input)
    from thop import profile, clever_format

    flops, params = profile(model, inputs=(Input,))
    flops, params = clever_format([flops, params], '%.3f')
    print("flops {}, params {}".format(flops, params))

