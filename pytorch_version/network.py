import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable, grad
import numpy as np
import math


def relu():
    return nn.ReLU(inplace=True)


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding = 1, nonlinearity = relu):
    conv_layer = nn.Conv2d(in_channels = in_channels, out_channels= out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
    nn.init.xavier_uniform(conv_layer.weight, gain=np.sqrt(2.0))

    nll_layer = nonlinearity()
    bn_layer = nn.BatchNorm2d(out_channels)
    # nn.init.constant_(bn_layer.weight, 1)
    # nn.init.constant_(bn_layer.bias, 0)

    layers = [conv_layer, bn_layer, nll_layer]
    return nn.Sequential(*layers)


def conv_blocks_2(in_channels, out_channels, strides=1):
    conv1 = conv(in_channels, out_channels, stride = strides)
    conv2 = conv(out_channels, out_channels, stride=1)
    layers = [conv1, conv2]
    return nn.Sequential(*layers)


def conv_blocks_3(in_channels, out_channels, strides=1):
    conv1 = conv(in_channels, out_channels, stride = strides)
    conv2 = conv(out_channels, out_channels, stride=1)
    conv3 = conv(out_channels, out_channels, stride=1)
    layers = [conv1, conv2, conv3]
    return nn.Sequential(*layers)


def generate_grid(x, offset):
    x_shape = x.size()
    grid_w, grid_h = torch.meshgrid([torch.linspace(-1, 1, x_shape[2]), torch.linspace(-1, 1, x_shape[3])])  # (h, w)
    grid_w = grid_w.cuda().float()
    grid_h = grid_h.cuda().float()

    grid_w = nn.Parameter(grid_w, requires_grad=False)
    grid_h = nn.Parameter(grid_h, requires_grad=False)

    offset_h, offset_w = torch.split(offset, 1, 1)
    offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
    offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)

    offset_w = grid_w + offset_w
    offset_h = grid_h + offset_h

    offsets = torch.stack((offset_h, offset_w), 3)
    return offsets


class Registration_Net(nn.Module):
    """Deformable registration network with input from image space """
    def __init__(self, n_ch=1):
        super(Registration_Net, self).__init__()

        self.conv_blocks = [conv_blocks_2(n_ch, 64), conv_blocks_2(64, 128, 2), conv_blocks_3(128, 256, 2), conv_blocks_3(256, 512, 2), conv_blocks_3(512, 512, 2)]
        self.conv = []
        for in_filters in [128, 256, 512, 1024, 1024]:
            self.conv += [conv(in_filters, 64)]

        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        self.conv = nn.Sequential(*self.conv)

        self.conv6 = nn.Conv2d(64 * 5, 64, 1)
        self.conv7 = conv(64, 64, 1, 1, 0)
        self.conv8 = nn.Conv2d(64, 2, 1)

    def forward(self, x, x_pred, x_img):
        # x: source image; x_pred: target image; x_img: source image or segmentation map
        net = {}
        net['conv0'] = x
        net['conv0s'] = x_pred
        for i in range(5):
            net['conv%d'% (i+1)] = self.conv_blocks[i](net['conv%d'%i])
            net['conv%ds' % (i + 1)] = self.conv_blocks[i](net['conv%ds' % i])
            net['concat%d'%(i+1)] = torch.cat((net['conv%d'% (i+1)], net['conv%ds' % (i + 1)]), 1)
            net['out%d'%(i+1)] = self.conv[i](net['concat%d'%(i+1)])
            if i > 0:
                net['out%d_up'%(i+1)] = F.interpolate(net['out%d'%(i+1)], scale_factor=2**i, mode='bilinear', align_corners=True)

        net['concat'] = torch.cat((net['out1'], net['out2_up'], net['out3_up'], net['out4_up'], net['out5_up']), 1)
        net['comb_1'] = self.conv6(net['concat'])
        net['comb_2'] = self.conv7(net['comb_1'])

        net['out'] = torch.tanh(self.conv8(net['comb_2']))
        net['grid'] = generate_grid(x_img, net['out'])
        net['fr_st'] = F.grid_sample(x_img, net['grid'])

        return net


class Seg_Motion_Net(nn.Module):
    """Joint motion estimation and segmentation """
    def __init__(self, n_ch=1):
        super(Seg_Motion_Net, self).__init__()
        self.conv_blocks = [conv_blocks_2(n_ch, 64), conv_blocks_2(64, 128, 2), conv_blocks_3(128, 256, 2), conv_blocks_3(256, 512, 2), conv_blocks_3(512, 512, 2)]
        self.conv = []
        for in_filters in [128, 256, 512, 1024, 1024]:
            self.conv += [conv(in_filters, 64)]

        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        self.conv = nn.Sequential(*self.conv)

        self.conv6 = nn.Conv2d(64 * 5, 64, 1)
        self.conv7 = conv(64, 64, 1, 1, 0)
        self.conv8 = nn.Conv2d(64, 2, 1)

        self.convs = []
        for in_filters in [64, 128, 256, 512, 512]:
            self.convs += [conv(in_filters, 64)]
        self.convs = nn.Sequential(*self.convs)
        self.conv6s = nn.Conv2d(64*5,64,1)

        self.conv7s = conv(64,64,1,1,0)
        self.conv8s = nn.Conv2d(64,4,1)

    def forward(self, x, x_pred, x_img):
        # x: source image; x_pred: target image; x_img: image to be segmented
        # motion estimation branch
        net = {}
        net['conv0'] = x
        net['conv0s'] = x_pred
        for i in range(5):
            net['conv%d'% (i+1)] = self.conv_blocks[i](net['conv%d'%i])
            net['conv%ds' % (i + 1)] = self.conv_blocks[i](net['conv%ds' % i])
            net['concat%d'%(i+1)] = torch.cat((net['conv%d'% (i+1)], net['conv%ds' % (i + 1)]), 1)
            net['out%d'%(i+1)] = self.conv[i](net['concat%d'%(i+1)])
            if i > 0:
                net['out%d_up'%(i+1)] = F.interpolate(net['out%d'%(i+1)], scale_factor=2**i, mode='bilinear', align_corners=True)

        net['concat'] = torch.cat((net['out1'], net['out2_up'], net['out3_up'], net['out4_up'], net['out5_up']), 1)
        net['comb_1'] = self.conv6(net['concat'])
        net['comb_2'] = self.conv7(net['comb_1'])

        net['out'] = torch.tanh(self.conv8(net['comb_2']))
        net['grid'] = generate_grid(x_img, net['out'])
        net['fr_st'] = F.grid_sample(x_img, net['grid'])

        # segmentation branch
        net['conv0ss'] = x_img
        for i in range(5):
            net['conv%dss' % (i + 1)] = self.conv_blocks[i](net['conv%dss' % i])
            net['out%ds' % (i+1)] = self.convs[i](net['conv%dss' % (i+1)])
            if i > 0:
                net['out%ds_up'%(i+1)] = F.interpolate(net['out%ds'%(i+1)], scale_factor=2**i, mode='bilinear', align_corners=True)

        net['concats'] = torch.cat((net['out1s'],
                                       net['out2s_up'],
                                       net['out3s_up'],
                                       net['out4s_up'],
                                       net['out5s_up']), 1)
        net['comb_1s'] = self.conv6s(net['concats'])
        net['comb_2s'] = self.conv7s(net['comb_1s'])
        net['outs'] = self.conv8s(net['comb_2s'])
        net['outs_softmax'] = F.softmax(net['outs'], dim=1)
        # net['warped_outs'] = F.grid_sample(net['outs_softmax'], net['grid'], padding_mode='border')

        return net