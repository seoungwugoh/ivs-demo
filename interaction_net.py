from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import os
import argparse
import copy
import sys

from utils import ToCudaVariable, load_UnDP

print('Interaction Network: initialized')


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.conv1_p = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_n = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/16, 1024
        self.res5 = resnet.layer4 # 1/32, 2048

        # freeze BNs
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m, in_p, in_n):
        f = (in_f - Variable(self.mean)) / Variable(self.std)
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim
        p = torch.unsqueeze(in_p, dim=1).float() # add channel dim
        n = torch.unsqueeze(in_n, dim=1).float() # add channel dim

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_p(p) + self.conv1_n(n)
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 64
        r3 = self.res3(r2) # 1/8, 128
        r4 = self.res4(r3) # 1/16, 256
        r5 = self.res5(r4) # 1/32, 512

        return r5, r4, r3, r2


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)


    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)
        
        return x + r 


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.ResFS = ResBlock(inplanes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(f)
        m = s + F.upsample(pm, scale_factor=self.scale_factor, mode='bilinear')
        m = self.ResMM(m)
        return m

class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.ResFM = ResBlock(2048, mdim)
        self.RF4 = Refine(1024, mdim) # 1/16 -> 1/8
        self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim) # 1/4 -> 1

        self.pred5 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.pred4 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.pred3 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r5, r4, r3, r2):
        m5 = self.ResFM(r5)
        m4 = self.RF4(r4, m5) # out: 1/16, 256
        m3 = self.RF3(r3, m4) # out: 1/8, 256
        m2 = self.RF2(r2, m3) # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        p3 = self.pred3(F.relu(m3))
        p4 = self.pred4(F.relu(m4))
        p5 = self.pred5(F.relu(m5))
        
        p = F.upsample(p2, scale_factor=4, mode='bilinear')
        
        return p, p2, p3, p4, p5



class Inet(nn.Module):
    def __init__(self):
        super(Inet, self).__init__()
        mdim = 256
        self.Encoder = Encoder() # inputs:: ref: rf, rm / tar: tf, tm 
        self.Decoder = Decoder(mdim) # input: m5, r4, r3, r2 >> p
        self.cnt = 0

    def get_ROI_grid(self, roi, src_size, dst_size, scale=1.):
        # scale height and width
        ry, rx, rh, rw = roi[:,0], roi[:,1], scale * roi[:,2], scale * roi[:,3]
        
        # convert ti minmax  
        ymin = ry - rh/2.
        ymax = ry + rh/2.
        xmin = rx - rw/2.
        xmax = rx + rw/2.
        
        h, w = src_size[0], src_size[1] 
        # theta
        theta = ToCudaVariable([torch.zeros(roi.size()[0],2,3)])[0]
        theta[:,0,0] = (xmax - xmin) / (w - 1)
        theta[:,0,2] = (xmin + xmax - (w - 1)) / (w - 1)
        theta[:,1,1] = (ymax - ymin) / (h - 1)
        theta[:,1,2] = (ymin + ymax - (h - 1)) / (h - 1)

        #inverse of theta
        inv_theta = ToCudaVariable([torch.zeros(roi.size()[0],2,3)])[0]
        det = theta[:,0,0]*theta[:,1,1]
        adj_x = -theta[:,0,2]*theta[:,1,1]
        adj_y = -theta[:,0,0]*theta[:,1,2]
        inv_theta[:,0,0] = w / (xmax - xmin) 
        inv_theta[:,1,1] = h / (ymax - ymin) 
        inv_theta[:,0,2] = adj_x / det
        inv_theta[:,1,2] = adj_y / det
        # make affine grid
        fw_grid = F.affine_grid(theta, torch.Size((roi.size()[0], 1, dst_size[0], dst_size[1])))
        bw_grid = F.affine_grid(inv_theta, torch.Size((roi.size()[0], 1, src_size[0], src_size[1])))
        return fw_grid, bw_grid, theta


    def all2yxhw(self, mask, pos, neg, scale=1.0):
        np_mask = mask.data.cpu().numpy()
        np_pos = pos.data.cpu().numpy()
        np_neg = neg.data.cpu().numpy()

        np_yxhw = np.zeros((np_mask.shape[0], 4), dtype=np.float32)
        for b in range(np_mask.shape[0]):
            mys, mxs = np.where(np_mask[b] >= 0.49)
            pys, pxs = np.where(np_pos[b] >= 0.49)
            nys, nxs = np.where(np_neg[b] >= 0.49)
            all_ys = np.concatenate([mys,pys,nys])
            all_xs = np.concatenate([mxs,pxs,nxs])

            if all_ys.size == 0 or all_xs.size == 0:
                # if no pixel, return whole
                ymin, ymax = 0, np_mask.shape[1]
                xmin, xmax = 0, np_mask.shape[2]
            else:
                ymin, ymax = np.min(all_ys), np.max(all_ys)
                xmin, xmax = np.min(all_xs), np.max(all_xs)

            # make sure minimum 128 original size
            if (ymax-ymin) < 128:
                res = 128. - (ymax-ymin)
                ymin -= int(res/2)
                ymax += int(res/2)

            if (xmax-xmin) < 128:
                res = 128. - (xmax-xmin)
                xmin -= int(res/2)
                xmax += int(res/2)

            # apply scale
            # y = (ymax + ymin) / 2.
            # x = (xmax + xmin) / 2.
            orig_h = ymax - ymin + 1
            orig_w = xmax - xmin + 1

            ymin = np.maximum(-5, ymin - (scale - 1) / 2. * orig_h)  
            ymax = np.minimum(np_mask.shape[1]+5, ymax + (scale - 1) / 2. * orig_h)    
            xmin = np.maximum(-5, xmin - (scale - 1) / 2. * orig_w)  
            xmax = np.minimum(np_mask.shape[2]+5, xmax + (scale - 1) / 2. * orig_w)  

            # final ywhw
            y = (ymax + ymin) / 2.
            x = (xmax + xmin) / 2.
            h = ymax - ymin + 1
            w = xmax - xmin + 1

            yxhw = np.array([y,x,h,w], dtype=np.float32)
            
            np_yxhw[b] = yxhw
            
        return ToCudaVariable([torch.from_numpy(np_yxhw.copy()).float()])[0]


    def is_there_scribble(self, p, n ):
        num_pixel_p = np.sum(p.data.cpu().numpy(), axis=(1,2))
        num_pixel_n = np.sum(n.data.cpu().numpy(), axis=(1,2))
        num_pixel = num_pixel_p + num_pixel_n
        yes = (num_pixel > 0).astype(np.float32)
        mulplier = 1 / (np.mean(yes) + 0.001)
        yes = yes * mulplier
        return ToCudaVariable([torch.from_numpy(yes.copy()).float()])[0]

    def forward(self, tf, tm, tp, tn, gm, loss_weight):  # b,c,h,w // b,4 (y,x,h,w)
        if tm is None:
            tm = ToCudaVariable([0.5*torch.ones(gm.size())], requires_grad=False)[0]
        tb = self.all2yxhw(tm, tp, tn, scale=1.5)
        
        oh, ow = tf.size()[2], tf.size()[3] # original size
        fw_grid, bw_grid, theta = self.get_ROI_grid(tb, src_size=(oh, ow), dst_size=(256,256), scale=1.0)

        #  Sample target frame
        tf_roi = F.grid_sample(tf, fw_grid)
        tm_roi = F.grid_sample(torch.unsqueeze(tm, dim=1).float(), fw_grid)[:,0]
        tp_roi = F.grid_sample(torch.unsqueeze(tp, dim=1).float(), fw_grid)[:,0]
        tn_roi = F.grid_sample(torch.unsqueeze(tn, dim=1).float(), fw_grid)[:,0]

        # run Siamese Encoder
        tr5, tr4, tr3, tr2 = self.Encoder(tf_roi, tm_roi, tp_roi, tn_roi)
        em_roi = self.Decoder(tr5, tr4, tr3, tr2)

        ## Losses are computed within ROI
        # CE loss
        gm_roi = F.grid_sample(torch.unsqueeze(gm, dim=1).float(), fw_grid)[:,0]
        gm_roi = gm_roi.detach()
        # CE loss
        CE = nn.CrossEntropyLoss(reduce=False)
        batch_CE = ToCudaVariable([torch.zeros(gm_roi.size()[0])])[0] # batch sized loss container 
        sizes=[(256,256), (64,64), (32,32), (16,16), (8,8)]
        for s in range(5):
            if s == 0:
                CE_s = CE(em_roi[s], torch.round(gm_roi).long()).mean(-1).mean(-1) # mean over h,w
                batch_CE += loss_weight[s] * CE_s
            else:
                if loss_weight[s]:
                    gm_roi_s = torch.round(F.upsample(torch.unsqueeze(gm_roi, dim=1), size=sizes[s], mode='bilinear')[:,0]).long()
                    CE_s = CE(em_roi[s], gm_roi_s).mean(-1).mean(-1) # mean over h,w
                    batch_CE += loss_weight[s] * CE_s

        batch_CE = batch_CE * self.is_there_scribble(tp, tn)


        # get final output via inverse warping
        em = F.grid_sample(F.softmax(em_roi[0], dim=1), bw_grid)[:,1]
        # return em, batch_CE, [tr5, tr4, tr3, tr2]
        return em, batch_CE, tr5