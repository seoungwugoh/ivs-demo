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
import random

# my libs
from utils import ToCudaVariable, ToCudaPN, Dilate_mask, load_UnDP, Get_weight, overlay_davis, overlay_checker, overlay_color, overlay_fade
from interaction_net import Inet
from propagation_net import Pnet

# davis
from davisinteractive.utils.scribbles import scribbles2mask, annotated_frames

# palette
palette = Image.open('00000.png').getpalette()
palette[3:6] = [0,0,128]

class model():
    def __init__(self, frames):
        self.model_I = Inet()
        self.model_P = Pnet()
        if torch.cuda.is_available():
            print('Using GPU')
            self.model_I = nn.DataParallel(self.model_I)
            self.model_P = nn.DataParallel(self.model_P)
            self.model_I.cuda()
            self.model_P.cuda()
            self.model_I.load_state_dict(torch.load('I_e290.pth'))
            self.model_P.load_state_dict(torch.load('P_e290.pth'))
        else:
            print('Using CPU')
            self.model_I.load_state_dict(load_UnDP('I_e290.pth'))
            self.model_P.load_state_dict(load_UnDP('P_e290.pth'))

        self.model_I.eval() # turn-off BN
        self.model_P.eval() # turn-off BN

        self.frames = frames.copy()
        self.num_frames, self.height, self.width = self.frames.shape[:3]

        self.init_variables(self.frames)
        
    def init_variables(self, frames):
        self.all_F = torch.unsqueeze(torch.from_numpy(np.transpose(frames, (3, 0, 1, 2))).float() / 255., dim=0) # 1,3,t,h,w
        self.all_E = torch.zeros(1, self.num_frames, self.height, self.width)  # 1,t,h,w
        self.prev_E = torch.zeros(1, self.num_frames, self.height, self.width)  # 1,t,h,w
        self.dummy_M = torch.zeros(1, self.height, self.width).long()
        # to cuda
        self.all_F, self.all_E, self.prev_E, self.dummy_M = ToCudaVariable([self.all_F, self.all_E, self.prev_E, self.dummy_M], volatile=True)
        
        self.ref = None
        self.a_ref = None
        self.next_a_ref = None
        self.prev_targets = []


    def Prop_forward(self, target, end):
        for n in range(target+1, end+1):  #[1,2,...,N-1]
            print('[MODEL: propagation network] >>>>>>>>> {} to {}'.format(n-1, n))
            self.all_E[:,n], _, self.next_a_ref = self.model_P(self.ref, self.a_ref, self.all_F[:,:,n], self.prev_E[:,n], torch.round(self.all_E[:,n-1]), self.dummy_M, [1,0,0,0,0])

    def Prop_backward(self, target, end):
        for n in reversed(range(end, target)): #[N-2,N-3,...,0]
            print('[MODEL: propagation network] {} to {} <<<<<<<<<'.format(n+1, n))
            self.all_E[:,n], _, self.next_a_ref = self.model_P(self.ref, self.a_ref, self.all_F[:,:,n], self.prev_E[:,n], torch.round(self.all_E[:,n+1]), self.dummy_M, [1,0,0,0,0])


    def Run_propagation(self, target, mode='linear', at_least=-1, std=None):
        # when new round begins
        self.a_ref = self.next_a_ref
        self.prev_E = self.all_E  

        if mode == 'naive':
            left_end, right_end, weight = 0, self.num_frames-1, num_frames*[1.0]
        elif mode == 'linear':
            left_end, right_end, weight = Get_weight(target, self.prev_targets, self.num_frames, at_least=at_least)
        else:
            raise NotImplementedError

        self.Prop_forward(target, right_end)
        self.Prop_backward(target, left_end)

        for f in range(self.num_frames):
            self.all_E[:,:,f] = weight[f] * self.all_E[:,:,f] + (1-weight[f]) * self.prev_E[:,:,f]

        self.prev_targets.append(target)
        print('[MODEL] Propagation finished.')    

    def Run_interaction(self, scribbles):
        
        # convert davis scribbles to torch
        target = scribbles['annotated_frame']
        scribble_mask = scribbles2mask(scribbles, (self.height, self.width))[target]
        scribble_mask = Dilate_mask(scribble_mask, 1)
        self.tar_P, self.tar_N = ToCudaPN(scribble_mask)

        self.all_E[:,target], _, self.ref = self.model_I(self.all_F[:,:,target], self.all_E[:,target], self.tar_P, self.tar_N, self.dummy_M, [1,0,0,0,0]) # [batch, 256,512,2]

        print('[MODEL: interaction network] User Interaction on {}'.format(target))    

    def Get_mask(self):
        return torch.round(self.all_E[0]).data.cpu().numpy().astype(np.uint8) 

    def Get_mask_range(self, start, end):
        pred_masks = torch.round(self.all_E[0, start:end]).data.cpu().numpy().astype(np.uint8) # t,h,w
        return torch.round(self.all_E[0, start:end]).data.cpu().numpy().astype(np.uint8)

    def Get_mask_index(self, index):
        return torch.round(self.all_E[0, index]).data.cpu().numpy().astype(np.uint8)


