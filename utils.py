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
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import copy
import cv2
import random
import glob



def ToCudaVariable(xs, volatile=False, requires_grad=False):
    if torch.cuda.is_available():
        return [Variable(x.cuda(), volatile=volatile, requires_grad=requires_grad) for x in xs]
    else:
        return [Variable(x, volatile=volatile, requires_grad=requires_grad) for x in xs]

def ToCudaPN(mask):
    P = (mask == 1).astype(np.float32)
    N = (mask == 0).astype(np.float32)
    P = torch.unsqueeze(torch.from_numpy(P), dim=0).float()
    N = torch.unsqueeze(torch.from_numpy(N), dim=0).float()
    return ToCudaVariable([P, N], volatile=True)

def Dilate_mask(mask, num_objects):
    # assume sparse indexed mask (ignore = -1)
    new_mask = -np.ones_like(mask, dtype=np.int64)
    for o in range(num_objects+1): # include bg scribbles
        bmask = (mask == o).astype(np.uint8)
        dmask = cv2.dilate(bmask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)), iterations=2)
        new_mask[np.where(dmask == 1)] = o
    return new_mask

def Get_weight(target, prev_targets, num_frames, at_least=-1):
    right_end = min(filter(lambda x: x > target, prev_targets+[9999]))
    if right_end == 9999:
        NO_R_END = True
        right_end  = num_frames-1
    else:
        NO_R_END = False
    left_end = max(filter(lambda x: x < target, prev_targets+[-9999]))
    if left_end == -9999:
        NO_L_END = True
        left_end  = 0
    else:
        NO_L_END = False

    weight = num_frames*[1.0]

    if (right_end - target) < at_least:
        right_end = min(target + at_least, num_frames-1)
    if (target - left_end) < at_least:
        left_end = max(target - at_least, 0)

    if NO_R_END: # no right end
        pass # set 1.0
    else:
        step = 1.0 / (right_end - target)
        for n,f in enumerate(range(target+1, num_frames)):
            weight[f] = max(0.0, 1.0 - (n+1)*step)

    if NO_L_END: # no left end
        pass # set 1.0
    else:
        step = 1.0 / (target - left_end)
        for n, f in enumerate(reversed(range(0, target))):
            weight[f] = max(0.0, 1.0 - (n+1)*step)

    return left_end, right_end, weight


def To_np_label(all_E, K, index):
    # assume numpy input E: 1,o,t,h,w -> t,h,w 
    sh_E = all_E[0].data.cpu().numpy()
    inv_index = [index.index(i) for i in range(K)]
    E = sh_E[inv_index]
    fgs = np.argmax(E, axis=0)
    return fgs.astype(np.uint8)

def load_frames(path, size=None, num_frames=None):
    fnames = glob.glob(os.path.join(path, '*.jpg')) 
    fnames.sort()
    frame_list = []
    for i, fname in enumerate(fnames):
        if size:
            frame_list.append(np.array(Image.open(fname).convert('RGB').resize((size[0], size[1]), Image.BICUBIC), dtype=np.uint8))
        else:
            frame_list.append(np.array(Image.open(fname).convert('RGB'), dtype=np.uint8))
        if num_frames and i > num_frames:
            break
    frames = np.stack(frame_list, axis=0)
    return frames

def load_UnDP(path):
    # load dataparallel wrapped model properly
    state_dict = torch.load(path, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict



def overlay_davis(image,mask,rgb=[255,0,0],cscale=2,alpha=0.5):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # import skimage
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    im_overlay = image.copy()

    foreground = im_overlay*alpha + np.ones(im_overlay.shape)*(1-alpha) * np.array(rgb, dtype=np.uint8)[None, None, :]
    binary_mask = mask == 1
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    countours = binary_dilation(binary_mask) ^ binary_mask
    im_overlay[countours,:] = 0
    return im_overlay.astype(image.dtype)


def checkerboard(img_size, block_size):
    width = int(np.maximum( np.ceil(img_size[0] / block_size), np.ceil(img_size[1] / block_size)))
    b = np.zeros((block_size, block_size), dtype=np.uint8) + 32
    w = np.zeros((block_size, block_size), dtype=np.uint8) + 255 - 32
    row1 = np.hstack([w,b]*width)
    row2 = np.hstack([b,w]*width)
    board = np.vstack([row1,row2]*width)
    board = np.stack([board, board, board], axis=2)
    return board[:img_size[0], :img_size[1], :] 

BIG_BOARD = checkerboard([1000, 1000], 20)
def overlay_checker(image,mask):
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    # board = checkerboard(image.shape[:2], block_size=20)
    board = BIG_BOARD[:im_overlay.shape[0], :im_overlay.shape[1], :].copy()
    binary_mask = (mask == 1)
    # Compose image
    board[binary_mask] = im_overlay[binary_mask]
    return board.astype(image.dtype)

def overlay_color(image,mask, rgb=[255,0,255]):
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    board = np.ones(image.shape, dtype=np.uint8) * np.array(rgb, dtype=np.uint8)[None, None, :]
    binary_mask = (mask == 1)
    # Compose image
    board[binary_mask] = im_overlay[binary_mask]
    return board.astype(image.dtype)


def overlay_fade(image, mask):
    from scipy.ndimage.morphology import binary_erosion, binary_dilation
    im_overlay = image.copy()

    # Overlay color on  binary mask
    binary_mask = mask == 1
    not_mask = mask != 1

    # Compose image
    im_overlay[not_mask] = 0.4 * im_overlay[not_mask]


    countours = binary_dilation(binary_mask) ^ binary_mask
    im_overlay[countours,0] = 0
    im_overlay[countours,1] = 255
    im_overlay[countours,2] = 255

    return im_overlay.astype(image.dtype)
