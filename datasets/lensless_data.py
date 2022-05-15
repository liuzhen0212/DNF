import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import random
import cv2
import math
import matplotlib.pyplot as plt

from scipy.io import loadmat


class lensless_dataset(Dataset):
    def __init__(self, hparams, split='train'):
        self.hparams = hparams;
        self.split = split
        self.read_meta()
        # print(self.hparams.root_dir)
        
    def read_meta(self):      
        # grid
        [x, y] = torch.meshgrid(torch.linspace(-1, 1, self.hparams.ctf_wh[1]), torch.linspace(-1, 1, self.hparams.ctf_wh[0]))
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        x = x.view(1, self.hparams.ctf_wh[1], self.hparams.ctf_wh[0], 1)
        y = y.view(1, self.hparams.ctf_wh[1], self.hparams.ctf_wh[0], 1)
        xy = torch.cat([x, y], dim=-1)     # (1, h, w, 2)
        
        self.rawimg_max = torch.tensor(1.)
        if self.hparams.simulation:    # simulation experiment
            path_RawImgSetMax = os.path.join(self.hparams.root_dir, 'RawImgSetMax.mat')
            RawImgSetMax = torch.tensor((loadmat(path_RawImgSetMax))['RawImgSetMax'], dtype=torch.float32)
            self.rawimg_max = RawImgSetMax[0]
            # print(self.rawimg_max)
        
        if self.split == 'train':
            self.all_rgbs = []  # raw images
            self.all_ctfs = []  # ctfs 
            self.img_h = self.hparams.img_wh[1]
            self.img_w = self.hparams.img_wh[0]            
       
            # shape: (h, w, 8(default))
            ctf_path = os.path.join(self.hparams.root_dir, 'PropCTFSet.mat')
            ctf_set = loadmat(ctf_path)            
            ctfs_all = ctf_set['PropCTFSet']
            ctfs_all = ctfs_all[:,:,0:self.hparams.img_num];
       
            for imgid in range(self.hparams.img_num):            
                img_path = os.path.join(self.hparams.root_dir, 'RawImg_' + str(imgid + 1).zfill(2) + '.png')
                img = ((cv2.imread(img_path, -1)).astype(np.float32))/255.
          
                img = np.reshape(img, [self.img_h, self.img_w, 1])
                img = np.transpose(img, [2,0,1])
                img = torch.tensor(img)   # (1, h, w)
                self.all_rgbs += [img]

                ctf_t = ctfs_all[:, :, imgid]
                ctf = np.zeros([1, self.hparams.ctf_wh[1], self.hparams.ctf_wh[0]], dtype=np.complex64) # (1, h, w) complex64
                ctf[0, :, :] = ctf_t[::-1, ::-1]
                ctf = torch.tensor(ctf)
                self.all_ctfs.append(ctf)
                
            self.all_rgbs = torch.cat(self.all_rgbs, dim=0)          # (8, h, w)
            self.all_rays = xy                                       # (1, h, w, 2)
            self.all_ctfs = torch.cat(self.all_ctfs, dim=0)          # (8, h, w)
        else:
            # GT object
            img_path = os.path.join(self.hparams.root_dir, 'GT_amp.tif')
            img_amp = ((plt.imread(img_path)).astype(np.float32))/255.

            img_path = os.path.join(self.hparams.root_dir, 'GT_phs.tif')
            img_phase = ((plt.imread(img_path)).astype(np.float32))/255.
            img_phase = img_phase * math.pi * self.hparams.pai_scale;
            
            input_img = np.concatenate([img_amp[np.newaxis, :,:, np.newaxis], img_phase[np.newaxis, :,:, np.newaxis]], axis=-1)
            input_img = torch.tensor(input_img)     # [1, h, w, 2]

            self.all_rgbs = input_img    # (1, h, w, 2) amp, phase
            self.all_rays = xy           # (1, h, w, 2)

    def __len__(self):
        return self.all_rgbs.shape[0]

    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {'rays': self.all_rays[0, ...],          # [h, w, 2]
                      'rgbs': self.all_rgbs[idx, ...],        # [h, w]
                      'ctfs': self.all_ctfs[idx, ...],        # [h, w]
                      'rmax': self.rawimg_max,                 
                      'zidx': idx
                     }
        else:
            sample = {'rays': self.all_rays[0, ...],          # [h, w, 2]
                      'rgbs': self.all_rgbs[idx, ...]         # [h, w, 2]
                     }
        return sample


