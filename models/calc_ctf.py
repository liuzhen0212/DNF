import torch
from torch import nn
from math import pi
import os, sys
import numpy as np


class CTF(nn.Module):
    def __init__(self, hparams):
        super(CTF, self).__init__()
        self.root_dir  = hparams.root_dir
        self.exp_name  = hparams.exp_name
        self.h         = hparams.ctf_wh[1]
        self.w         = hparams.ctf_wh[0]
        self.num       = hparams.img_num
    
        # for better convergence
        self.z_scale = 0.1
        
        if hparams.simulation == False:
            # real-world data (precision board)
            self.wavelength = 638e-9   # unit: m
            self.UpSampleRatio = 2.0
            self.pixel_size = 1.67e-6 / self.UpSampleRatio
            
            # self.z_matlab         = torch.tensor([[0.6480, 0.6820, 0.7340, 0.7760, 0.8200, 0.8710, 0.9140, 0.9650]]) * self.z_scale    # generate by matlab
            self.z_scale = 1.0
            self.z = nn.Parameter(torch.tensor([[0.6614, 0.7278, 0.6775, 0.7976, 0.8280, 0.8383, 0.9032, 0.9736]]) * self.z_scale, requires_grad=True)  # add noise (randnom) sigma = 0.025 
        else:
            # simulation data (rabbit, dog)
            self.wavelength = 537e-9 # unit: m
            self.UpSampleRatio = 1.0
            self.pixel_size = 3.45e-6 / self.UpSampleRatio
            # self.z_GT = (torch.linspace(0, 350, 8).reshape(1, num) * 1e-3 + 0.5) * self.z_scale # GT 
            self.z = nn.Parameter((torch.ones(1, self.num) * 175 * 1e-3 + 0.5) * self.z_scale, requires_grad=True)        # init: mid value
       
        self.cnt = 0
        self.printZ_atfirst = True
        
    def forward(self, zidx):
    
        if self.printZ_atfirst == True:
            print(self.z.cpu().detach().numpy())
            self.printZ_atfirst = False
        
        wavelength = self.wavelength
        pixel_size = self.pixel_size
        k0 = 2 * pi / wavelength
        m = self.h
        n = self.w
        
        kx = torch.linspace(-pi/pixel_size, pi/pixel_size, steps=n)
        ky = torch.linspace(-pi/pixel_size, pi/pixel_size, steps=m)
        kxm, kym = torch.meshgrid(kx, ky)
        kzm = torch.sqrt(k0 ** 2 - torch.pow(kxm, 2) - torch.pow(kym, 2))
        kzm = kzm.cuda()
        
        batch_size = zidx.shape[0]
        ctfs = []
        for i in range(batch_size):
            ctf = torch.exp(1j * torch.mul(kzm, self.z[0][zidx[i]]) / (1000 * self.z_scale))  # [m, n]
            ctfs.append(ctf)
        
        ctfs = torch.stack(ctfs) # [batch, m, n]
          
        # output
        if self.cnt % 800 == 799 or self.cnt == 0:
            z_learned = self.z.cpu().detach().numpy()
            path = os.path.join(self.root_dir, 'zlog', self.exp_name, 'zlog_' + str(self.cnt).zfill(6) + '.txt')
            np.savetxt(path, z_learned)
        
        self.cnt = self.cnt + 1

        return ctfs
