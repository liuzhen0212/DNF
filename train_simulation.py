import os, sys
from opt import get_opts
import torch
from torch import nn
from torch import fft
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict
import cv2

# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays_lensless_2stream
from models.calc_ctf import CTF

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.logging import TestTubeLogger
from math import pi

import numpy as np

import time
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams

        self.loss = loss_dict[hparams.loss_type]()
        self.freq_amp = 20
        self.freq_phase = 20
        self.embedding_amp = Embedding(2, self.freq_amp) 
        self.embedding_phase = Embedding(2, self.freq_phase) 
        self.embeddings = [self.embedding_amp, self.embedding_phase]

        self.nerf_amp = NeRF(W=256, in_channels_xy=self.freq_amp*2*2+2)    
        self.nerf_phase = NeRF(W=256, in_channels_xy=self.freq_phase*2*2+2) 
        self.calc_ctf = CTF(hparams)
        self.models = [self.nerf_amp, self.nerf_phase, self.calc_ctf]
        
        self.batch_cnt = 0
                
    def decode_train_batch(self, batch):
        rays = batch['rays']        # [batch_size, h, w, 2]
        rgbs = batch['rgbs']        # [batch_size, h, w] 
        ctfs = batch['ctfs']        # [batch_size, h, w]
        return rays, rgbs, ctfs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B, kernel_h, kernel_w, chans = rays.shape # [batch, kernel_h, kernel_w, 2]
        rays = rays.view(-1,chans)
        B = rays.shape[0]       
        results_amp = []
        results_phase = []
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks_amp, rendered_ray_chunks_phase = \
                render_rays_lensless_2stream(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk])

            results_amp.append(rendered_ray_chunks_amp)
            results_phase.append(rendered_ray_chunks_phase)
        results_amp = torch.cat(results_amp, dim=0)
        results_phase = torch.cat(results_phase, dim=0)

        results_amp = results_amp.view(-1, kernel_h, kernel_w)
        results_phase = results_phase.view(-1, kernel_h, kernel_w)

        return results_amp, results_phase

    def prepare_data(self):     # OK
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'hparams': self.hparams }
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):         # OK
        return DataLoader(self.train_dataset,
                          shuffle = True,
                          num_workers = 4,
                          batch_size = self.hparams.batch_size,
                          pin_memory = True)
    
    def val_dataloader(self):           # OK
        return DataLoader(self.val_dataset,
                          shuffle = False,
                          num_workers = 4,
                          batch_size = 1, # validate one image (H*W rays) at a time
                          pin_memory = True)
    
    
    def training_step(self, batch, batch_idx):
        log = {'lr': get_learning_rate(self.optimizer)}
                
        rays, rgbs, ctfs = self.decode_train_batch(batch)
        rawimg_max = batch['rmax']
        zidx = batch['zidx']  
        batch_size, row, col = ctfs.shape
        
        rawimg_max = torch.reshape(rawimg_max, [batch_size, 1])
        zidx = torch.reshape(zidx, [batch_size, 1])
        
        if self.hparams.joint_training:    
            # joint training to optimize z(distance)
            model_calc_ctf = self.models[2]
            ctfs = model_calc_ctf(zidx) 
                
        results_amp, results_phase = self(rays)          # [batch, h, w]
        results_phase_copy = results_phase
        results_phase = results_phase * pi * self.hparams.pai_scale;
        
        # pred object
        results_real = results_amp * torch.cos(results_phase)
        results_imag = results_amp * torch.sin(results_phase)
        results_complex = torch.complex(results_real, results_imag)  # [batch, h, w]
        
        # CTF
        CTF_complex = ctfs  # [batch, h, w]
        
        observes = []     
        for i in range(batch_size):
            # CTF i
            CTF_i = CTF_complex[i, ...]
            pred_obj_i = results_complex[i, ...]
            
            objFT     = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(pred_obj_i)))
            objFT_t   = torch.mul(objFT, CTF_i)
            observe_i = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(objFT_t)))
               
            observes.append(observe_i)
 
        observesImg = torch.stack(observes)  # [batch, h, w]
        observes_amp = torch.pow(observesImg.real, 2) + torch.pow(observesImg.imag, 2)

        if self.hparams.simulation:
            rawimg_max = rawimg_max[0][0]
            observes_amp = observes_amp / rawimg_max
        
        # output ------------------------------
        if self.batch_cnt % 2000 == 1999:
            image_out_path = os.path.join(self.hparams.root_dir, 'image_out', self.hparams.exp_name)
        
            result_amp_np = results_amp.cpu().detach().numpy();
            result_amp_np[result_amp_np>1] = 1
            result_amp_np[result_amp_np<0] = 0
            result_amp_np = result_amp_np * 255
            result_amp_np = result_amp_np.astype(np.uint8)
            result_amp_np_i = result_amp_np[0, :]; 
            img_path = os.path.join(image_out_path, 'amp_'+ str(self.batch_cnt).zfill(6) + '.png')
            cv2.imwrite(img_path, result_amp_np_i)
            
            result_phase_np = results_phase_copy.cpu().detach().numpy();
            result_phase_np[result_phase_np>1] = 1
            result_phase_np[result_phase_np<0] = 0
            result_phase_np = result_phase_np * 255
            result_phase_np = result_phase_np.astype(np.uint8)
            result_phase_np_i = result_phase_np[0, :]; 
            img_path = os.path.join(image_out_path, 'phase_'+ str(self.batch_cnt).zfill(6) + '.png')
            cv2.imwrite(img_path, result_phase_np_i)
        
            results_conv_amp_final_np = observes_amp.cpu().detach().numpy()
            results_conv_amp_final_np[results_conv_amp_final_np>1] = 1
            results_conv_amp_final_np[results_conv_amp_final_np<0] = 0
            results_conv_amp_final_np = results_conv_amp_final_np*255
            results_conv_amp_final_np = results_conv_amp_final_np.astype(np.uint8)    
            results_conv_amp_final_np_i = results_conv_amp_final_np[0, ...]
            img_path = os.path.join(image_out_path, 'observe_'+ str(self.batch_cnt).zfill(6) + '.png')
            cv2.imwrite(img_path, results_conv_amp_final_np_i)
        
        self.batch_cnt += 1
        # output ------------------------------                

        # shape [batch, h, w]
        log['train/loss'] = loss = self.loss(observes_amp, rgbs)

        with torch.no_grad():
            psnr_ = psnr(observes_amp, rgbs)
            log['train/psnr'] = psnr_
            
        return {'loss': loss*1000,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
               }

    
    def validation_step(self, batch, batch_nb):
        '''comparison for reconstructed light fields'''
        rays = batch['rays']        # [batch, kernel_h, kernel_w, 2]
        rgbs = batch['rgbs']        # [batch, kernel_h, kernel_w, 2], amp & phase

        results_amp, results_phase = self(rays)
        results_phase = results_phase * pi * self.hparams.pai_scale;

        gt_amp = rgbs[..., 0]
        gt_phase = rgbs[..., 1]
        
        val_loss_amp = self.loss(results_amp, gt_amp)
        val_loss_phase = self.loss(results_phase, gt_phase)
        val_loss = val_loss_amp + val_loss_phase
        
        log = {'val_loss': val_loss}
        log['val_loss_amp_rmse'] = torch.sqrt(val_loss_amp)
        log['val_loss_angle_rmse'] = torch.sqrt(val_loss_phase)

        return log

    def validation_epoch_end(self, outputs):

        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_loss_amp_rmse = torch.stack([x['val_loss_amp_rmse'] for x in outputs]).mean()
        mean_loss_angle_rmse = torch.stack([x['val_loss_angle_rmse'] for x in outputs]).mean()
        
        return {'progress_bar': {'val_loss': mean_loss,
                                 'loss_amp_rmse': mean_loss_amp_rmse,
                                 'loss_angle_rmse': mean_loss_angle_rmse},
                'log': {'val/loss': mean_loss,
                        'val/loss_amp': mean_loss_amp_rmse,
                        'val/loss_angle': mean_loss_angle_rmse}
               }

if __name__ == '__main__':
    hparams = get_opts()
    hparams.simulation = True
    
    image_out_path = os.path.join(hparams.root_dir, 'image_out', hparams.exp_name)
    # print (image_out_path)
    if not os.path.exists(image_out_path):
        os.makedirs(image_out_path) 
    
    if hparams.joint_training:
        zlog_out_path = os.path.join(hparams.root_dir, 'zlog', hparams.exp_name)
        if not os.path.exists(zlog_out_path):
                os.makedirs(zlog_out_path) 
    
    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(hparams.root_dir, f'ckpts/{hparams.exp_name}', '{epoch:d}'),
                                                                monitor='val/loss',
                                                                mode='min',
                                                                save_top_k=10)
    logger = TestTubeLogger(
        save_dir=os.path.join(hparams.root_dir, f'logs'),
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False
    )
    # print(os.path.join(hparams.root_dir, f'logs'))
    
    trainer = Trainer(max_epochs=hparams.num_epochs,
                    checkpoint_callback=checkpoint_callback,
                    resume_from_checkpoint=hparams.ckpt_path,
                    logger=logger,
                    early_stop_callback=None,
                    weights_summary=None,
                    progress_bar_refresh_rate=1,
                    gpus=hparams.num_gpus,
                    distributed_backend='ddp' if hparams.num_gpus>1 else None,
                    num_sanity_val_steps=1,
                    benchmark=True,
                    amp_level = 'O2',
                    profiler=hparams.num_gpus==1,
                    log_every_n_steps=1)

    trainer.fit(system)