import numpy as np
import torch
import lightning as L
import ml_collections
import os
from model.transformer import DDitTransformer,TransformerEncoder
from diffusion.discrete_diffusion import *
from optimizer.lr_scheduler import get_cosine_with_hard_restarts_schedule_with_warmup
from metrics.bpc import bpc

class DiscreteDiffusionTrainer(L.LightningModule):
 
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.condition_dim = config.diffusion.condition_dim
        self.m = None
        if self.config.model.name == 'TransformerEncoder':
            self.module = TransformerEncoder(self.config)
        else:
            self.module = DDitTransformer(self.config)
        self.num_classes = config.diffusion.num_classes
        if self.config.diffusion.noise_type== 'absorb':
            self.num_classes = self.num_classes+ 1
            self.m = torch.zeros((1,self.num_classes),dtype=float)
            self.m[:,-1]=1
        self.diffusion = UnifiedDiscreteDiffusion(num_steps = config.diffusion.num_steps,
                                                  num_classes = self.num_classes,
                                                  noise_schedule_type=config.diffusion.noise_schedule_type,
                                                  noise_schedule_args=config.diffusion.noise_schedule_args,
                                                  simplified_max_val=config.diffusion.simplified_max_val
                                                  )
        self.optimizer = torch.optim.AdamW(self.module.parameters(),
                                           lr = self.config.training.lr,
                                           betas = (self.config.training.beta1, self.config.training.beta2),
                                           weight_decay=self.config.training.weight_decay,
                                           eps=1e-6)
        self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                                         self.optimizer,
                                         num_warmup_steps = self.config.training.num_warmup_steps,
                                         num_training_steps = self.config.training.num_training_steps,
                                        )
        
    
    def _get_m(self,batch_size,feature_size, device):
        m = None if self.m is None else self.m.reshape(-1,self.num_classes).repeat(batch_size,feature_size,1).to(device)
        return m

        
    def forward(self,batch,batch_idx=None):
        batch_size = batch.shape[0]
        feature_size = batch.shape[1]
        x_0 = batch.view(batch_size,-1).long()
        m = self._get_m(batch_size,feature_size, batch.device)
        conditional_mask = None
        if self.condition_dim > 0:
            conditional_mask = torch.zeros_like(x_0, dtype=torch.bool)
            conditional_mask[:, :self.condition_dim] = True

        #sample time steps
        if self.diffusion.num_steps == 0: 
            t = torch.rand((batch_size,), device=batch.device)
            t = torch.clip(t, min=self.config.diffusion.min_time)
        else:
            t = torch.randint(1, self.diffusion.num_steps+1, (batch_size,), device=batch.device)

        # forward sampling
        x_t = self.diffusion.qt_0_sample(x_0, t, m, conditional_mask=conditional_mask)  
        time_steps = t
        if self.diffusion.num_steps != 0:
            time_steps =  t/torch.tensor([self.diffusion.num_steps],dtype=torch.float,device=t.device)
        net_out = self.module(x_t, time_steps) #DDiT transformer expects noise, but we put in time here
        
        losses = self.diffusion.compute_loss(net_out,
                                             x_t,
                                             x_0, 
                                             t, 
                                             m, 
                                             coeff_ce=self.config.diffusion.nll_weight, 
                                             conditional_mask=conditional_mask,
                                             simplified_vlb=self.config.simplified_vlb) 
        return losses
    
    def training_step(self,batch,batch_idx=None):
        loss = self(batch)
        self.log('train/loss',
                 loss['loss'],
                 on_step = True, 
                 prog_bar=True,
                 logger=True, 
                 sync_dist = True,
                 rank_zero_only=True,)
        self.log('train/ce',
                 loss['ce_loss'],
                 on_step = True, 
                 prog_bar=True,
                 logger=True, 
                 sync_dist = True,
                 rank_zero_only=True,)
        self.log('train/vlb',
                 loss['vlb_loss'],
                 on_step = True, 
                 prog_bar=True,
                 logger=True, 
                 sync_dist = True,
                 rank_zero_only=True,)
        return loss['loss']
    
    
    def validation_step(self,batch,batch_idx,dataloader_idx=0):
        #use one validation step for sampling eval
        loss = self(batch)
        self.log('valid/loss',
                 loss['loss'],
                 on_step=True,
                 prog_bar = True,
                 logger = True,
                 sync_dist = True,
                 rank_zero_only=True,
                 )
        self.log('valid/ce',
                 loss['ce_loss'],
                 on_step=True,
                 prog_bar = True,
                 logger = True,
                 sync_dist = True,
                 rank_zero_only=True,
                 )
        self.log('valid/vlb',
                 loss['vlb_loss'],
                 on_step=True,
                 prog_bar = True,
                 logger = True,
                 sync_dist = True,
                 rank_zero_only=True,
                 )
        if self.sample_step_done:
            return None  
        N,D = batch.shape[0],batch.shape[1]
        
        conditional_mask = None
        
        if self.config.diffusion.condition_dim != 0:
            conditional_mask = torch.zeros(batch.shape, dtype=torch.bool).to(batch.device)
            conditional_mask[:,:self.config.diffusion.condition_dim] = 1
        
        samples = self.sample(N,
                              D,
                              batch,
                              batch.device,
                              num_intermediates=self.config.sampler.num_intermediates,
                              conditional_mask=conditional_mask)
        
        nll = bpc(batch,samples,num_classes = self.num_classes)
        self.log('valid/NLL',
                 nll,
                 on_step=True,
                 prog_bar = True,
                 logger = True,
                 sync_dist = True,
                 rank_zero_only=True,
                 )
        self.valid_nll_loss.append(nll)
        self.valid_sample_step+=1
        if self.valid_sample_step == 1:
            self.sample_step_done = True
        return nll
    
    def on_validation_epoch_start(self):
        self.sample_step_done = False
        self.valid_sample_step = 0
        self.valid_nll_loss =[]
        
    def on_validation_epoch_end(self):
        self.log('valid/NLL_epoch',
                 torch.mean(torch.stack(self.valid_nll_loss)),
                 on_step=False,
                 prog_bar = True,
                 logger = True,
                 sync_dist = True,
                 rank_zero_only=True,
                 )
    
    
    def configure_optimizers(self):
        return {
         "optimizer":  self.optimizer,
         "lr_scheduler":{"scheduler": self.scheduler,
                         "monitor": "train_loss",
                         "interval": "step"}   
        }
        
        
    def sample(self,
               N,
               D,
               batch=None,
               device='cuda',
               num_intermediates=None,
               conditional_mask=None):
        m = self._get_m(N,D,device)
        if m is None:
            if self.config.diffusion.noise_type == 'uniform':
                m =  1 / self.num_classes * torch.ones((N,D,self.num_classes),\
                dtype=float).to(device)
            elif self.config.diffusion.noise_type == 'absorb':
                m = torch.zeros((N,D,self.num_classes),dtype=float).to(device)
                m[:,:,-1]=1
            else:
                raise NotImplementedError
        return self.diffusion.sample(
                                    denoising_fn = self.module,
                                    num_backward_steps=num_intermediates,
                                    m=m,
                                    conditional_mask = conditional_mask,
                                    conditional_input= batch,
                                    mcmc_step_size = self.config.sampler.mcmc_step_size,
                                    mcmc_num_steps = self.config.sampler.mcmc_num_steps,
                                    mcmc_start_ratio = self.config.sampler.mcmc_start_ratio) 
        
        