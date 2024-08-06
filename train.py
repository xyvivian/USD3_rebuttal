import torch
import numpy as np
import argparse
import torch
torch.set_num_threads(24)
import ml_collections
import os
from pathlib import Path
import logging
import lightning as L
import sys
from trainer.trainer import DiscreteDiffusionTrainer
from dataloader.dataloader import DataModule,CustomDataset
from pytorch_lightning.loggers import WandbLogger

from optimizer.ema import EMACallback
import sys
import datetime

logging.basicConfig(
    level='INFO',
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('file.log')]
)
logger = logging.getLogger(__name__)


def main(cfg,checkpoint_path):
    gpu = [4] #[0,2] #list(range(1,1+cfg.training.num_gpus))
    #precision = 32
    #if cfg.training.enable_16_precision:
    precision = 'bf16'
        
    #logger.info(os.environ['RANK'])
    #logger.info(os.environ["WORLD_SIZE"])
    device_count = torch.cuda.device_count()
    cur_device = torch.cuda.current_device()
    logger.info(f"Device Count:{device_count}, Current Device: {cur_device}")
    
    train_model = DiscreteDiffusionTrainer(cfg)
    wandb_logger = WandbLogger(project=cfg.exp_name)
    cur_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S%z")
    output_dir = f"{cur_time}_{cfg.exp_name}"
    
    train_dataset = CustomDataset(dataset_name = cfg.dataset_name,split='train')
    valid_dataset = CustomDataset(dataset_name = cfg.dataset_name,split='valid')
    data_module = DataModule(cfg,
                             train_dataset=train_dataset,
                             val_dataset=valid_dataset,
                             )
    
    callbacks = [L.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
                 L.pytorch.callbacks.ModelCheckpoint(dirpath = output_dir + "/checkpoints",
                                                     save_weights_only=False,
                                                     monitor='valid/NLL_epoch',
                                                     mode = 'min',
                                                     save_top_k=5,
                                                     filename = '{epoch:02d}-{valid/NLL:.3f}'
                                                     ),
                 L.pytorch.callbacks.ModelCheckpoint(dirpath= output_dir + "/checkpoints",
                                                     monitor='step',
                                                     save_top_k=5,
                                                     mode='max',
                                                     every_n_train_steps = 1000),
                 L.pytorch.callbacks.ModelCheckpoint(dirpath=output_dir + "/checkpoints",
                                                     monitor='valid/loss_epoch',
                                                     filename = '{epoch:02d}-{valid/loss_epoch:.3f}',
                                                     save_top_k = 5),
                 EMACallback(parameters=train_model.module.parameters(),
                             wait_steps = cfg.training.ema_wait_steps,
                             decay = cfg.training.ema_decay)]
    
    trainer = L.Trainer(strategy='ddp',
                        max_steps = cfg.training.num_training_steps,
                        devices = gpu,
                        callbacks = callbacks,
                        log_every_n_steps = 10,
                        logger=wandb_logger,
                        enable_progress_bar=True,
                        gradient_clip_val=1.0,
                        accumulate_grad_batches=1,
                        gradient_clip_algorithm='norm',
                        precision=precision,
                        num_sanity_val_steps=1,
                        # limit_train_batches=5,
                        # limit_val_batches=5,
                        )
    
    
    if checkpoint_path is not None:
        logger.info("resume training from {checkpoint_path}")
        trainer.fit(train_model, ckpt_path = checkpoint_path, datamodule=data_module)
    else:
        trainer.fit(train_model,datamodule=data_module)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Discrete Diffusion')
    parser.add_argument('--checkpoint_path', type=str,default=None)
    parser.add_argument('--simplified_vlb',action='store_true',default=False)
    parser.add_argument('--data',type=str,default='text8')
    parser.add_argument('--simplified_max_val',type=float,default=1.0)
    args = parser.parse_args()
    if args.data =='text8':
        from config.text8_train import get_config
    elif args.data == 'piano':
        from config.piano_train import get_config
    elif args.data == 'cifar10':
        from config.cifar10_train import get_config
    cfg =get_config()
    cfg.simplified_vlb = args.simplified_vlb
    cfg.simplified_max_val = args.simplified_max_val
    cfg.exp_name = f'{cfg.model.name}_{cfg.data.name}_lr_{cfg.training.lr}_wd_{cfg.training.weight_decay}_nll_{cfg.diffusion.nll_weight}_l2_{cfg.simplified_vlb}_{cfg.simplified_max_val}'
    main(cfg,checkpoint_path=args.checkpoint_path)
    