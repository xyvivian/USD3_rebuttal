import torch
import argparse
import torch
torch.set_num_threads(24)
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
    precision = 'bf16'
    device_count = torch.cuda.device_count()
    cur_device = torch.cuda.current_device()
    logger.info(f"Device Count:{device_count}, Current Device: {cur_device}")
    
    train_model = DiscreteDiffusionTrainer(cfg)
    wandb_logger = WandbLogger(project='USD3', name=cfg.exp_name, log_model=False, config=cfg)
    cur_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
    output_dir = f"{cfg.exp_name}_{cur_time}"
    
    train_dataset = CustomDataset(dataset_name = cfg.dataset_name,split='train')
    valid_dataset = CustomDataset(dataset_name = cfg.dataset_name,split='valid')
    data_module = DataModule(cfg,
                             train_dataset=train_dataset,
                             val_dataset=valid_dataset,
                             )
    
    callbacks = [L.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
                 L.pytorch.callbacks.ModelCheckpoint(dirpath = os.path.join('checkpoints', output_dir),
                                                     save_weights_only=False,
                                                     monitor='valid/NLL_epoch',
                                                     mode = 'min',
                                                     save_top_k=5,
                                                     filename = '{epoch:02d}-{valid/NLL:.3f}'
                                                     ),
                 L.pytorch.callbacks.ModelCheckpoint(dirpath= os.path.join('checkpoints', output_dir),
                                                     monitor='step',
                                                     save_top_k=5,
                                                     mode='max',
                                                     every_n_train_steps = 1000),
                 L.pytorch.callbacks.ModelCheckpoint(dirpath=os.path.join('checkpoints', output_dir),
                                                     monitor='valid/loss_epoch',
                                                     filename = '{epoch:02d}-{valid/loss_epoch:.3f}',
                                                     save_top_k = 5),
                 EMACallback(parameters=train_model.module.parameters(),
                             wait_steps = cfg.training.ema_wait_steps,
                             decay = cfg.training.ema_decay)]
    
    trainer = L.Trainer(strategy='ddp',
                        max_steps = cfg.training.num_training_steps,
                        devices = args.gpus,
                        callbacks = callbacks,
                        logger=wandb_logger,
                        enable_progress_bar=True,
                        gradient_clip_val=1.0,
                        accumulate_grad_batches=1,
                        gradient_clip_algorithm='norm',
                        precision=precision,
                        check_val_every_n_epoch=20,
                        # val_check_interval=10, 
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
    parser.add_argument('--gpus',type=int,default=[0], nargs='*', help='GPUs to use')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--continuous',action='store_true',default=False)

    args = parser.parse_args()
    if args.data =='text8':
        from config.text8_train import get_config
    elif args.data == 'piano':
        from config.piano_train import get_config
    elif args.data == 'cifar10':
        from config.cifar10_train import get_config
    cfg = get_config()
    
    if args.continuous:
        cfg.diffusion.num_steps = 0
        type='continuous'
    else:
        cfg.diffusion.num_steps = 1000
        type='discrete'
    
    cfg.simplified_vlb = args.simplified_vlb
    cfg.simplified_max_val = args.simplified_max_val
    cfg.training.lr = args.lr
    cfg.exp_name = f'{cfg.model.name}_{cfg.data.name}_lr{cfg.training.lr}_nll{cfg.diffusion.nll_weight}_l2{cfg.config.simplified_vlb}_{type}'
    main(cfg,checkpoint_path=args.checkpoint_path)
    