import numpy as np
import os
import torch
import lightning as L
from torch.utils.data import Dataset,DataLoader


class CustomDataset(torchDataset):
    def __init__(self,
                 dataset_name = "text8",
                 split = "train"):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        if self.dataset_name == "text8":
            self.X = torch.from_numpy(np.loadtxt(f"../USD3_rebuttal/data/text8/{split}.txt", dtype=np.int32))
        elif self.dataset_name == "cifar10":
            self.X = torch.from_numpy(np.load(f'../USD3_rebuttal/data/cifar10/{split}.npy')).long()
        elif self.dataset_name == "piano":
            self.X = torch.from_numpy(np.load(f'../USD3_rebuttal/data/piano/{split}.npy')).long()
            
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,idx):
        return self.X[idx]
    
class DataModule(L.LightningDataModule):
    def __init__(self,
                 config,
                 train_dataset=None,
                 val_dataset=None,
                 test_dataset=None):
        super().__init__()
        self.config = config
        if train_dataset is not None:
            self.train_dataset = train_dataset
            self.train_dataloader=self._train_dataloader
        if val_dataset is not None:
            self.val_dataset = val_dataset
            self.val_dataloader = self._val_dataloader
        if test_dataset is not None:
            self.test_dataset = test_dataset
            self.test_dataloader = self._test_dataloader
    
    @staticmethod
    def get_dataloader(dataset,shuffle,batch_size,num_workers,drop_last):
        return DataLoader(dataset=dataset,
                          num_workers=num_workers,
                          batch_size=batch_size,
                          drop_last=drop_last)
        
    def _train_dataloader(self):
        return self.get_dataloader(self.train_dataset,
                                   shuffle=True,
                                   num_workers=self.config.data.train_num_workers,
                                   batch_size = self.config.data.train_batch_size,
                                   drop_last = False)
        
    def _test_dataloader(self):
        return self.get_dataloader(self.test_dataset,
                                   shuffle=False,
                                   num_workers=self.config.data.test_num_workers,
                                   batch_size = self.config.data.test_batch_size,
                                   drop_last = False)
        
    def _val_dataloader(self):
        return self.get_dataloader(self.val_dataset,
                                   shuffle=True,
                                   num_workers=self.config.data.val_num_workers,
                                   batch_size = self.config.data.val_batch_size,
                                   drop_last = False)
    
    # @property
    # def prepare_data_per_node(self):
    #     return False
    
    
