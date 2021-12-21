'''
Since we are only using image weights for downstream tasks, this
dataset will only encompass images and mortality
'''

import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
import random
from numbers import Number
import pytorch_lightning as pl
from typing import Optional

cxr_dir = '../data/mimic-cxr-jpg/'
mimic_dir = '../data/mimic-iv/'

class MortalityDataset(Dataset):
    '''
    Dataset encapsulating all multi-modal data available for patients
    in the MIMIC_CXR-JPG dataset
    '''
    
    def __init__(self, config, split, neg_class_frac=1.0, class_ratio=None, image_size=(224,224),
                 records_csv='../data/saved/records_new.csv',
                 split_csv='../data/saved/new_splits.csv', indexed_vocab=True,
                 tab_preprocessor=None, prepared_data=False, imagenet=False, train_ver=1):
        
        super().__init__()
        
                
        assert split in ('train','val','validation','validate','test','all')
        assert neg_class_frac <= 1.0
        self.split = split if split not in ['validation','val'] else 'validate'
        self.imagenet = imagenet
        self.train_ver = train_ver
        
        # Verified that splits csv has some ordering of images as records csv, 
        # so this is a safe way to filter to the split
        self.labels = pd.read_csv(mimic_dir+'core/patients.csv', index_col='subject_id',
                                      usecols=['dod','subject_id'])
        self.labels = (~self.labels.dod.isna()).astype('int')
        
        if self.split == 'all':
            self.records = pd.read_csv(records_csv)
        else:
            splits_mortality = pd.read_csv('../data/saved/splits_mortality_3split.csv')
            splits_all = pd.read_csv(split_csv)
            if config.use_all_data: # Use the train/val/test splits of split_new
                self.records = pd.read_csv(records_csv) \
                    .loc[splits_all.split_new == self.split].reset_index()
            else: # Re-split the test set of split_new, i.e. use split_new_mortality
                self.records = pd.read_csv(records_csv) \
                    .loc[splits_all.split_new == 'test'].reset_index()
                self.records = self.records[pd.read_csv(splits).split_new_mortality==self.split]
                
            if neg_class_frac < 1.0: # Specified what fraction of negative labels to use
                pos_ids = self.labels[self.labels == 1].index
                pos_records = self.records[self.records.subject_id.isin(pos_ids)]
                neg_records = self.records.drop(index=pos_records.index).sample(frac=neg_class_frac)
                self.records = pd.concat([pos_records, neg_records], axis=0)
            elif class_ratio is not None and split=='train': # Specified desires neg:pos class ratio
                pos_ids = self.labels[(self.labels == 1) & 
                                      (self.labels.index.isin(self.records.subject_id))].index
                pos_records = self.records[self.records.subject_id.isin(pos_ids)]
                neg_ids = self.labels[(self.labels == 0) &
                                      (self.labels.index.isin(self.records.subject_id))] \
                    .sample(n=int(len(pos_ids)*class_ratio)).index
                neg_records = self.records[self.records.subject_id.isin(neg_ids)]
                self.records = pd.concat([pos_records, neg_records], axis=0)
                print(self.labels[self.labels.index.isin(self.records.subject_id)].value_counts())
                
        
        self.image_size = image_size
        if self.split in ['train', 'all']:
            # Transforms used are based on SimCLR:
            # https://github.com/sthalles/SimCLR/tree/
            ks = int(0.1*self.image_size[0])
            ks = ks - ((ks+1)%2)
            
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=self.image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine((-20, 20),
                                        translate=(0.1, 0.1),
                                        scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6,1.4)),
                transforms.GaussianBlur(kernel_size=ks, sigma=(0.1,3.0))
            ])
                
        else:
            self.transforms = nn.Identity()
            
        self.totensor = transforms.ToTensor()
        
    def __len__(self):
        return self.records.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        subset = self.records.iloc[idx]
        subject = subset['subject_id']
        
        # Image handling
        ## Greyscale for now, change when using retrained weight
        image = Image.open(cxr_dir+subset['path_compressed'][:-3]+'jpg')
        image = self.totensor(image)
        if self.imagenet or self.train_ver > 1:
            image = image.repeat(3,1,1)
        image = self.transforms(image) 
        #image = self.preprocess_image(image)
        
        try:
            label = self.labels[subject]
        except:
            label = 0
        
        return image, label
    
    def preprocess_image(self, image):
        # Uses weighted mean and std from MIMIC-CXR-JPG, Stanford, NIH Chest X-ray data:
        image = self.normalizer(image)
        #image = F.interpolate(image.unsqueeze(0), size=self.image_size).squeeze(0)
        #image = image.repeat(3, 1, 1)
        
        # Downsampling to make all images the same size
        # Note that interpolate wants images in shape (batch_size, num_channels, *additional_dims)
        return image

class MortalityDataModule(pl.LightningDataModule):
    '''
    Parameters:
        split: bool or iterable. If True, creates train/test/val dataloaders;
            If False, creates a single dataloader for entire dataset;
            If iterable, creates only the dataloaders specified
        batch_size: Batch size
        num_workers: Number of workers
        tokenizer: The tokenizer to use for BERT
    '''
    def __init__(self, config, imagenet=False, train_ver=1, class_ratio=None):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.imagenet = imagenet
        self.train_ver = train_ver
        self.class_ratio = class_ratio
        
    def setup(self, stage:Optional[str]=None) -> None:
        self.train_data = MortalityDataset(config=self.config, split='train',
                                           imagenet=self.imagenet, train_ver=self.train_ver,
                                           class_ratio=self.class_ratio)
        self.val_data = MortalityDataset(config=self.config, split='validate',
                                           imagenet=self.imagenet, train_ver=self.train_ver,
                                           class_ratio=self.class_ratio)
        self.test_data = MortalityDataset(config=self.config, split='test',
                                           imagenet=self.imagenet, train_ver=self.train_ver,
                                           class_ratio=self.class_ratio)

        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True)
                
                