import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer
from pytorch_widedeep.preprocessing import TabPreprocessor
import random
from numbers import Number
from functools import partial
import pytorch_lightning as pl
import pickle


cxr_dir = '../data/mimic-cxr-jpg/'
mimic_dir = '../data/mimic-iv/'
saved_dir = '../data/saved/'
remove = ['FINAL REPORT', 'EXAMINATION', 'TECHNIQUE', 'COMPARISON',
          'WET READ', 'EXAM']
cat_cols = ['admission_location', 'discharge_location', 'insurance', 'ethnicity',
            'gender', 'anchor_age', 'eventtype', 'careunit', 'icd_code',
            'icd_version', 'rhythm', 'pain', 'seq_num',
            'icd_code_diagnoses_icd', 'icd_version_diagnoses_icd', 'drg_type',
            'drg_code', 'drg_severity', 'drg_mortality', 'curr_service',
            'spec_itemid', 'spec_type_desc', 'test_itemid', 'test_name',
            'org_itemid', 'org_name', 'ab_itemid', 'dilution_value',
            'interpretation', 'medication', 'itemid', 'flag', 'priority',
            'medication_pharmacy', 'status', 'order_type', 'gsn','icd_code_procedures_icd',
            'icd_version_procedures_icd']
cont_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
sampling_csvs = ['diagnosis','diagnoses_icd','drgcodes','microbiologyevents','labevents',
                 'procedures_icd']

class MIMICDataset(Dataset):
    '''
    Dataset encapsulating all multi-modal data available for patients
    in the MIMIC_CXR-JPG dataset
    '''
    
    def __init__(self, config, split, image_size=(224,224),
                 records_csv='../data/saved/records_new.csv',
                 split_csv='../data/saved/new_splits.csv',
                 tab_preprocessor=None, prepared_data=False):
        
        super().__init__()
        
        self.cxr_dir = cxr_dir
        self.mimic_dir = mimic_dir
        
        assert split in ('train','val','validation','validate','test','all')
        self.split = split if split not in ['validation','val'] else 'validate'
        print(self.split)
        
        # Verified that splits csv has some ordering of images as records csv, 
        # so this is a safe way to filter to the split
        if self.split == 'all':
            self.records = pd.read_csv(records_csv)
        else:
            self.records = pd.read_csv(records_csv)[pd.read_csv(split_csv).split_new==self.split]
        self.our_subjects = self.records.subject_id.unique()
        
        # Read in and preprocess the tabular data
        if prepared_data:
            self.tabular_nonsample = pd.read_csv(f'{saved_dir}{self.split}_nonsample_newsplit.csv',
                                                 index_col='subject_id')
            with open(f'{saved_dir}{self.split}_sample_newsplit.pkl', 'rb') as f:
                self.tabular_sample = pickle.load(f)
        else:
            self.data = self.read_data(config.data_to_use)
            
            
        self.tab_preprocessor = config.tab_preprocessor
        
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
                #GrayscaleJitter([-0.15, 0.15], [0.9,1.1]),
                transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6,1.4)),
                #GaussianNoise(0., (0.1, 3.0)),
                transforms.GaussianBlur(kernel_size=ks, sigma=(0.1,3.0)),
                transforms.ToTensor()
            ])
                
        else:
            self.transforms = transforms.ToTensor()
        self.normalizer = transforms.Normalize(mean=[0.4860]*3, std=[0.2874]*3)
        
    def __len__(self):
        return self.records.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        subset = self.records.iloc[idx]
        subject = subset['subject_id']
        
        # Image handling
        image = Image.open(self.cxr_dir+subset['path_compressed'][:-3]+'jpg')
        image = self.transforms(image) 
        image = image.repeat(3, 1, 1)
        
        # Text handling
        text_path = self.cxr_dir + 'files-compressed/s' + str(subset['study_id']) + '.txt'
        if os.path.exists(text_path):
            with open(text_path) as f:
                text = f.read()
            text = text.strip()
            text = ''.join([t.replace('_','') for t in text.split('\n') if self.valid_text(t)])
        else:
            std = subset['study_id']
            print(f'No text for study {std}')
            print(text_path)
            text = np.array([''])
            
        # Tabular handling - need to preprocess
        tabular = self.tabular_getitem(subject)
        
        return {'image': image, 'text': text, 'tabular': tabular}
    
    def preprocess_image(self, image):
        # Uses weighted mean and std from MIMIC-CXR-JPG, Stanford, NIH Chest X-ray data:
        '''
        normalizer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4860], std=[0.2874])
        ])
        '''
        
        image = self.normalizer(image)
        # Downsampling to make all images the same size
        # Note that interpolate wants images in shape (batch_size, num_channels, *additional_dims)
        image = F.interpolate(image.unsqueeze(0), size=self.image_size).squeeze(0)
        return image
    
    def read_data(self, data_dict):
        # By the time this function finishes, every subject should have ONE row
        data = pd.DataFrame(index = self.records.subject_id.unique())
        for module in data_dict.keys(): # Iteration over modules
            for csv in data_dict[module].keys(): # Iteration over csvs within module
                agg_csv = partial(self.mimic_agg, csv=csv)
                df = pd.read_csv(f'{mimic_dir}{module}/{csv}.csv',
                                 index_col = 'subject_id',
                                 usecols = data_dict[module][csv])
                df = df[df.index.isin(self.our_subjects)] 
                # Aggregation should happen below these comments
                # One row per subject per table. Likely requires different logic per table...
                
                #df = df.groupby(df.index).apply(agg_csv)
                
                # For now just using most recent data for everyone
                df = df.groupby(df.index).tail(1)
                data = data.join(df, rsuffix=f'_{csv}')
        return data

    def get_features_for_tabular(self):
        '''
        Returns the columnx_idx and continuous_cols from the widedeep preprocessor
        for use with the module's TabTransformer
        '''
        col_idx = {k:v for v,k in enumerate(self.data.columns)}
        cont_cols = self.tab_preprocessor.continuous_cols
        embed_in = self.tab_preprocessor.embeddings_input
        return col_idx, cont_cols, embed_in
    
    def tabular_getitem(self, subject):
        '''
        Returns the tabular data, with random sampling among the saved
        rows from csvs marked for sampling
        '''
        try:
            series = self.tabular_nonsample.loc[subject]
        except:
            #print(f'subject not found in tabular_nonsample')
            series = {}
            for col in self.tabular_nonsample.columns:
                series[col] = np.nan if self.tabular_nonsample[col].dtype.kind in 'biufc' else 'NA'
            series = pd.Series(series)
        for csv in sampling_csvs:
            try:
                tmp = self.tabular_sample[csv][subject]
                series = series.append(tmp[np.random.randint(0,len(tmp))])
            except:
                #print(f'subject {subject} not found in tabular_sample for {csv} '+
                #      f'in split {self.split}')
                # Get first series of someone in the dict just for the col names
                t = self.tabular_sample[csv][next(iter(self.tabular_sample[csv].keys()))][0]
                for col in t.index:
                    series[col] = np.nan if col in cont_cols else 'NA'
            
        return series[cat_cols+cont_cols]
    
    def valid_text(self, text):
        return not any([r in text for r in remove]) and len(text) > 1
        
    
def collate_mimic(batch, tokenizer, tab_preprocessor):
    image = torch.stack([item['image'] for item in batch])
    text = [item['text'] for item in batch]
    tabular = [item['tabular'] for item in batch]
    
    # Alt: Turn batch from dict of lists to list of dicts. Time the difference
    # batch = {k: [d[k] for d in batch] for k in batch[0]}
    # image, text = batch['image'], batch['text']
    
    # Pad all text vectors so same length with <PAD> tokens
    # Tokenize in collate, not getitem, for proper padding
    try:
        encoding = tokenizer(text, return_tensors='pt',
                         padding=True, truncation=True)
    except:
        print(text, type(text))
        print(len(text)) 
        raise ValueError
    text = encoding['input_ids']
    attention_masks = encoding['attention_mask']
    
    # Turn tabular into a DataFrame for easier manipulation
    # May not be needed with widedeep
    tabular = pd.DataFrame(tabular)
    tabular[cont_cols] = tabular[cont_cols].fillna(0)
    tabular = tab_preprocessor.transform(tabular)
    ## Figure out the new nan number
    #nan_token = sum([i[1] for i in tab_preprocessor.embeddings_input])+1
    tabular = np.nan_to_num(tabular, nan=31506)
    tabular = torch.tensor(tabular)
    return image, text, attention_masks, tabular

class MIMICDataModule(pl.LightningDataModule):
    '''
    Parameters:
        split: bool or iterable. If True, creates train/test/val dataloaders;
            If False, creates a single dataloader for entire dataset;
            If iterable, creates only the dataloaders specified
        batch_size: Batch size
        num_workers: Number of workers
        tokenizer: The tokenizer to use for BERT
    '''
    def __init__(self, config):
        super().__init__()
        assert config.split in ['no_test', True, False], "config.split must be one of 'no_test', True, False"
        self.config = config
        self.split = config.split
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",
                                                       model_max_length=512)
        self.prepared_data = config.prepared_data
        
    def setup(self, stage=None):
        if self.split == 'no_test':
            self.train_data = MIMICDataset(config=self.config, split='train',
                                           prepared_data=self.prepared_data,
                                           tab_preprocessor=self.config.tab_preprocessor)
            self.tab_preprocessor = self.train_data.tab_preprocessor
            self.val_data = MIMICDataset(config=self.config, split='validate',
                                         tab_preprocessor=self.tab_preprocessor,
                                         prepared_data=self.prepared_data)
        
        elif not self.split:
            self.train_data = MIMICDataset(config=self.config, split='all',
                                           prepared_data=self.prepared_data,
                                           tab_preprocessor=self.config.tab_preprocessor)
            self.tab_preprocessor = self.train_data.tab_preprocesor
        
        else:
            self.train_data = MIMICDataset(config=self.config, split='train',
                                           prepared_data=self.prepared_data,
                                           tab_preprocessor=self.config.tab_preprocessor)
            self.val_data = MIMICDataset(config=self.config, split='validate',
                                         prepared_data=self.prepared_data,
                                         tab_preprocessor=self.config.tab_preprocessor)
            self.test_data = MIMICDataset(config=self.config, split='test',
                                          prepared_data=self.prepared_data,
                                          tab_preprocessor=self.config.tab_preprocessor)
        
        self.col_names = cat_cols+cont_cols
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True,
                          collate_fn=partial(collate_mimic, tokenizer=self.tokenizer,
                                             tab_preprocessor = self.tab_preprocessor))
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=False,
                          collate_fn=partial(collate_mimic, tokenizer=self.tokenizer,
                                             tab_preprocessor = self.tab_preprocessor))
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=False,
                          collate_fn=partial(collate_mimic, tokenizer=self.tokenizer,
                                             tab_preprocessor = self.tab_preprocessor))

    def get_features_for_tabular(self):
        '''
        Returns which features are categorical and continuous for use with a
        pytorch-widedeep model, as well as number of unique values for embed column
        '''
        # First term creates column name:number index from data's column names
        col_idx = {k:v for v,k in enumerate(self.col_names)}
        cont_cols = self.tab_preprocessor.continuous_cols
        embed_in = self.tab_preprocessor.embeddings_input
        return col_idx, cont_cols, embed_in
                
                