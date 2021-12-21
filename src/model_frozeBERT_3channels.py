import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from losses import NTXentLoss
from torchvision.models import resnet50
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_widedeep.models import TabFastFormer, TabTransformer
import numpy as np

dont_freeze = [f'encoder.layer.{i}' for i in range(6,12)]+['pooler']

class MIMICModel(nn.Module):
    
    def __init__(self, config, col_idx, cont_cols, embed_input):
        super().__init__()
        
        self.batch_size = config.batch_size
        
        self.image1 = resnet50()
        self.image_project = nn.Linear(1000, 512)
        self.text1 = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        for name, param in self.text1.named_parameters():
            if all([s not in name for s in dont_freeze]):
                param.requires_grad = False
        self.text_project = nn.Linear(768, 512)
        self.tab1 = TabTransformer(column_idx=col_idx,
                                  continuous_cols=cont_cols,
                                  embed_input=embed_input)
        self.tab1.cat_and_cont_embed.cat_embed.embed.num_embeddings += 1 # For added nan value
        #print(self.tab1.cat_and_cont_embed.cat_embed.embed.num_embeddings)
        # 2380 with 43 cols
        # 2508 for transformer
        self.tab_project = nn.Linear(2508,512)
        
    def forward(self, image, text, masks, tabular):
        x_image = self.image_project(self.image1(image))
        
        x_text = self.text1(text, masks).last_hidden_state
        x_text = torch.max(x_text, axis=1).values
        x_text = self.text_project(x_text)
        
        x_tab = self.tab1(tabular)
        x_tab = self.tab_project(x_tab)
        return (x_image, x_text, x_tab)
    
class MIMICModule(pl.LightningModule):
    
    def __init__(self, config, col_idx, cont_cols, embed_in):
        super().__init__()
        self.model = MIMICModel(config, col_idx, cont_cols, embed_in)
        
        self.criterion_image_EHR = NTXentLoss(device='cuda', batch_size=config.batch_size,
                                              temperature=0.1, use_cosine_similarity=True,
                                              alpha_weight=0.75)
        self.criterion_image_text = NTXentLoss(device='cuda', batch_size=config.batch_size,
                                               temperature=0.7, use_cosine_similarity=True,
                                               alpha_weight=0.75)
        
    def forward(self, image, text, masks, tabular):
        return self.model(image, text, masks, tabular)
    
    def training_step(self, batch, batch_idx):

        image, text, masks, tabular = batch
        image_out, text_out, tabular_out = self(image, text, masks, tabular)
        loss1 = self.criterion_image_EHR(image_out, tabular_out)
        loss2 = self.criterion_image_text(image_out, text_out)
        loss = loss1 + loss2
        
        self.log('train_loss_imgtab', loss1)
        self.log('train_loss_imgtxt', loss2)
        self.log('train_total_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):

        image, text, masks, tabular = batch
        image_out, text_out, tabular_out = self(image, text, masks, tabular)
        loss1 = self.criterion_image_EHR(image_out, tabular_out)
        loss2 = self.criterion_image_text(image_out, text_out)
        loss = loss1 + loss2
        
        self.log('val_loss_imgtab', loss1)
        self.log('val_loss_imgtxt', loss2)
        self.log('val_total_loss', loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_total_loss',
            'strict': True
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}