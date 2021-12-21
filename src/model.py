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
from sklearn.metrics import roc_auc_score, recall_score, precision_score, precision_recall_curve
import os
from torchmetrics import AUROC, PrecisionRecallCurve

dont_freeze = [f'encoder.layer.{i}' for i in range(6,12)]+['pooler']

class MIMICModel(nn.Module):
    
    def __init__(self, config, col_idx, cont_cols, embed_input, train_ver=1):
        super().__init__()
        
        self.batch_size = config.batch_size
        
        self.image1 = resnet50()
        if train_ver==1:
            self.image1.conv1 = nn.Conv2d(1,64,kernel_size=7, stride=2, padding=3, bias=False)
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
    
    def __init__(self, config, col_idx, cont_cols, embed_in, train_ver=1):
        super().__init__()
        self.model = MIMICModel(config, col_idx, cont_cols, embed_in, train_ver)
        
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
        
        self.log('train_loss1', loss1.item())
        self.log('train_loss2', loss2.item())
        self.log('train_total_loss', loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):

        image, text, masks, tabular = batch
        image_out, text_out, tabular_out = self(image, text, masks, tabular)
        loss1 = self.criterion_image_EHR(image_out, tabular_out)
        loss2 = self.criterion_image_text(image_out, text_out)
        loss = loss1 + loss2
        
        self.log('val_loss1', loss1.item())
        self.log('val_loss2', loss2.item())
        self.log('val_total_loss', loss.item())
        return loss
    
    def test_step(self, batch, batch_idx):

        image, text, tabular = batch
        features = self(image, text, tabular)
        loss1 = self.criterion_image_EHR(image, tabular)
        loss2 = self.criterion_image_text(image, text)
        self.log('test_loss1', loss1.item())
        self.log('test_loss2', loss2.item())
        return loss
    
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_total_loss',
            'strict': True
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
    
class MortalityModule(pl.LightningModule):
    def __init__(self, ourmodel=None, args=None, drd=None):
        super().__init__()
        if not args.imagenet:
            print('Using our model')
            self.image_model = ourmodel.model.image1
            self.image_project = ourmodel.model.image_project
        else:
            print(f'Using stock resnet. Pretrained is {args.pretrained}')
            self.image_model = resnet50(pretrained=args.pretrained)
            self.image_project = nn.Linear(1000, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(512, 1)
        self.criterion_train = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([4.]))
        self.criterion = nn.BCEWithLogitsLoss()
        self.auc_val = AUROC(compute_on_step=False)
        self.auc_test = AUROC(compute_on_step=False)
        self.pr_curve = PrecisionRecallCurve()
        
    def forward(self, x):
        x = self.image_project(self.image_model(x))
        x = self.dropout(x)
        return self.classifier(x)
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        out = self(x)
        loss = self.criterion_train(out, labels.unsqueeze(1).float())
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        out = self(x)
        loss = self.criterion(out, labels.unsqueeze(1).float())
        cpu_labels = labels.cpu()
        #preds = torch.argmax(out, dim=1).cpu()
        
        self.log('val_loss', loss)
        self.auc_val.update(torch.sigmoid(out), labels.long())
        self.log('val_auc', self.auc_val, on_step=False, on_epoch=True, prog_bar=True)
        #self.log('val_recall', recall_score(cpu_labels, preds))
        #self.log('val_precision', precision_score(cpu_labels, preds))
        return loss
    
    def test_step(self, batch, batch_idx):
        x, labels = batch
        out = self(x)
        loss = self.criterion(out, labels.unsqueeze(1).float())
        cpu_labels = labels.cpu()
        preds = (out > 0).cpu()
        #print(preds.float().mean(), labels.float().mean())
        
        self.log('test_loss', loss)
        self.log('test_accuracy', torch.mean((preds==cpu_labels).float()))
        self.auc_test.update(torch.sigmoid(out), labels.long())
        self.log('test_auc', self.auc_test, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_recall', recall_score(cpu_labels, preds))
        self.log('test_precision', precision_score(cpu_labels, preds))
        
        prec, rec, thresh = self.pr_curve(torch.sigmoid(out), labels)
        
        return loss
        
    def on_test_end(self):
        prec, rec, thresh = self.pr_curve.compute()
        if not os.path.isdir(self.logger.log_dir+'/numpy/'):
            os.mkdir(self.logger.log_dir+'/numpy/')
        with open(self.logger.log_dir+'/numpy/prec.pt', 'w') as f:
            torch.save(prec, f)
        with open(self.logger.log_dir+'/numpy/rec.pt', 'w') as f:
            torch.save(rec, f)
        with open(self.logger.log_dir+'/numpy/thresh.pt', 'w') as f:
            torch.save(thresh, f)
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=1e-4)
        return optimizer