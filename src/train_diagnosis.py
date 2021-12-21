from torchvision.models import resnet50
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pandas as pd
from PIL import Image, ImageOps
from numbers import Number
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Optional
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor, EarlyStopping, BaseFinetuning
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import AUROC
from sklearn.model_selection import train_test_split
import os.path
import json
import sys
import ast
from model import MIMICModule

import configs
import dataset
from sklearn.metrics import roc_auc_score, recall_score, precision_score


if len(sys.argv) >= 3: # TRAIN_FRAC = 1.0 FOR THIS FILE
    image_net = ast.literal_eval(sys.argv[1]) #True/Fasle, is this for starting with imagenet weights (True) or our own (False)?
    train_frac = float(sys.argv[2]) #How much of the training data should we use? (I removed sampling functionality so always 1.0)
    #theirs = ast.literal_eval(sys.argv[3])
    theirs = False
else:
    sys.exit("You must provide arguments image_net (True/False) and train_frac (float -- 0.1/0.5/1.0).")

chexpert_path_em = '/scratch/em4449/stanford_data/CheXpert-v1.0/'
chexpert_path_yl = '/scratch/yl7971/capstone/CheXpert-v1.0/'

### GET POS_WEIGHTS ### --- For weighted loss function
pos_weights = {}
for file in [chexpert_path_yl + 'seth_' + split + '.csv' for split in ['train','valid','test']]:
    csv = pd.read_csv(file, index_col=['Path'])
    pos_weight = (csv.count() - csv.sum()) / csv.sum() # neg count / pos count
    pos_weights[file.split('_')[1].split('.')[0]] = torch.Tensor(pos_weight) # these will be very similar across our 3 balanced splits
    
# Equivalent to Dataset.py
class CheXpertDataset(Dataset):
    '''
    Trimmed Dataset for diagnosis classification task
    '''    
    def __init__(self, split, sample=1.0, rgb=False, image_size=(224,224), data_dir=chexpert_path_yl): #sample is not used right now        
        super().__init__()       
        self.data_dir = data_dir
        assert split in ('train','val','validation','validate','test')
        self.split = split if split not in ['validation','val','validate'] else 'valid'
        # Use augmented images for the training split
        if self.split == 'train':
            #self.data_dir = '/scratch/yl7971/capstone/CheXpert-Aug/'
            self.data_dir = '/scratch/yl7971/capstone/CheXpert-v1.0/'
        self.data_csv = pd.read_csv(chexpert_path_yl + 'seth_'+ self.split + '.csv',
                                        index_col = ['Path'])

        self.image_size = image_size
        self.rgb = rgb # True = model input images need to be RGB; False = Grayscale
        
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
        
    def __len__(self):
        return self.data_csv.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        subset = self.data_csv.iloc[idx]
        path = self.data_csv.index[idx].split('/',1)[1] #Path to image
        image = Image.open(self.data_dir+path)     
        if not self.rgb: #The pre-processed images are already RGB, just grayscale them for our modified resnet
            image = image.convert('L')
        # Image handling
        image = self.transforms(image)
        image = transforms.ToTensor()(image)    
        # Diagnosis labels
        label = torch.tensor(subset.astype(float)) #e.g. [1.0, 0.0, 0.0, 0.0, 1.0]

        return {'image': image, 'label': label}
    
def collate_cheXpert(batch):
    image = torch.stack([item['image'] for item in batch])
    label = torch.stack([item['label'] for item in batch])

    return image, label
    
    
class CheXpertDataModule(pl.LightningDataModule):

    def __init__(self,batch_size,num_workers,train_frac=1.0,rgb=False):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_frac = train_frac #gets passed to 'sample' in dataset
        self.rgb = rgb
    def setup(self, stage:Optional[str]=None) -> None:

        self.train_data = CheXpertDataset(sample=self.train_frac, rgb=self.rgb, split='train')
        self.val_data = CheXpertDataset(rgb=self.rgb, split='validate')
        self.test_data = CheXpertDataset(rgb=self.rgb, split='test')
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True,
                          collate_fn=collate_cheXpert, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True,
                          collate_fn=collate_cheXpert, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=True,
                          collate_fn=collate_cheXpert, pin_memory=True)


tune_all = True #True = fine-tune all weights. Flase = only tune classifier weights. Almost always been using True.
batch_size = 64
num_workers = 8 #match CPU count and increase RAM if increase
num_classes=5

# Freeze ResNet and tune classification head only for 200 steps
if image_net:
    RN50 = nn.Sequential(
        resnet50(pretrained=True),
        nn.Linear(2048,512),
        nn.Linear(512, num_classes)
    )
elif theirs:
    RN50 = nn.Sequential(
        resnet50(pretrained=False),
        nn.Linear(2048,512),
        nn.Linear(512, num_classes)
    )
    with open('../Seth/convirt_chest_mimic.pt','rb') as f:
        weights = torch.load(f)
else:
    # Initialize our module
    ds = dataset.MIMICDataModule(configs.config_train_1())
    ds.setup()
    col_idx, cont_cols, embed_in = ds.get_features_for_tabular()
    ckpt = '/scratch/arz8448/capstone/outputs/train_v1/lightning_logs/version_13_backup/checkpoints/epoch=112-step=651444.ckpt'
    m = MIMICModule.load_from_checkpoint(ckpt, config=configs.config_train_2(),
                                           col_idx=col_idx, cont_cols=cont_cols,
                                           embed_in=embed_in, train_ver=1)
    RN50 = nn.Sequential(
        m.model.image1,
        m.model.image_project,
        nn.Linear(512, num_classes)
    )

class DiagnosisModule_2(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.image_model = model[0]
        self.image_project = model[1]
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = model[2]
        
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion_train_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weights['train'])
        self.criterion_val_weighted = nn.BCEWithLogitsLoss()
        self.criterion_test_weighted = nn.BCEWithLogitsLoss()
        self.auroc_train = AUROC(num_classes=num_classes, compute_on_step=False) #Computes every epoch
        self.auroc_val = AUROC(num_classes=num_classes, compute_on_step=False)
        self.auroc_test = AUROC(num_classes=num_classes, compute_on_step=False)
        
    def training_step(self, batch, batch_idx):

        x, y = batch
        x = self.image_project(self.image_model(x))
        x = self.dropout(x)
        y_hat = self.classifier(x)
        loss = self.criterion_train_weighted(y_hat, y)
        with torch.no_grad():
            loss2 = self.criterion(y_hat, y)
            self.auroc_train.update(torch.sigmoid(y_hat),y.long())
        self.log('train_loss', loss)
        self.log('train_loss_unweighted',loss2)
        self.log('train_auc', self.auroc_train, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        #Runs with no grad
        x, y = batch
        x = self.image_project(self.image_model(x))
        x = self.dropout(x)
        y_hat = self.classifier(x)
        loss = self.criterion_val_weighted(y_hat, y)
        loss2 = self.criterion(y_hat, y)
        probas = torch.sigmoid(y_hat)            
        self.auroc_val.update(probas, y.long())
        self.log('val_loss', loss)
        self.log('val_loss_unweighted',loss2)
        self.log('val_auc', self.auroc_val, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        #Runs with no grad
        x, y = batch
        x = self.image_project(self.image_model(x))
        x = self.dropout(x)
        y_hat = self.classifier(x)
        loss = self.criterion_test_weighted(y_hat, y)
        loss2 = self.criterion(y_hat, y)
        probas = torch.sigmoid(y_hat) 
        preds = (probas > 0).cpu()
        cpu_labels = y.cpu()
        
        self.auroc_test.update(probas, y.long())
        self.log('test_loss', loss)
        self.log('test_loss_unweighted',loss2)
        self.log('test_auc', self.auroc_test, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_recall', recall_score(cpu_labels, preds, average='macro'))
        self.log('test_precision', precision_score(cpu_labels, preds, average='macro'))

        return {'test_loss': loss,'test_loss_unweighted': loss2, 'test_auc': self.auroc_test}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6) # Per Manning
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3) # Per Manning
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}


class ResNetFinetuning(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=1):
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        
    def freeze_before_training(self, pl_module):
        self.freeze([pl_module.image_model, pl_module.image_project])
        
    def finetune_function(self, pl_module, epoch, optimizer, optimizer_idx):
        if epoch == self.unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules = [pl_module.image_model, pl_module.image_project],
                optimizer = optimizer,
                train_bn = True
            )
            
mc_all = ModelCheckpoint(monitor='val_auc', mode='max', every_n_epochs=1, save_last=True)
callbacks = [
    mc_all,
    GPUStatsMonitor(),
    EarlyStopping(monitor='val_auc', patience=18, mode='max'),
    ResNetFinetuning(unfreeze_at_epoch = 1)
] 
ckpt = '../outputs/diagnosis/lightning_logs/version_1/checkpoints/last.ckpt'
strat = None if torch.cuda.device_count() < 2 else 'ddp'
#drd = '/scratch/yl7971/capstone/DT/outputs/'
drd = '/scratch/arz8448/capstone/outputs/diagnosis/'
trainer = pl.Trainer(default_root_dir=drd,
                     gpus=torch.cuda.device_count(),
                     accelerator='auto', strategy=strat, precision=16, callbacks=callbacks,
                     check_val_every_n_epoch=1, max_epochs=100,
                     progress_bar_refresh_rate=50,
                     resume_from_checkpoint=None)
data = CheXpertDataModule(batch_size,num_workers,train_frac,image_net)
#print('SETUP DATA')
model = DiagnosisModule_2(RN50)
#print('SETUP MODEL, BEGIN TRAININ')
trainer.fit(model, datamodule=data)
print('Finished tuning all hparams.')
test_res = trainer.test(ckpt_path="best")
#test_res = trainer.test(model)

if image_net:
    txt_file = 'Diag_CheXpert_imagenet_' + str(train_frac) + '_bal5_ds2_weighted.txt'
else:
    txt_file = 'Diag_CheXpert_ourweights_' + str(train_frac) + '_bal5_ds2_weighted.txt'
with open(txt_file, 'w') as file:
     file.write(json.dumps(test_res[0]))
print(test_res[0])