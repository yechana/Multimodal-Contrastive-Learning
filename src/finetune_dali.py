import model
#import dataset_mortality
#import dataset_mortality_kornia as dataset_mortality
import dataset
import dali_mortality as dataset_mortality
from configs import config_mortality, config_train_1
import torch
from torch import nn
import torch.nn.functional as F
from importlib import reload
from torchvision.models import resnet50
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor, BaseFinetuning
import argparse
from sklearn.metrics import roc_auc_score
from importlib import reload
import warnings
reload(dataset_mortality)
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--imagenet", action='store_true',
                    help="If used, will not use self-supervised weights")
parser.add_argument("--pretrained", action='store_true',
                    help="If using --imagenet, whether to use imagenet pretrained weights or random init")
#args = parser.parse_args()

class MortalityModule(pl.LightningModule):
    def __init__(self, image_model):
        super().__init__()
        self.image_model = image_model
        self.image_project = nn.Linear(1000, 512) ## REPLACE WITH TRAINED PROJECTION
        self.classifier = nn.Linear(512, 1)
        self.criterion = nn.BCEWithLogitsLoss(torch.FloatTensor([7.]))
        
    def forward(self, x):
        x = self.image_project(self.image_model(x))
        return self.classifier(x)
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        out = self(x)
        loss = self.criterion(out, labels.unsqueeze(1).float())
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        out = self(x)
        loss = self.criterion(out, labels.unsqueeze(1).float())
        
        self.log('val_loss', loss)
        self.log('val_auc', roc_auc_score(labels.cpu(), F.sigmoid(out).cpu()))
        return loss
    
    def test_step(self, batch, batch_idx):
        x, labels = batch
        out = self(x)
        loss = self.criterion(out, labels.unsqueeze(1).float())
        self.log('test_loss', loss)
        
        preds = out > 0
        self.log('test_accuracy', sum(preds==labels))
        self.log('test_auc', roc_auc_score(labels.cpu(), F.sigmoid(out).cpu()))
        
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=1e-3)
        return optimizer
    
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
ckpt = '/scratch/arz8448/capstone/outputs/train_v1/lightning_logs/version_2/checkpoints/last.ckpt'

# Tab preprocessor saved for newsplit has wrong # classes, need the one
# created by train_v1 dataset class
ds = dataset.MIMICDataModule(config_train_1())
ds.setup()
#with open('/scratch/arz8448/capstone/data/saved/tab_preprocessor_newsplit.pkl','rb') as f:
#    tab_preprocessor = pickle.load(f)
    
col_idx, cont_cols, embed_in = ds.get_features_for_tabular()

#if args.imagenet:
#    module = MortalityModule(resnet50(pretrained=args.pretrained))
#else:
m = model.MIMICModule.load_from_checkpoint(ckpt, config=config_train_1(), col_idx=col_idx,
                                           cont_cols=cont_cols, embed_in=embed_in)
module = MortalityModule(m.model.image1)
dm = dataset_mortality.MortalityDataModule(config_mortality(),
                                           device='gpu')
dm.setup()

callbacks = [
    ModelCheckpoint(monitor='val_auc', every_n_epochs=1, save_last=True),
    GPUStatsMonitor(),
    ResNetFinetuning()
]

print(torch.backends.cudnn.enabled, torch.backends.cudnn.version(), torch.cuda.device_count())
strat = None if torch.cuda.device_count() < 2 else 'ddp'
trainer = pl.Trainer(default_root_dir='/scratch/arz8448/capstone/outputs/mortality/',
                     gpus=1, accelerator='gpu',
                     precision=16, callbacks=callbacks, max_epochs=50,
                     progress_bar_refresh_rate=1)
trainer.fit(module, datamodule=dm)
