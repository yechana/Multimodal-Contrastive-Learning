import model
import dataset_mortality
#import dataset_mortality_kornia as dataset_mortality
import dataset
#import dataset_3ch_textsample_tabsample as dataset
import configs
import torch
from torch import nn
from importlib import reload
from torchvision.models import resnet50
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor, BaseFinetuning
import argparse
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from collections import OrderedDict

tv = 1 # Train version
cr = 4.0 # neg:pos class ratio

parser = argparse.ArgumentParser()
parser.add_argument("--imagenet", action='store_true',
                    help="If used, will not use self-supervised weights")
parser.add_argument("--pretrained", action='store_true',
                    help="If using --imagenet, whether to use imagenet pretrained weights or random init")
args = parser.parse_args()
    
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

# Tab preprocessor saved for newsplit has # classes for train_v2, not train_v1
ds = dataset.MIMICDataModule(configs.config_train_1())
ds.setup()
    
col_idx, cont_cols, embed_in = ds.get_features_for_tabular()

ckpt = '/scratch/arz8448/capstone/outputs/mortality/lightning_logs/version_19/checkpoints/epoch=2-step=2756.ckpt'
drd = '/scratch/arz8448/capstone/outputs/mortality/testing/'
if args.imagenet:
    drd += 'imagenet/'
    if args.pretrained:
        drd += 'pretrained/'
        
class BlankModule(): # So we can pass ourmodel properly 
    def __init__(self):
        self.model = nn.Sequential(OrderedDict([
            ('image1', resnet50()),
            ('image_project', nn.Linear(1000,512))
        ]))
        if not args.imagenet:
            self.model.image1.conv1 = nn.Conv2d(1,64,kernel_size=7, stride=2, padding=3, bias=False)
        
module = model.MortalityModule.load_from_checkpoint(ckpt, ourmodel=BlankModule(), args=args,
                                                    drd=drd)

dm = dataset_mortality.MortalityDataModule(configs.config_mortality(), imagenet=args.imagenet,
                                           train_ver=tv, class_ratio=cr)
dm.setup()    

strat = None if torch.cuda.device_count() < 2 else 'ddp'
trainer = pl.Trainer(default_root_dir=drd,
                     gpus=torch.cuda.device_count(), accelerator='auto',
                     precision=16, max_epochs=80,
                     progress_bar_refresh_rate=50)
trainer.test(model=module, datamodule=dm, verbose=True)
