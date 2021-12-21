import model
import dataset_mortality
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

tv = 1 # Train version
cr = 4.0 # neg:pos class ratio for downsampling

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
#with open('/scratch/arz8448/capstone/data/saved/tab_preprocessor_newsplit.pkl','rb') as f:
#    tab_preprocessor = pickle.load(f)
    
col_idx, cont_cols, embed_in = ds.get_features_for_tabular()

ckpt = '/scratch/arz8448/capstone/outputs/train_v1/lightning_logs/version_13/checkpoints/epoch=112-step=651444.ckpt'
#ckpt = '/scratch/arz8448/capstone/outputs/train_v2/lightning_logs/version_19/checkpoints/last.ckpt'
if args.imagenet:
    module = model.MortalityModule(args=arg)
else:
    print('Using our weights')
    m = model.MIMICModule.load_from_checkpoint(ckpt, config=configs.config_train_1(),
                                               col_idx=col_idx, cont_cols=cont_cols,
                                               embed_in=embed_in, train_ver=tv)
    module = model.MortalityModule(ourmodel=m, args=args)
    
dm = dataset_mortality.MortalityDataModule(configs.config_mortality(), imagenet=args.imagenet,
                                           train_ver=tv, class_ratio=cr)
dm.setup()

if args.imagenet and not args.pretrained: # If using random init, train whole model together
    callbacks = [
        ModelCheckpoint(monitor='val_auc', mode='max', every_n_epochs=1, save_last=True),
        GPUStatsMonitor()
    ]
else: # Else, warm up the classification layer 
    print('Using finetune')
    callbacks = [
        ModelCheckpoint(monitor='val_auc', mode='max', every_n_epochs=1, save_last=True),
        GPUStatsMonitor(),
        ResNetFinetuning(unfreeze_at_epoch = 1)
    ]
    

strat = None if torch.cuda.device_count() < 2 else 'ddp'
drd = '/scratch/arz8448/capstone/outputs/mortality/'
if args.imagenet:
    drd += 'imagenet/'
    if args.pretrained:
        drd += 'pretrained/'
trainer = pl.Trainer(default_root_dir=drd,
                     gpus=torch.cuda.device_count(), accelerator='auto',
                     precision=16, callbacks=callbacks, max_epochs=80,
                     progress_bar_refresh_rate=50)
trainer.fit(module, datamodule=dm)
trainer.test(ckpt_path='best', verbose=True, model=module, datamodule=dm)
print(module.logger.log_dir)