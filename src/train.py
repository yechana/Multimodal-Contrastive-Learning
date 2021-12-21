import dataset
from configs import config_train_1
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor
from model import MIMICModule
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--cont_from', type=str,
                    help=('Directory of run from which we want to continue training, '+
                          'specify as absolute directory, or relative directory ' +
                          'from this script. Should end with /'),
                    default=None)
args = parser.parse_args()
if args.cont_from:
    ckpt = args.cont_from + 'checkpoints/last.ckpt'
else:
    ckpt = None
        
callbacks = [
    ModelCheckpoint(monitor='val_total_loss', every_n_epochs=1, save_last=True),
    GPUStatsMonitor()
]

cfg = config_train_1()
strat = None if torch.cuda.device_count() < 2 else 'ddp'
#print(torch.cuda.device_count(), torch.cuda.is_available(), torch.version.cuda, torch.backends.cudnn.enabled, torch.backends.cudnn.version())
#print(torch.cuda.device_count())
#print(start, torch.cuda.device_count(), torch.cuda.is_available(), torch.version.cuda)
trainer = pl.Trainer(default_root_dir='/scratch/arz8448/capstone/outputs/train_v1/',
                     gpus=torch.cuda.device_count(),
                     accelerator='auto', strategy=strat, precision=16, callbacks=callbacks,
                     check_val_every_n_epoch=1, max_epochs=200,
                     progress_bar_refresh_rate=50, resume_from_checkpoint=ckpt)
data = dataset.MIMICDataModule(cfg)
data.setup()
col_idx, cont_cols, embed_in = data.get_features_for_tabular()
model = MIMICModule(cfg, col_idx, cont_cols, embed_in)
trainer.fit(model, datamodule=data)
print('End training')