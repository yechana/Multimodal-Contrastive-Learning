import dataset_3ch_textsample_tabsample as dataset
import model_frozeBERT_3channels as model
from configs import config_train_2
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cont_from', type=str,
                    help=('Directory of run (not the checkpoint! from which we want to continue '
                          'training, '+
                          'specify as absolute directory, or relative directory ' +
                          'from this script. Should end with /'),
                    default=None)
args = parser.parse_args()

cfg = config_train_2()
if args.cont_from:
    ckpt = args.cont_from + 'checkpoints/last.ckpt'
else:
    ckpt = None

callbacks = [
    ModelCheckpoint(monitor='val_total_loss', every_n_epochs=1, save_last=True),
    GPUStatsMonitor()
]

strat = None if torch.cuda.device_count() < 2 else 'ddp'
trainer = pl.Trainer(default_root_dir='/scratch/arz8448/capstone/outputs/train_v2/',
                     gpus=torch.cuda.device_count(),
                     accelerator='auto', strategy=strat, precision=16, callbacks=callbacks,
                     check_val_every_n_epoch=1, max_epochs=100,
                     progress_bar_refresh_rate=50, resume_from_checkpoint=ckpt)

data = dataset.MIMICDataModule(cfg)
data.setup()

col_idx, cont_cols, embed_in = data.get_features_for_tabular()
model = model.MIMICModule(cfg, col_idx, cont_cols, embed_in)

trainer.fit(model, datamodule=data)
print('End training')