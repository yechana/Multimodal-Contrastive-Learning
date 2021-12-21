import model
import dataset_mortality
import dataset
from configs import config_mortality, config_train_1
import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, recall_score, precision_score

class Args(object):
    def __init__(self, imagenet=True, pretrained=True):
        self.imagenet = imagenet
        self.pretrained = pretrained
# For ours, imgnet, ran, use version 1, 0, 0 respectively
ckpt_ours = '../outputs/mortality/lightning_logs/version_1/checkpoints/epoch=125-step=168083.ckpt'
ckpt_imgnet = ('../outputs/mortality/imagenet/pretrained/lightning_logs/'+
               'version_0/checkpoints/epoch=154-step=206769.ckpt')
ckpt_imgnet_rand = ('../outputs/mortality/imagenet/lightning_logs/'+
                    'version_0/checkpoints/epoch=164-step=220109.ckpt')

ds = dataset.MIMICDataModule(config_train_1())
ds.setup()
col_idx, cont_cols, embed_in = ds.get_features_for_tabular()

mod = model.MIMICModule(
    config=config_train_1(), col_idx=col_idx, cont_cols=cont_cols, embed_in=embed_in
)
module_ours = model.MortalityModule.load_from_checkpoint(ckpt_ours, ourmodel=mod,
                                                         args=Args(imagenet=False))
module_imgnet = model.MortalityModule.load_from_checkpoint(ckpt_imgnet, args=Args())
module_imgnet_rand = model.MortalityModule.load_from_checkpoint(ckpt_imgnet_rand,
                                                                args=Args(pretrained=False)
                                                               )
dm_ours = dataset_mortality.MortalityDataModule(config_mortality())
dm_ours.setup()
dm_imgnet = dataset_mortality.MortalityDataModule(config_mortality(), imagenet=True)
dm_imgnet.setup()

strat = None if torch.cuda.device_count() < 2 else 'ddp'
drd = '/scratch/arz8448/capstone/outputs/testing_mortality/'

trainer = pl.Trainer(default_root_dir=drd,
                     gpus=torch.cuda.device_count(), accelerator='gpu',
                     precision=16, max_epochs=200,
                     progress_bar_refresh_rate=50)
trainer.test(model=module_ours, datamodule=dm_ours, verbose=True)

trainer = pl.Trainer(default_root_dir=drd+'imagenet/pretrained/',
                     gpus=torch.cuda.device_count(), accelerator='gpu',
                     precision=16, max_epochs=200,
                     progress_bar_refresh_rate=50)
trainer.test(model=module_imgnet, datamodule=dm_imgnet, verbose=True)

trainer = pl.Trainer(default_root_dir=drd+'imagenet/',
                     gpus=torch.cuda.device_count(), accelerator='gpu',
                     precision=16, max_epochs=200,
                     progress_bar_refresh_rate=50)
trainer.test(model=module_imgnet_rand, datamodule=dm_imgnet, verbose=True)