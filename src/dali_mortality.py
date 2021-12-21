from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy, DALIGenericIterator
import pytorch_lightning as pl
import torch

cxr_dir = '../data/mimic-cxr-jpg/'

@pipeline_def
def mortality_pipeline(split, device):
    jpegs, labels = fn.readers.file(file_list=f'../data/saved/mortality_labels_{split}.txt',
                                    file_root=cxr_dir, random_shuffle=True,
                                    name='Reader')
    images = fn.decoders.image(jpegs, 
                               device='mixed' if device == 'gpu' else 'cpu',
                               output_type=types.GRAY)
    if device == 'gpu':
        labels = labels.gpu()
    labels = fn.cast(labels, dtype=types.INT64)
    images = fn.resize(images, size=(224,224)) / 225.
    images = fn.crop_mirror_normalize(
        images,
        mean=0.4860,
        std=0.2874
    )
    return images, labels

# This takes the place of a pytorch DataLoader
class LightningWrapper(DALIClassificationIterator):
    def __init__(self, *kargs, **kvargs):
        super().__init__(*kargs, **kvargs)
        
    def __next__(self):
        out = super().__next__()
        # DDP is used so only one pipeline per process
        # also we need to transform dict returned by DALIClassificationIterator to iterable
        # and squeeze the lables
        out = out[0]
        return [out[k] if k != "label" else torch.squeeze(out[k]) for k in self.output_map]

class MortalityDataModule(pl.LightningDataModule):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.device = device
        
    def setup(self, stage=None):
        self.train_pipe = mortality_pipeline(split='train', device=self.device,
                                        batch_size=self.config.batch_size,
                                        num_threads=self.config.num_workers, device_id=0)
        self.val_pipe = mortality_pipeline(split='validate', device=self.device,
                                      batch_size=self.config.batch_size,
                                      num_threads=self.config.num_workers, device_id=0)
        self.test_pipe = mortality_pipeline(split='test', device=self.device,
                                       batch_size=self.config.batch_size,
                                       num_threads=self.config.num_workers, device_id=0)
    
    def train_dataloader(self):
        return LightningWrapper(self.train_pipe, auto_reset=True,
                              last_batch_policy=LastBatchPolicy.PARTIAL,
                              reader_name='Reader')
         
    def val_dataloader(self):
        return LightningWrapper(self.val_pipe, auto_reset=True,
                                last_batch_policy=LastBatchPolicy.PARTIAL,
                                reader_name='Reader')
    
    def test_dataloader(self):
        return LightningWrapper(self.test_pipe, auto_reset=True,
                                last_batch_policy=LastBatchPolicy.PARTIAL,
                                reader_name='Reader')