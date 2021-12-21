import pandas as pd
from PIL import Image, ImageOps
from torchvision import transforms 
import torch

def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

normalizer = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=0.4860, std=0.2874),
    transforms.ToPILImage() 
])

records_new = pd.read_csv('/scratch/arz8448/capstone/data/saved/records_new.csv')
cxr_dir = '/scratch/arz8448/capstone/data/mimic-cxr-jpg/'

for i, p in enumerate(records_new.path_new):
    img = Image.open(cxr_dir+p)
    img = resize_with_padding(img, (224, 224))
    img = normalizer(img)
    name = p.split('/')[-1]
    img.save(cxr_dir+'files-compressed/'+name)
    if i%100 == 0:
        print(i)