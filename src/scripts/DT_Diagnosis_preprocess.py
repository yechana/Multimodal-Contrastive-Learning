from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pandas as pd

# Do all preprocessing for images and save.
# Images are convirted to RGB (for use with imagenet weights).

chexpert_path_em = '/scratch/em4449/stanford_data/CheXpert-v1.0/'
chexpert_path_yl = '/scratch/yl7971/capstone/CheXpert-v1.0/'

train_paths = pd.read_csv(chexpert_path_em + 'train.csv', usecols = ['Path']).Path
val_paths = pd.read_csv(chexpert_path_em + 'valid.csv', usecols = ['Path']).Path
all_paths = train_paths.append(val_paths)
total = len(all_paths)
for i,p in enumerate(all_paths):
    path = p.split('/',1)[1]
    img = Image.open(chexpert_path_em + path)
    pre_process = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5032, std=0.2919), #Stanford mean and std
            transforms.Pad([0,0,max(0,max(img.size)-img.size[0]),max(0,max(img.size)-img.size[1])]),
            transforms.Resize(224,InterpolationMode.NEAREST),
            transforms.ToPILImage()
        ])
    pre_process(img).convert('RGB').save(chexpert_path_yl + path)
    if not i%10000:
        print(f'{(i+1)/total*100}% done!')
