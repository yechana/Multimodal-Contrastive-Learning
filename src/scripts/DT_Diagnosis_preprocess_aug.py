from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pandas as pd
import os
chexpert_path_em = '/scratch/em4449/stanford_data/CheXpert-v1.0/'
chexpert_path_aug = '/scratch/yl7971/capstone/CheXpert-Aug/'

img_size = 224
ks = int(0.1*img_size)
ks = ks - ((ks+1)%2)

train_paths = pd.read_csv(chexpert_path_em + 'train.csv', usecols = ['Path']).Path
val_paths = pd.read_csv(chexpert_path_em + 'valid.csv', usecols = ['Path']).Path
all_paths = train_paths.append(val_paths)
total = len(all_paths)
for i,p in enumerate(all_paths):
    path = p.split('/',1)[1]
    img = Image.open(chexpert_path_em + path)
    augment = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5032, std=0.2919), #Stanford mean and std
        transforms.Pad([0,0,max(0,max(img.size)-img.size[0]),max(0,max(img.size)-img.size[1])]),
        transforms.Resize(224,InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine((-20, 20),
                                translate=(0.1, 0.1),
                                scale=(0.95, 1.05)),
        #transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6,1.4)),
        transforms.GaussianBlur(kernel_size=ks, sigma=(0.1,3.0)),
        transforms.ToPILImage()
    ])
    if not os.path.exists(chexpert_path_aug+path[:path.rfind('/')+1]):
        os.makedirs(chexpert_path_aug+path[:path.rfind('/')+1])
    augment(img).convert('RGB').save(chexpert_path_aug + path)
    if not i%10000:
        print(f'{(i+1)/total*100}% done!')