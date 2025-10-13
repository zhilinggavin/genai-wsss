'''
The datasets are for diffusion autoencoder
'''

import os
from pathlib import Path
from os import listdir
from os.path import splitext, isfile, join
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as trans_fn, InterpolationMode

from PIL import Image
import numpy as np
import random
from tqdm import tqdm



import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# logging.info(f'Creating dataset with {len(self.ids)} examples')

class Dataset_diffae_osic(Dataset):
    def __init__(self, x: list[str], y: list[int]):
        self.names = x
        self.labels = y
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.root_dir = 'data/OSIC/processed'

    def __getitem__(self, index):
        _label = self.labels[index] 
        self.img_dir = join(self.root_dir, 'fibrosis' if _label == 1 else 'no_fibrosis')
        
        try:
            I = Image.open(join(self.img_dir, self.names[index]))
        except FileNotFoundError:
            self.img_dir = '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/no_fibrosis_covid/'
            I = Image.open(join(self.img_dir, self.names[index]))


        # from torchvision.transforms import InterpolationMode
        # I_resize = trans_fn.resize(I.convert("RGB"), 256, InterpolationMode.LANCZOS)
        # _img = self.transform(I_resize)
        _img : torch.Tensor = self.transform(I) #type: ignore
        return _img, _label, self.names[index]

    def __len__(self):
        return len(self.names)
    
if __name__ == "__main__":
    img_names = ["033_fibrosis_026.png", "033_fibrosis_031.png"]
    labels = [1, 1]
    dataset = Dataset_diffae_osic(img_names, labels)
    img,label,name = dataset[0]
    logging.info(f'Image: {name}, Label: {label}, Shape: {img.shape}')
    for i in tqdm(range(len(dataset))):
        img, label, name = dataset[i]
        logging.info(f'Image: {name}, Label: {label}, Shape: {img.shape}')

