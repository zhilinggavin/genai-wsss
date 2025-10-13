import os
import sys
sys.path.append("")
from src.diffae.useage import model_load
from utils.datasets import Dataset_diffae_osic
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
logging.basicConfig(level=logging.INFO)

'''
    Load dataset
'''
IMG_ROOT = 'data/OSIC/processed'
BATCH_SIZE = 10
fibrosis_dir = os.path.join(IMG_ROOT, 'fibrosis')
no_fibrosis_dir = os.path.join(IMG_ROOT, 'no_fibrosis')

def get_label_map(dir1: str, dir2: str) -> dict[str, int]:
    '''
    Get image filename to label mapping.
    Label 1: fibrosis
    Label 0: no_fibrosis
    '''
    img_label_map = {}
    
    imgfib_names = sorted(f for f in os.listdir(dir1) if f.endswith('.png'))
    imgnofib_names = sorted(f for f in os.listdir(dir2) if f.endswith('.png'))
    img_label_map = {name: 1 for name in imgfib_names}
    img_label_map.update({name: 0 for name in imgnofib_names})
    return img_label_map


img_label_map = get_label_map(fibrosis_dir, no_fibrosis_dir)

filenames = list(img_label_map.keys())
labels = [img_label_map[name] for name in filenames]
dataset = Dataset_diffae_osic(filenames, labels)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
logging.info("Dataset loaded successfully.")

img, label, name = dataset[0]
logging.info(f'Image: {name}, Label: {label}, Shape: {img.shape}')

'''
    Load ISBI diff model and classification model
'''
device = 'cuda'
model_diff, model_cls = model_load(device, diff=True, cls=True)
logging.info("ISBI Diffae and Classification Models loaded successfully.")

'''
    Encode image to latent space
'''
save_encode = True
img_cond = []

load_names = []
anno_names = []
all_orig_img = []
labels = []
names = []
max_count = 20
count = 0
for imgs, labels, names in tqdm(data_loader, total=len(data_loader)):
    imgs = imgs.to(device)
    count += 1
    if save_encode:
        cond= model_diff.encode(imgs)
        img_cond.append(cond.cpu().numpy())

    npy_name = f"{names[0].split('.')[0]}_BS{BATCH_SIZE}_count{count}.npy"
    load_names.append(npy_name)
    labels.append(labels)
    names.append(names)
    
    orig_img_show = ((imgs + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
    all_orig_img.append(orig_img_show)
    
    if count >= max_count:
        all_orig_img = np.stack(all_orig_img, axis=0)
        labels = np.stack(labels, axis=0)
        dataname = 'fibrosis' if labels[0][0] == 1 else 'no_fibrosis'
        print('data loaded: ',dataname)
        print(f'len(load_names): {len(load_names)}, all_orig_img.shape: {all_orig_img.shape}')

        if save_encode:
            img_cond = np.stack(img_cond, axis=0)

            tmp = f'/media/NAS06/gavinyue/disentanglement/scripts_segmentation/npy/osic/manip_shuffled/{dataname}/{exp}/cond/recon'
            
            os.makedirs(tmp,exist_ok=True)
            # Check if the file exists
            if os.path.exists(f'{tmp}/count{count}.npy'):
                # If the file exists, raise an error
                raise FileNotFoundError(f"The file '{tmp}/count{count}.npy' already exists.")
            
            np.save(f'{tmp}/count{count}.npy',img_cond)
            print(f'{tmp}/count{count}.npy Saved!')