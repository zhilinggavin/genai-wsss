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
logging.getLogger("PIL").setLevel(logging.WARNING)
# logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
# logging.getLogger("imageio").setLevel(logging.WARNING)

'''
    Load dataset
'''
IMG_ROOT = 'data/OSIC/processed'
BATCH_SIZE = 10 #10
IMG_COUNT = 200 #200
fibrosis_dir = os.path.join(IMG_ROOT, 'fibrosis')
no_fibrosis_dir = os.path.join(IMG_ROOT, 'no_fibrosis')

def get_label_map(dir1: str, dir2: str) -> dict[str, int]:
    '''
    Get image filename to label mapping.
    Label 1: fibrosis
    Label 0: no_fibrosis
    '''
    
    imgfib_names = sorted(f for f in os.listdir(dir1) if f.endswith('.png'))
    imgnofib_names = sorted(f for f in os.listdir(dir2) if f.endswith('.png'))
    label1_map = {name: 1 for name in imgfib_names}
    # img_label_map.update({name: 0 for name in imgnofib_names})
    label0_map = {name: 0 for name in imgnofib_names}
    return label1_map, label0_map


label1_map, label0_map = get_label_map(fibrosis_dir, no_fibrosis_dir)

filenames_label1 = list(label1_map.keys())
filenames_label0 = list(label0_map.keys())
# labels = [img_label_map[name] for name in filenames]

# deterministically shuffle dataset (seeded)
def shuffle_list(input_list, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(input_list), generator=g).tolist()
    return [input_list[i] for i in perm]

shuffled_filenames_label1 = sorted(shuffle_list(filenames_label1)[0:IMG_COUNT])
shuffled_filenames_label0 = sorted(shuffle_list(filenames_label0)[0:IMG_COUNT])
shuffled_filenames_label0_manip = sorted(shuffle_list(filenames_label0)[IMG_COUNT:IMG_COUNT*2])

logging.info(f"Shuffled filenames label 1 (first {IMG_COUNT}): {shuffled_filenames_label1[:5]}")
logging.info(f"Shuffled filenames label 0 (first {IMG_COUNT}): {shuffled_filenames_label0[:5]}")
logging.info(f"Shuffled filenames label 0 manipulation (first {IMG_COUNT}): {shuffled_filenames_label0_manip[:5]}")

# save as csv for human-readable mapping
save_list_dir = f'experiments/diffae_usage/encoded_shuffled/bs_{BATCH_SIZE}'
os.makedirs(save_list_dir, exist_ok=True)
import csv
csv_path = os.path.join(save_list_dir, 'shuffled_filenames_labels.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'label'])
    for fn in shuffled_filenames_label1:
        writer.writerow([fn, 1])
    for fn in shuffled_filenames_label0:
        writer.writerow([fn, 0])
    for fn in shuffled_filenames_label0_manip:
        writer.writerow([fn, -1])  # use label -1 for manipulation set

logging.info(f"Saved shuffled lists to {save_list_dir} (n1={len(shuffled_filenames_label1)}, n0={len(shuffled_filenames_label0)})")


dataset_label1 = Dataset_diffae_osic(shuffled_filenames_label1, [1]*len(shuffled_filenames_label1))
dataset_label0 = Dataset_diffae_osic(shuffled_filenames_label0, [0]*len(shuffled_filenames_label0))


dataloader_label1 = DataLoader(dataset_label1, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
dataloader_label0 = DataLoader(dataset_label0, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

logging.info("Datasets for label 1 and label 0 are loaded successfully.")
img, label, name = dataset_label1[0]
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
img_xT = []

all_labels = []
flat_names = []
# max_count = 20
count = 0
for imgs, labels, names in tqdm(dataloader_label1, total=len(dataloader_label1)):
    imgs = imgs.to(device)
    count += 1
    
    # # Classification check
    # with torch.no_grad():
    #     logits = model_cls(imgs)
    #     preds = torch.argmax(logits, dim=1)
    #     acc = (preds == labels.to(device)).float().mean().item()
    #     logging.info(f'Batch {count}, Classification Accuracy: {acc*100:.2f}%')
    #     if acc < 0.9:
    #         logging.warning(f'Batch {count}, Classification Accuracy below threshold: {acc*100:.2f}%')
    #         pass

    if save_encode:
        cond= model_diff.encode(imgs)
        img_cond.append(cond.cpu())
        
        # get xT to store encoded img general information
        # xT = model_diff.encode_stochastic(imgs.to(device), cond, T=250)
        # recon = model_diff.render(xT, cond, T=100)
        # img_xT.append(xT.cpu())
        
        all_labels.append(labels)
        flat_names.extend([str(n) for n in names])

save_name = f"label1_bs{BATCH_SIZE}_count{count}.pt"


if save_encode:
    # img_cond = np.stack(img_cond, axis=0)
    img_cond = torch.cat(img_cond, dim=0)
    img_xT = torch.cat(img_xT, dim=0)
    img_labels = torch.cat(all_labels, dim=0)

    dataname = 'fibrosis' if img_labels.all() == 1 else 'no_fibrosis'
    logging.info('data loaded: %s', dataname)
    save_dir = f'experiments/diffae_usage/encoded_shuffled/bs_{BATCH_SIZE}/{dataname}'
    os.makedirs(save_dir, exist_ok=True)
    

    savepath = os.path.join(save_dir, save_name)
    # 1) PyTorch checkpoint (common for deep-learning pipelines)
    torch.save({
        'cond': img_cond,   # tensor (N, 256)
        # 'xT': img_xT,
        'names': flat_names,        # list of strings
        'labels': img_labels        # tensor (N,)
    }, savepath)
    logging.info(f'{savepath} Saved!')