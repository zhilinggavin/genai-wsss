import os
import sys
sys.path.append("")
from src.diffae.useage import model_load
from utils.datasets import Dataset_diffae_osic
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("PIL").setLevel(logging.WARNING)


'''
    Load selected dataset for manipulation. Different selection between fibrosis and no_fibrosis.
    Use the csv file generated in encode.py to load the filenames and labels.
    Label 0: no_fibrosis
    Label 1: fibrosis
    Label -1: no_fibrosis for manipulation
'''
BATCH_SIZE = 10
COUNT = 200 #Each class count
# MANIP_STRENGTH = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
MANIP_STRENGTH = [0]
SAVE_ROOT = f'experiments/diffae_usage/encoded_shuffled/bs_{BATCH_SIZE}_count{COUNT}'
for ms in MANIP_STRENGTH:
    SAVE_DIR = os.path.join(SAVE_ROOT, f'manip_{ms}_nofib_fib')
    os.makedirs(SAVE_DIR, exist_ok=True)

# load filename list
file_list = os.path.join(SAVE_ROOT, 'shuffled_filenames_labels.csv')
import csv
filenames = {}
with open(file_list, 'r') as f:
    reader = csv.reader(f)
    next(reader, None)  # skip header row if present
    for row in reader:
        filenames[row[0]] = int(row[1])
logging.info(f"Total {len(filenames)} files to decode.")

filenames_label0_manip = {k: v for k, v in filenames.items() if v == -1}
logging.info(f"Total {len(filenames_label0_manip)} files for manipulation (label -1).")
dataset_label0_manip = Dataset_diffae_osic(list(filenames_label0_manip.keys()), list(filenames_label0_manip.values()))
dataloader_label0_manip = DataLoader(dataset_label0_manip, batch_size=BATCH_SIZE, shuffle=False)

'''
    Load ISBI diff model and classification model
'''
device = 'cuda'
model_diff, model_cls = model_load(device, diff=True, cls=True)
direction_class_1 = model_cls.direction_class_1
logging.info("ISBI Diffae and Classification Models loaded successfully.")

'''
    Inference for manipulation
'''
count = 0
for imgs, labels, names in tqdm(dataloader_label0_manip, total=len(dataloader_label0_manip)):
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
    
    if not torch.all(labels == -1):
        raise ValueError(f"Error: Expected all labels to be -1, but got different labels from names: {names} with labels: {labels}")

    cond = model_diff.encode(imgs)
    xT = model_diff.encode_stochastic(imgs, cond, T=250)

    # Manipulation
    for ms in MANIP_STRENGTH:
        add = ms * direction_class_1

        recon = model_diff.render(xT, cond + add, T=100)
        recon_show = (recon.permute(0,2,3,1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy() #range[0,1]
        # img_manip.append(recon_show)
        for i in range(len(recon_show)):
            save_path = os.path.join(SAVE_ROOT, f'manip_{ms}_nofib_fib', f'{names[i].split(".")[0]}_manip{ms:.2f}.png')
            Image.fromarray(recon_show[i]).save(save_path)
    
logging.info(f'Saved manipulated image to {SAVE_DIR}')