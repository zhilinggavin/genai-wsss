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
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

BATCH_SIZE = 10
# MANIP_STRENGTH = [1.0, 1.5, 2.0]  # 1.0, 1.5, 2.0
MANIP_STRENGTH = MS = 1.5
LOAD_ROOT = f'experiments/diffae_usage/encoded_shuffled/bs_{BATCH_SIZE}'
SAVE_DIR = 'experiments/tsne/tsne-20251011'

# load filename list
file_list = os.path.join(LOAD_ROOT, 'shuffled_filenames_labels.csv')
import csv
filenames = {}
with open(file_list, 'r') as f:
    reader = csv.reader(f)
    next(reader, None)  # skip header row if present
    for row in reader:
        filenames[row[0]] = int(row[1])
logging.info(f"Total {len(filenames)} files to decode.")

filenames_label0 = {k: v for k, v in filenames.items() if v == 0}
filenames_label1 = {k: v for k, v in filenames.items() if v == 1}
filenames_label0_manip = {k: v for k, v in filenames.items() if v == -1}
logging.info(f"Total {len(filenames_label0)} files for no_fibrosis (label 0). \n                {len(filenames_label1)} files for fibrosis (label 1). \n                {len(filenames_label0_manip)} files for manipulation (label -1).")


'''
    Get encoded condition space for all data (including manipulated)
'''
ms = MANIP_STRENGTH 
manip_dir = f'experiments/diffae_usage/encoded_shuffled/bs_10/manip_{ms}_nofib_fib'
dataset_all = Dataset_diffae_osic(list(filenames.keys()), list(filenames.values()), manip_dir=manip_dir)
dataloader_all = DataLoader(dataset_all, batch_size=BATCH_SIZE, shuffle=False)

save_path = os.path.join(SAVE_DIR, f'cond_manip{ms}_bs{BATCH_SIZE}_all600.npy')

'''
    Load ISBI classification model
'''
device = 'cuda'
_, model_cls = model_load(device, diff=False, cls=True)
direction_class_1 = model_cls.direction_class_1.detach().cpu().numpy()

list_label = []
list_pred = []
list_score = []
# load condition space and do classification check
load_path = os.path.join(SAVE_DIR, f'cond_manip{ms}_bs{BATCH_SIZE}_all600.npy')
cond_all = np.load(load_path)
cond_all = torch.tensor(cond_all, dtype=torch.float32)
# for imgs, labels, names in tqdm(dataloader_all):
for cond in tqdm(cond_all):
    cond = cond.unsqueeze(0)  # add batch dimension
    cond = cond.to(device)

    outputs = model_cls(cond)
    
    pred = torch.softmax(outputs, dim=1)
    # pred = F.softmax(outputs, dim=1).reshape(outputs.size()[0], -1)
    score = pred.data[:, 1]
    _, pred_class = torch.max(outputs, dim=1)

    # list_label.extend(labels.tolist())
    # list_pred.extend(pred_class.tolist())
    list_score.extend(score.tolist())

import pandas as pd
df = pd.DataFrame({
    'filename': list(filenames.keys()),
    'label': list(filenames.values()),
    'score': list_score
})
csv_save_path = os.path.join(SAVE_DIR, f'cls_scores_manip{ms}_bs{BATCH_SIZE}_all600.csv')
df.to_csv(csv_save_path, index=False)
logging.info(f"Saved classification scores to {csv_save_path}")