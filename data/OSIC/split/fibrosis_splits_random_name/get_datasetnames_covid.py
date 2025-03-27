from pathlib import Path
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import sys
sys.path.append("/media/NAS06/gavinyue/disentanglement/scripts_segmentation/Pytorch-UNet")
from utils.data_loading import FibNameDataset
from os.path import basename,join
from sklearn.model_selection import KFold,train_test_split
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')



# setup_seed(20)
dir_img = Path('/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/fid/no_fibrosis/fid_dataset/orig_covid')
# dir_mask = Path('/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/fid/fibrosis/fid_dataset/orig_mask')

dataset = FibNameDataset(dir_img, mask_dir=None)

# Split dataset into 80% training and 20% testing (test set as standard for all No. experiments)
train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=20)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, test_indices)


logging.info(f'''
        Train set: {len(train_dataset)}
        Val set: {len(val_dataset)}
      ''')

save_path = "/media/NAS06/gavinyue/disentanglement/benchmark/counterfactual-search/data/fibrosis_splits"
train_name = join(save_path, 'train_covid.txt')
val_name = join(save_path, 'val_covid.txt')

open(train_name, 'a').close()
open(val_name, 'a').close()



with open(train_name, 'a') as file:
    for name in tqdm(train_dataset):
        file.write(name + '\n')

with open(val_name, 'a') as file:
    for name in tqdm(val_dataset):
        file.write(name + '\n')