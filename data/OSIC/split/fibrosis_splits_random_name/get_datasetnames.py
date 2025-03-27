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
dir_img = Path('/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/fid/fibrosis/fid_dataset/orig')
dir_mask = Path('/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/fid/fibrosis/fid_dataset/orig_mask')

dataset = FibNameDataset(dir_img, dir_mask, mask_suffix='_mask')

# Split dataset into 80% training and 20% testing (test set as standard for all No. experiments)
train_indices, test_indices = train_test_split(range(12625), test_size=0.2, random_state=20)
train_dataset = Subset(dataset, train_indices)
assert len(train_dataset) == 10100


# 3. Get the last fold's (best fold-5) indices from Real Fibrosis in No1.
fold = 4
kf = KFold(n_splits=5, shuffle=True, random_state=20) 
train_idx, val_idx = list(kf.split(train_dataset))[4]
train_subset = Subset(train_dataset, train_idx)
val_subset = Subset(train_dataset, val_idx)
assert len(train_subset) == 8080
assert list(val_idx[:10]) == [ 2,  8, 15, 16, 26, 30, 43, 46, 48, 51]
assert len(val_subset) == 2020
print('Valset: Best fold-5 from Real Fibrosis in No1')




test_dataset = Subset(dataset, test_indices)
assert test_dataset.indices[:10] == [8350, 12466, 3059, 809, 2275, 8584, 4956, 5802, 5334, 3247]
assert len(test_dataset) == 2525
print(f'len(test_dataset): {len(test_dataset)}')

logging.info(f'''
        Train set: {len(train_subset)}
        Val set: {len(val_subset)}
        Test set: {len(test_dataset)}
      ''')

save_path = "/media/NAS06/gavinyue/disentanglement/benchmark/DuPL/datasets/fibrosis"
train_name = join(save_path, 'train.txt')
val_name = join(save_path, 'val.txt')
test_name = join(save_path, 'test.txt')

open(train_name, 'a').close()
open(val_name, 'a').close()
open(test_name, 'a').close()



# with open(test_name, 'a') as file:
#     for name in tqdm(test_dataset):
#         file.write(name + '\n')
    
# with open(val_name, 'a') as file:
#     for name in tqdm(val_subset):
#         file.write(name + '\n')
        
# with open(train_name, 'a') as file:
#     for name in tqdm(train_subset):
#         file.write(name + '\n')

