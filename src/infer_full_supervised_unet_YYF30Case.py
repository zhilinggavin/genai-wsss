import os, glob
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from Pytorch_UNet.unet import UNet
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from os import listdir
from os.path import splitext, isfile, join
import pandas as pd
from infer_wsss_unet_YYF30Case import model_loading, make_dataloader, infer


# Constants
GPU_ID = '0'
IMAGE_SIZE = 256

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Set current working directory to the script's directory


if __name__ == '__main__': 
    '''
    load model
    '''
    device = 'cuda'
    # full_supervised unet model
    model_root = '../experiments/full_supervised_unet'
    model_name = 'fold5_best_dice_epoch205.pth'
    # root_dir = '../data/YYF_30Case/preprocessed_size256'
    root_dir = '../data/YYF_30Case/preprocessed_size256_cropped'
    
    model_path = os.path.join(model_root, 'checkpoints', model_name)
    save_base_dir = os.path.join(model_root, 'results', root_dir.split('data/')[-1])
    model = model_loading(model_path, device)
    
    # data loading
    # Load test set case IDs
    case_dirs = os.listdir(root_dir)
    assert len(case_dirs) == 30, f'No image dirs found in {root_dir}'
    for case_id in tqdm(case_dirs, leave=True):
        img_dir_path = os.path.join(root_dir, case_id)
        data_loader = make_dataloader(img_dir_path, img_scale=0.5, ids=None)
        
        save_dir = join(save_base_dir, 'pred_mask', case_id)
        os.makedirs(save_dir, exist_ok=True)
        
        csv_file_path = join(save_base_dir, 'quant', 'slice_check.csv')
        os.makedirs(join(save_base_dir, 'quant'), exist_ok=True)
        
        
        infer(model, data_loader, device, save_dir, csv_file_path)
        
        # break