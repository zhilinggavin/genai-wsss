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
# from infer_wsss_unet_YYF30Case import model_loading, make_dataloader, infer
from infer_wsss_unet_OSIC import model_loading, make_dataloader, infer


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
    img_dir = '../data/OSIC/processed/fibrosis'
    
    model_path = os.path.join(model_root, 'checkpoints', model_name)
    save_base_dir = os.path.join(model_root, 'results', 'OSIC')
    model = model_loading(model_path, device)
    
    # data loading
    # Load test set case IDs from CSV
    df = pd.read_csv('../data/OSIC/doctor_category.csv')
    caseids_test = df['case_id'][df['test'] == 1].tolist()
    caseids_test = [str(caseid).zfill(3) for caseid in caseids_test]
    
    
    for caseid in tqdm(caseids_test, leave=True):
        file_names = sorted(os.listdir(img_dir))
        file_names = [name for name in file_names if name.startswith(caseid) and name.endswith(('.png', '.jpg', '.jpeg'))]
        
        
        data_loader = make_dataloader(img_dir, img_scale=0.5, ids = file_names)
        
        save_dir = join(save_base_dir,'pred_mask')
        os.makedirs(save_dir, exist_ok=True)
        
        csv_file_path = join(save_base_dir, 'quant', 'slice_check.csv')
        os.makedirs(join(save_base_dir, 'quant'), exist_ok=True)
        
        # Run inference
        infer(model, data_loader, device, save_dir, csv_file_path)
        
        # break