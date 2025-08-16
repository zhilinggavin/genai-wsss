import os
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
# Specify the root directory
from collections import defaultdict
# import matplotlib.pyplot as plt
import SimpleITK as sitk
from lungmask import mask
import numpy as np
from os.path import basename, dirname, join
from PIL import Image
import glob
import copy

from torchvision.transforms import functional as trans_fn
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from debug_inference import model_loading, infer_main, quantify_volumes
import torch
from datetime import datetime
import sys
sys.path.append("..")
# from utils.preprocess_ct import image_3D_normalisation



def segmentation_inference(model, device, img_scale, base_dir, case_name, csv_file_path, save_dir = None):
    infer_main(model, device, base_dir, case_name, img_scale, csv_file_path, save_dir)
    print(f'Case {case_name} segmentation inference done!')


if __name__ == "__main__":
    # Specify the root directory for the raw files - .nii.gz files
    IMAGE_SIZE = [256, 256]
    LUNG_PIXEL_THRESHOLD = 0 # 0: no threshold, 400: 400 pixels
    INFER = True # True: run inference, False: only preprocess
    MODEL_NAME = 'wsss_unet' # Model name for loading the model
    root_directory = "../data/CT/raw"  # Replace with your folder path
    assert os.path.exists(root_directory), f"Error: The directory '{root_directory}' does not exist."
    
    
    base_dir = '../data/CT/processed' # for saving the preprocessed datasets
    save_dir = '../data/CT/processed_quant' # for saving the quantification results
    infer_save_dir = f'../experiments/{MODEL_NAME}/results/CT'
    mask_save_dir = join(infer_save_dir, 'pred_mask') # for saving the lung mask


    # Add the current time postfix to the base_dir and save_dir
    # current_time = datetime.now()
    # formatted_time = current_time.strftime("%Y-%m-%d_%H-%M")
    # base_dir = base_dir + '_' + formatted_time
    # save_dir = save_dir + '_' + formatted_time
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)
    csv_file_path = join(save_dir, 'slice_check.csv')
    infer_csv_file_path = join(infer_save_dir, 'quant','infer_slice_check.csv')
    os.makedirs(dirname(infer_csv_file_path), exist_ok=True)
    

    # Print results
    files = os.listdir(root_directory)
    files = [file for file in files if file.endswith('.nii.gz')]
    cts = sorted(files)
    print(f"Number of CT cases: {len(cts)}")
    
    '''
    Model loading
    '''
    if INFER:
        device = 'cuda'
        # full_supervised unet model
        model_path = '../experiments/wsss_unet/checkpoints/fold5_best_dice_epoch45.pth'
        model = model_loading(model_path, device)
        model.eval()
    
    import shutil
    if os.path.exists(csv_file_path) and not os.path.exists(infer_csv_file_path):
        shutil.copy2(csv_file_path, infer_csv_file_path)    
    count = 0
    for ct_name in tqdm(cts):
        # count += 1
        # if count == 3:
        #     break
        case_name = ct_name.split('.')[0]
        mask_save_subdir = join(mask_save_dir, case_name)
        os.makedirs(mask_save_subdir, exist_ok=True)
        segmentation_inference(model, device, 0.5, base_dir, case_name, infer_csv_file_path, mask_save_subdir)
    
    # get volumes for each case
    df_quantify = quantify_volumes(infer_csv_file_path)
    print(df_quantify.head())
    print("All cases processed and saved successfully")

