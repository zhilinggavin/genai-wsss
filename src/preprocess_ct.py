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
from debug_inference import model_loading, infer_main
import torch
from datetime import datetime
import sys
sys.path.append("..")
from utils.preprocess_ct import image_3D_normalisation



def segmentation_inference(model, device, img_scale, base_dir, case_name, csv_file_path, save_dir = None):
    infer_main(model, device, base_dir, case_name, img_scale, csv_file_path, save_dir)
    print(f'Case {case_name} segmentation inference done!')
    
def calculate_slice_volumes(csv_file_path):
    '''
    Calculate the slice_volume_lung | slice_volume_fibrosis
    '''
    df_slice = pd.read_csv(csv_file_path)
    if ('slice_volume_lung' and 'slice_volume_fibrosis') not in df_slice.columns:
        print('Add volume_lung and volume_fibrosis to the DataFrame')
        
        voxel_data = np.array(df_slice['voxel'].values)
        pixel_num_lung = np.array(df_slice['pixel_num_lung'].values)
        pixel_num_fibrosis = np.array(df_slice['pixel_num_fibrosis'].values)

        # Calculate the volume of lung and fibrosis
        slice_volume_lung = voxel_data * pixel_num_lung
        slice_volume_fibrosis = voxel_data * pixel_num_fibrosis

        df_slice['slice_volume_lung'] = slice_volume_lung
        df_slice['slice_volume_fibrosis'] = slice_volume_fibrosis

        df_slice.to_csv(csv_file_path, index=False)
        print(f'Case {basename(csv_file_path)} slice volume calculated and saved successfully')
    else:
        print(f'Case {basename(csv_file_path)} slice volume already exists')

def quantify_volumes(csv_file_path):
    '''
    Caluculate the volume_lung | volume_fibrosis for each case
    '''
    quantify_path = os.path.join(os.path.dirname(csv_file_path), 'quantification.csv')

    df_slice = pd.read_csv(csv_file_path)
    case_name = df_slice['Case'].values[0]

    slice_volume_lung = np.array(df_slice['slice_volume_lung'].values)
    slice_volume_fibrosis = np.array(df_slice['slice_volume_fibrosis'].values)

    volume_lung = np.sum(slice_volume_lung)
    volume_fibrosis = np.sum(slice_volume_fibrosis)

    data = {
        'Case': [case_name],
        'volume_lung': [volume_lung],
        'volume_fibrosis': [volume_fibrosis]
    }

    df_quantify = pd.DataFrame(data)
    df_quantify.to_csv(quantify_path, mode='a', header=not os.path.exists(quantify_path), index=False)
    print(f'Case {case_name} volume calculated and saved successfully')   



if __name__ == "__main__":
    # Specify the root directory for the raw files - .nii.gz files
    IMAGE_SIZE = [256, 256]
    LUNG_PIXEL_THRESHOLD = 0 # 0: no threshold, 400: 400 pixels
    INFER = True # True: run inference, False: only preprocess
    MODEL_NAME = 'full_supervised_unet' # Model name for loading the model
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
    csv_file_path = join(save_dir, 'slice_check_test.csv')
    infer_csv_file_path = join(infer_save_dir, 'quant','infer_slice_check.csv')
    
    
    

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
        model_path = '../experiments/full_supervised_unet/checkpoints/fold5_best_dice_epoch205.pth'
        model = model_loading(model_path, device)
        model.eval()

    case_count = 0
    for ct_name in tqdm(cts):
        print(f'count: {case_count}, ct_name: {ct_name}')
        case_count += 1

        case_name = ct_name.split('.')[0]
        case_dir = join(base_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)
        

        
        print(f"\nProcessing file: {ct_name}")

        ct_512_orig = os.path.join(root_directory, ct_name)
        

        volume_case = 0

        try:
            ct_512_orig_stik = sitk.ReadImage(ct_512_orig) 
        except:
            print("CT file read failed!")
            continue

        ct_512_orig_np = sitk.GetArrayFromImage(ct_512_orig_stik)#shape (slices, 512, 512)
        # ct_512_fibrosis_np = sitk.GetArrayFromImage(ct_512_fibrosis_stik)
        
        # Get spacing information (voxel size in mm)
        spacing = ct_512_orig_stik.GetSpacing()  # Returns (x_spacing, y_spacing, z_spacing)
        slice_thickness, pixel_spacing_x, pixel_spacing_y = spacing[2], spacing[0], spacing[1]  # Z-spacing is the slice thickness
        voxel_volume = pixel_spacing_x * pixel_spacing_y * slice_thickness
        # print(f"Voxel volume: {voxel_volume:.6f} mm³")
        

        ct_512_mask_np = mask.apply(ct_512_orig_np) # 0: background, 1: right-lung mask, 2: left-lung mask
        ct_512_mask_binary_np = copy.deepcopy(ct_512_mask_np) #range[0,2],uint8, 512x512
        ct_512_mask_binary_np[ct_512_mask_binary_np != 0] = 1
        
        
        
        # # get slices with lung mask bigger than 400 pixels
        if LUNG_PIXEL_THRESHOLD:            
            first_slice = None
            # TODO: remove pixel threshold
            for i in range(ct_512_mask_binary_np.shape[0]):
                if np.sum(ct_512_mask_binary_np[i] > 0) > 400:
                    if first_slice is None:
                        first_slice = i
                    last_slice = i
            if first_slice is None:
                raise ValueError(f"No slice found with lung mask bigger than 400 pixels for case {basename(case)} in shape of 512")
            print(f'Case {case_name} has lung mask from slice {first_slice} to {last_slice}')
        else:
            first_slice = 0
            last_slice = ct_512_orig_np.shape[0] - 1
        
        
    
        total_pixel = 0
        
        print("slice preprocessing Started")

        for z in range(first_slice, last_slice+1):
            slice = ct_512_orig_np[z]
            binary_mask_lung = ct_512_mask_binary_np[z]

            # break
            '''
            read parameters for voxel
            np_slice: 512*512*Z
            '''
            
            '''
            segmentation model input dtype:
            uint8 [0,256], shape 256*256, 3 channels
            '''



            slice_norm = image_3D_normalisation(slice) #norm to [0,1]
            slice_masked = np.where(binary_mask_lung, slice_norm, 0)
            
            # convert to uint8, this is the input for segmentation model
            slice_masked = (slice_masked * 255).astype(np.uint8) #[0,255], shape 512,512. 2 channels

            
            slice_name = f'case{case_name}_slice{z:03d}'
            img_slice_masked = Image.fromarray(slice_masked)
            '''Preprocessing Image for Input of Segmentation model'''
            # TODO
            # This step resizes the image to 256x256 using the same resize method as the previous image preprocessing.
            # Consider changing to the PIL resize method used in dataset processing for model input: img.resize((newW, newH), resample=Image.BICUBIC)
            # img_slice_resized = trans_fn.resize(img_slice_masked.convert("RGB"), [256], InterpolationMode.LANCZOS)
            pixel_num_lung = 0
            if not os.path.exists(f'{case_dir}/{slice_name}.png'):
                img_slice_resized = trans_fn.resize(img_slice_masked.convert("RGB"), [256], InterpolationMode.LANCZOS)
                img_slice_resized.save(f'{case_dir}/{slice_name}.png') #type: ignore
            
            else:
                img_slice_resized = Image.open(f'{case_dir}/{slice_name}.png')

            np_slice_resized = np.array(img_slice_resized)
            pixel_num_lung = np.sum(np_slice_resized != 0)

            # assert slice_masked.max() >= 0.9, f"slice_masked.max()={slice_masked.max()}"
            
            # np_slice_resized = np.array(img_slice_resized)
            # pixel_num_lung = np.sum(np_slice_resized != 0)
            
            
            data = {
                'Case': [case_name],
                'ID': [slice_name],
                'size': [img_slice_resized.size[0]],
                'voxel': [voxel_volume],
                'pixel_num_lung': [pixel_num_lung],
                'slice_volume_lung': [voxel_volume * pixel_num_lung]
            }
            # if pixel_num_lung > 0:
            # data['pixel_num_lung'] = [pixel_num_lung]
            
            df = pd.DataFrame(data)
            df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)

        print(f'Case {basename(case_name)} processed and saved successfully')
        # if case_count == 2:
        #     break
    
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
    print("All cases processed and saved successfully")

