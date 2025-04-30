import numpy as np
import copy
import cv2
import os
import matplotlib.pyplot as plt
import random
# from utils.train import setup_seed
import os, glob
import SimpleITK as sitk
from lungmask import mask
from tqdm import tqdm
from pathlib import Path
from torchvision.transforms import functional as trans_fn
from PIL import Image


def bbox_3D(img):
    # rows, columns, and slices (depth)
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    # selects the first ([0]) and last ([-1]) elements from the array of indices
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return [rmin, rmax, cmin, cmax, zmin, zmax]

def extract_number(filename):
    # exact the numerical numvers from the left
    import re
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

def image_3D_normalisation(npImage, min_value=-1024, max_value=-100):

    # crop
    npImage_norm = npImage
    npImage_norm[npImage < min_value] = min_value
    npImage_norm[npImage > max_value] = max_value

    # norm
    npImage_norm = (npImage_norm-min_value)/(max_value-min_value)

    # normalization: x-y
    # npImage_resample_adjust1 = (npImage_resample_adjust - min_value) / (max_value - min_value)
    # slice = npImage_resample_adjust1[16, :, :]
    # print("调节窗口窗位之后CT值的范围位为{}~{}".format(np.min(slice), np.max(slice)))
    # plt.figure(figsize=(5, 5))
    # plt.imshow(slice, 'gray')
    # plt.show()

    return npImage_norm

if __name__ == "__main__":
    IMAGE_SIZE = [256, 256]
    anno_label = False
    perform_crop = True #True, False
    GPU_ID = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    # Specify the root directory
    root_dir = "../data/YYF_30Case" # Replace with your folder path
    img_path = os.path.join(root_dir, "raw")
    assert os.path.exists(root_dir), f"Error: The directory '{root_dir}' does not exist."
    assert os.path.exists(img_path), f"Error: The directory '{root_dir}' does not exist."

    '''
    Load all annotated cts
    '''
    # path = '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/fibrosis_annotation/'
    files = os.listdir(img_path)
    # Sort the files based on their left numeric value
    files = [file for file in files if file.endswith('.nii.gz')]
    files = sorted(files, key=extract_number)

    if anno_label:
        anno_suffix = '_fibrosis.nii.gz'
        annos = [file for file in files if file.endswith(anno_suffix)]
        cts = [file for file in files if not file.endswith(anno_suffix)]
        print(f"Total annotated files: {len(annos)} \nTotal CT files: {len(cts)}")
    else:
        cts = files
        print(f"No annotated files;\nTotal CT files: {len(cts)}")
    
    '''
    The Anno file preprocss function is disabled in this file. To be corrected.
    '''

    count = 0
    for ct_name in tqdm(cts):
        print(f'count: {count}, ct_name: {ct_name}')
        count += 1

        ct_512_orig = os.path.join(img_path, ct_name)
        # ct_512_fibrosis = os.path.join(path, ct_name)
        
        try:
            ct_512_orig_stik = sitk.ReadImage(ct_512_orig) #shape (slices, 512, 512)
            # ct_512_fibrosis_stik = sitk.ReadImage(ct_512_fibrosis)
        except:
            print("CT file read failed!")
            continue

        ct_512_orig_np = sitk.GetArrayFromImage(ct_512_orig_stik)
        # ct_512_fibrosis_np = sitk.GetArrayFromImage(ct_512_fibrosis_stik)


        ct_512_mask_np = mask.apply(ct_512_orig_np) #to perform lung segmentation
        ct_512_mask_binary_np = copy.deepcopy(ct_512_mask_np) #range[0,2],uint8, 512x512
        ct_512_mask_binary_np[ct_512_mask_binary_np != 0] = 1

        if perform_crop:
            try:
                rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(ct_512_mask_binary_np)
            except:
                print("{}: crop failure".format(ct_name))
                continue
            ct_box_orig = ct_512_orig_np[rmin:rmax, cmin:cmax, zmin:zmax]
            # ct_box_mask = ct_512_mask_np[rmin:rmax, cmin:cmax, zmin:zmax]
            ct_box_mask_binary = ct_512_mask_binary_np[rmin:rmax, cmin:cmax, zmin:zmax]
            # ct_box_fibrosis = ct_512_fibrosis_np[rmin:rmax, cmin:cmax, zmin:zmax]
        else:
            ct_box_orig = ct_512_orig_np
            ct_box_mask_binary = ct_512_mask_binary_np

        # resample
        z_size = ct_box_orig.shape[0]
        ct_350_orig = np.zeros([z_size, *IMAGE_SIZE])
        ct_350_mask = np.zeros([z_size, *IMAGE_SIZE])
        ct_350_mask_binary = np.zeros([z_size, *IMAGE_SIZE])
        # ct_350_fibrosis = np.zeros([z_size, 350, 350])
        for z in range(z_size):
            # ct_350_orig[z, :, :] = cv2.resize(ct_box_orig[z, :, :], [256, 256], interpolation=cv2.INTER_AREA)
            # ct_350_mask[z, :, :] = cv2.resize(ct_box_mask[z, :, :], [350, 350], interpolation=cv2.INTER_NEAREST)
            # ct_350_mask_binary[z, :, :] = cv2.resize(ct_box_mask_binary[z, :, :], [350, 350], interpolation=cv2.INTER_NEAREST)
            # # ct_350_fibrosis[z, :, :] = cv2.resize(ct_box_fibrosis[z, :, :], [350, 350], interpolation=cv2.INTER_NEAREST)
            
            # Resize slices using LANCZOS for continuous data and NEAREST for masks; DO not use CV2
            ct_350_orig[z, :, :] = np.array(trans_fn.resize(
                Image.fromarray(ct_box_orig[z, :, :]), IMAGE_SIZE, interpolation=trans_fn.InterpolationMode.LANCZOS
            ))
            # ct_350_mask[z, :, :] = np.array(trans_fn.resize(
            #     Image.fromarray(ct_box_mask[z, :, :]), IMAGE_SIZE, interpolation=trans_fn.InterpolationMode.LANCZOS
            # ))
            ct_350_mask_binary[z, :, :] = np.array(trans_fn.resize(
                Image.fromarray(ct_box_mask_binary[z, :, :]), IMAGE_SIZE, interpolation=trans_fn.InterpolationMode.NEAREST
            ))


        # selected slices, the first and last 20 slices will be excluded range(20,z_size-20)
        for z in range(ct_350_mask.shape[0]): #for all slices, use range(ct_350_mask.shape[0])
    
            slice = ct_350_orig[z, :, :]
            slice_mask_binary = ct_350_mask_binary[z, :, :]
            # slice_fibrosis = ct_350_fibrosis[z, :, :]
            num_anno = sum(sum(slice_fibrosis)) if anno_label else 0
            slice_masked = image_3D_normalisation(slice) * slice_mask_binary #norm to 0-1
            
            # keep the original slice name and order
            imgname = ct_name.replace('.nii.gz', f'_{z+rmin:03d}.png') if perform_crop else ct_name.replace('.nii.gz', f'_{z:03d}.png')
            dir_name = "preprocessed_size256_cropped" if perform_crop else "preprocessed_size256"
            save_folder = os.path.join(root_dir, dir_name, ct_name.replace('.nii.gz',''))
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, imgname)
                
            slice_masked_img = Image.fromarray((slice_masked * 255).astype(np.uint8)).convert("RGB")
            slice_masked_img.save(save_path)
            
            # img = Image.open(save_path)
            # img_array = np.array(img)
            
        # break
