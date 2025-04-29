'''
This code is a copy from orginal preporess script /media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/save_nofibrosis_covid_dataset.py
Not test yet for this new repo
DO NOT RUN DIRECTLY!!
'''
raise ImportError

import cv2
import numpy as np
import copy
import cv2
import os
import matplotlib.pyplot as plt
import random
# from utils.train import setup_seed
import os, glob, json
import SimpleITK as sitk
import sys
sys.path.append("..")
from image_processing import image_3D_normalisation
from lungmask import mask
from tqdm import tqdm

def bbox_3D(img):
    # rows, columns, and slices (depth)
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return [rmin, rmax, cmin, cmax, zmin, zmax]

def extract_number(filename):
    import re
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

'''
Load all annotated cts
'''
# cts = os.listdir()
# fibrosis

# path = '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/fibrosis_annotation/'
path = '/media/NAS03/yyfang/dataset/Saber_Italy/512_orig/'

# path = '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/fibrosis'
# path = '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/no_fibrosis'
files = os.listdir(path)
# Sort the files based on their numeric value
files = sorted(files, key=extract_number)

# 0001_20200416.nii.gz
pattern = '_fibrosis.nii.gz'
fibrosis = [file for file in files if file.endswith(pattern)]
original = [file for file in files if 'fibrosis' not in file]
print(f"Total annotated files: {len(fibrosis)} \nTotal original files: {len(original)}")

count = 0
for ct_original in tqdm(original):
    print(f'count: {count}, ct_original: {ct_original}')
    count += 1

    ct_512_orig = os.path.join(path, ct_original)

    try:
        ct_512_orig_stik = sitk.ReadImage(ct_512_orig)
        # ct_512_fibrosis_stik = sitk.ReadImage(ct_512_fibrosis)
    except:
        print("failed!")
        continue

    ct_512_orig_np = sitk.GetArrayFromImage(ct_512_orig_stik)
    # ct_512_fibrosis_np = sitk.GetArrayFromImage(ct_512_fibrosis_stik)

# get mask of lung (perform lung segmentation)
    ct_512_mask_np = mask.apply(ct_512_orig_np) #to perform lung segmentation
    ct_512_mask_binary_np = copy.deepcopy(ct_512_mask_np) #range[0,2],uint8, 512x512
    ct_512_mask_binary_np[ct_512_mask_binary_np != 0] = 1
    perform_crop = True
    if perform_crop:
        try:
            rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(ct_512_mask_binary_np)
        except:
            # print("{}: crop failure".format(ct_fibrosis))
            print("{}: crop failure".format(ct_original))
            continue
    ct_box_orig = ct_512_orig_np[rmin:rmax, cmin:cmax, zmin:zmax]
    ct_box_mask = ct_512_mask_np[rmin:rmax, cmin:cmax, zmin:zmax]
    ct_box_mask_binary = ct_512_mask_binary_np[rmin:rmax, cmin:cmax, zmin:zmax]
    # ct_box_fibrosis = ct_512_fibrosis_np[rmin:rmax, cmin:cmax, zmin:zmax]

    # resample
    z_size = ct_box_orig.shape[0]
    ct_350_orig = np.zeros([z_size, 350, 350])
    ct_350_mask = np.zeros([z_size, 350, 350])
    ct_350_mask_binary = np.zeros([z_size, 350, 350])
    # ct_350_fibrosis = np.zeros([z_size, 350, 350])
    for z in range(10,z_size,10):
        ct_350_orig[z, :, :] = cv2.resize(ct_box_orig[z, :, :], [350, 350], interpolation=cv2.INTER_AREA)
        ct_350_mask[z, :, :] = cv2.resize(ct_box_mask[z, :, :], [350, 350], interpolation=cv2.INTER_NEAREST)
        ct_350_mask_binary[z, :, :] = cv2.resize(ct_box_mask_binary[z, :, :], [350, 350], interpolation=cv2.INTER_NEAREST)
        # ct_350_fibrosis[z, :, :] = cv2.resize(ct_box_fibrosis[z, :, :], [350, 350], interpolation=cv2.INTER_NEAREST)


    ct_350_orig_stik = sitk.GetImageFromArray(ct_350_orig)
    ct_350_mask_sitk = sitk.GetImageFromArray(ct_350_mask)
    ct_350_mask_binary_stik = sitk.GetImageFromArray(ct_350_mask_binary)
    # ct_350_fibrosis_sitk = sitk.GetImageFromArray(ct_350_fibrosis)



    # selected slices, the first and last 10 slices will be excluded
    for z in range(10,z_size,10):
  
        slice = ct_350_orig[z, :, :]
        slice_mask = ct_350_mask[z, :, :]
        slice_mask_binary = ct_350_mask_binary[z, :, :]
        # slice_fibrosis = ct_350_fibrosis[z, :, :]
        # num_fibrosis = sum(sum(slice_fibrosis))
        slice_masked = image_3D_normalisation(slice) * slice_mask_binary
        imgname = ct_original.replace('.nii.gz', '_'+str(z)+'.png')

        save_folder = '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/'
        os.makedirs(os.path.join(save_folder, 'no_fibrosis'), exist_ok=True)
        os.makedirs(os.path.join(save_folder, 'no_fibrosis_covid'), exist_ok=True)
        os.makedirs(os.path.join(save_folder, 'fibrosis'), exist_ok=True)
        

        tmp_save1 = os.path.join(save_folder, 'no_fibrosis_covid', imgname)

            


            # #
            # fig, ax = plt.subplots(1, 3, figsize=(30,10))
            # ax[0].imshow(slice, cmap='gray')
            # ax[0].set_title(f'slice({z})')
            # ax[1].imshow(slice_masked, cmap='gray')
            # ax[1].set_title(f'slice_masked')
            # ax[2].imshow(slice_fibrosis, cmap='gray')
            # ax[2].set_title(f'slice_fibrosis')
            # plt.show()
            # plt.savefig('tmp.png')
            # print('saved')
            # break
        cv2.imwrite(tmp_save1, slice_masked * 255)

        
    # break
