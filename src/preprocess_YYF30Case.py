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
    # Specify the root directory for the raw files - DCM files
    root_directory = "../data/YYF_30Case/raw"  # Replace with your folder path
    assert os.path.exists(root_directory), f"Error: The directory '{root_directory}' does not exist."

    '''
    Load all annotated cts
    '''
    # cts = os.listdir()
    # fibrosis

    path = '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/fibrosis_annotation/'
    files = os.listdir(path)
    # Sort the files based on their numeric value
    files = sorted(files, key=extract_number)


    pattern = '_fibrosis.nii.gz'
    fibrosis = [file for file in files if file.endswith(pattern)]
    original = [file for file in files if 'fibrosis' not in file]
    print(f"Total annotated files: {len(fibrosis)} \nTotal original files: {len(original)}")

    count = 0
    for ct_fibrosis in tqdm(fibrosis):
        print(f'count: {count}, ct_fibrosis: {ct_fibrosis}')
        count += 1

        ct_512_orig = os.path.join(path, ct_fibrosis.replace('_fibrosis', ''))
        ct_512_fibrosis = os.path.join(path, ct_fibrosis)
        

        # ct_512_orig_np = sitk.GetArrayFromImage(path + ct_512_orig)
        # ct_512_masked = path + ct_fibrosis

        # ct_350_orig = datapath_350_3D + ct
        # ct_350_mask = datapath_350_mask + ct
        # ct_350_masked = savepath_350_masked + ct

        try:
            ct_512_orig_stik = sitk.ReadImage(ct_512_orig)
            ct_512_fibrosis_stik = sitk.ReadImage(ct_512_fibrosis)
        except:
            print("failed!")
            continue

        ct_512_orig_np = sitk.GetArrayFromImage(ct_512_orig_stik)
        ct_512_fibrosis_np = sitk.GetArrayFromImage(ct_512_fibrosis_stik)


        ct_512_mask_np = mask.apply(ct_512_orig_np) #to perform lung segmentation
        ct_512_mask_binary_np = copy.deepcopy(ct_512_mask_np) #range[0,2],uint8, 512x512
        ct_512_mask_binary_np[ct_512_mask_binary_np != 0] = 1
        perform_crop = True
        if perform_crop:
            try:
                rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(ct_512_mask_binary_np)
            except:
                print("{}: crop failure".format(ct_fibrosis))
                continue
        ct_box_orig = ct_512_orig_np[rmin:rmax, cmin:cmax, zmin:zmax]
        ct_box_mask = ct_512_mask_np[rmin:rmax, cmin:cmax, zmin:zmax]
        ct_box_mask_binary = ct_512_mask_binary_np[rmin:rmax, cmin:cmax, zmin:zmax]
        ct_box_fibrosis = ct_512_fibrosis_np[rmin:rmax, cmin:cmax, zmin:zmax]

        # resample
        z_size = ct_box_orig.shape[0]
        ct_350_orig = np.zeros([z_size, 350, 350])
        ct_350_mask = np.zeros([z_size, 350, 350])
        ct_350_mask_binary = np.zeros([z_size, 350, 350])
        ct_350_fibrosis = np.zeros([z_size, 350, 350])
        for z in range(z_size):
            ct_350_orig[z, :, :] = cv2.resize(ct_box_orig[z, :, :], [350, 350], interpolation=cv2.INTER_AREA)
            ct_350_mask[z, :, :] = cv2.resize(ct_box_mask[z, :, :], [350, 350], interpolation=cv2.INTER_NEAREST)
            ct_350_mask_binary[z, :, :] = cv2.resize(ct_box_mask_binary[z, :, :], [350, 350], interpolation=cv2.INTER_NEAREST)
            ct_350_fibrosis[z, :, :] = cv2.resize(ct_box_fibrosis[z, :, :], [350, 350], interpolation=cv2.INTER_NEAREST)


        ct_350_orig_stik = sitk.GetImageFromArray(ct_350_orig)
        ct_350_mask_sitk = sitk.GetImageFromArray(ct_350_mask)
        ct_350_mask_binary_stik = sitk.GetImageFromArray(ct_350_mask_binary)
        ct_350_fibrosis_sitk = sitk.GetImageFromArray(ct_350_fibrosis)

        # sitk.WriteImage(ct_350_orig_stik, save_image_path)
        # sitk.WriteImage(ct_350_mask_sitk, save_image_segmentation_path)
        # sitk.WriteImage(ct_350_mask_sitk, save_image_segmentation_path)
        # ct_350_fibrosis_sitk


        # selected slices, the first and last 20 slices will be excluded
        for z in range(20,z_size-20): #for all slices, use range(ct_350_mask.shape[0])
    
            slice = ct_350_orig[z, :, :]
            slice_mask = ct_350_mask[z, :, :]
            slice_mask_binary = ct_350_mask_binary[z, :, :]
            slice_fibrosis = ct_350_fibrosis[z, :, :]
            num_fibrosis = sum(sum(slice_fibrosis))
            slice_masked = image_3D_normalisation(slice) * slice_mask_binary
            imgname = ct_fibrosis.replace('.nii.gz', '_'+str(z)+'.png')

            save_folder = '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/'
            os.makedirs(os.path.join(save_folder, 'no_fibrosis_selected'), exist_ok=True)
            os.makedirs(os.path.join(save_folder, 'fibrosis_selected'), exist_ok=True)
            if num_fibrosis == 0:
                tmp_save1 = os.path.join(save_folder, 'no_fibrosis_selected', imgname)
                tmp_save2 = os.path.join(save_folder, 'no_fibrosis_selected', imgname.replace('.png', '_anno.png'))
                
            else:
                # continue
                tmp_save1 = os.path.join(save_folder, 'fibrosis_selected', imgname)
                tmp_save2 = os.path.join(save_folder, 'fibrosis_selected', imgname.replace('.png', '_anno.png'))

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
            cv2.imwrite(tmp_save2, slice_fibrosis * 255) if num_fibrosis != 0 else None
            
        # break
