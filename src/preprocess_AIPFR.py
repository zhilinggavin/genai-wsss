import os
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
# Specify the root directory
from collections import defaultdict
# import matplotlib.pyplot as plt
import pydicom
from lungmask import mask
import numpy as np
from os.path import basename, dirname, join
from PIL import Image
import glob

from torchvision.transforms import functional as trans_fn
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
# from debug_inference import model_loading, infer_main
import torch

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # 转换为int16，int16是ok的，因为所有的数值都应该 <32k
    image = image.astype(np.int16)

    # 设置边界外的元素为0
    image[image == -2000] = 0
    # image[image == -2000] = -1024

    # 转换为HU单位
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def count_dcm_files_by_intervals(root_dir, interval=30, threshold=100):
    """
    Count the number of .dcm files in subfolders and group them by intervals.
    Also count cases where the number of .dcm files exceeds a threshold.

    :param root_dir: Root directory to start the search.
    :param interval: Interval for grouping counts.
    :param threshold: Threshold for counting cases.
    :return: Sorted list of intervals with case counts, total cases, and the count of cases above the threshold.
    """
    case_counts = []

    # Traverse the directory structure
    for root, _, files in os.walk(root_dir):
        # Count .dcm files in the current directory
        dcm_count = sum(1 for file in files if file.endswith('.dcm'))
        if dcm_count > 0:
            case_counts.append(dcm_count)

    # Group counts by intervals
    interval_counts = defaultdict(int)
    for count in case_counts:
        lower_bound = (count // interval) * interval
        upper_bound = lower_bound + interval
        interval_counts[(lower_bound, upper_bound)] += 1

    # Total number of cases
    total_cases = len(case_counts)

    # Count cases above the threshold
    cases_above_threshold = sum(1 for count in case_counts if count > threshold)

    return sorted(interval_counts.items()), total_cases, cases_above_threshold

def find_cases_below_threshold(root_dir, threshold):
    """
    Find cases (directories) with fewer than a specified number of .dcm files,
    and print the count of .dcm files for each case.

    :param root_dir: Root directory to start the search.
    :param threshold: Threshold for counting cases.
    :return: A tuple containing the count of cases and a list of tuples with case names and their .dcm file counts.
    """
    cases_below_threshold = []

    # Traverse the directory structure
    for root, _, files in os.walk(root_dir):
        # Count .dcm files in the current directory
        dcm_count = sum(1 for file in files if file.endswith('.dcm'))
        if dcm_count > 0 and dcm_count < threshold:
            cases_below_threshold.append((root, dcm_count))  # Append case name and count

    # Return the count and list of case names with their .dcm file counts
    return len(cases_below_threshold), cases_below_threshold

def find_cases_above_threshold(root_dir, threshold):
    """
    Find cases (directories) with fewer than a specified number of .dcm files,
    and print the count of .dcm files for each case.

    :param root_dir: Root directory to start the search.
    :param threshold: Threshold for counting cases.
    :return: A tuple containing the count of cases and a list of tuples with case names and their .dcm file counts.
    """
    cases_below_threshold = []

    # Traverse the directory structure
    for root, _, files in os.walk(root_dir):
        # Count .dcm files in the current directory
        dcm_count = sum(1 for file in files if file.endswith('.dcm'))
        if dcm_count >= threshold:
            cases_below_threshold.append((root, dcm_count))  # Append case name and count

    # Return the count and list of case names with their .dcm file counts
    return len(cases_below_threshold), cases_below_threshold

def plot_histogram(interval_statistics):
    """
    Plot a histogram based on interval statistics.

    :param interval_statistics: List of intervals with case counts.
    """
    # Prepare data for plotting
    intervals = [f"{low}-{high}" for (low, high), _ in interval_statistics]
    counts = [count for _, count in interval_statistics]

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.bar(intervals, counts, width=0.8)
    plt.xlabel("Number of DCM Files (Range)", fontsize=12)
    plt.ylabel("Number of Cases", fontsize=12)
    plt.title("Distribution of DCM Files Across Cases", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def get_voxel_size(dicom_file_path):
    """
    Get the voxel size (PixelSpacing and SliceThickness) from a DICOM file.

    :param dicom_file_path: Path to the .dcm file.
    :return: Tuple of (voxel_size, dicom_file_path) or None if information is incomplete.
    """
    try:
        # Read the DICOM file
        ds = pydicom.dcmread(dicom_file_path)
        # np_lung = XXX

        # Get PixelSpacing (row spacing, column spacing)
        if "PixelSpacing" in ds:
            pixel_spacing = ds.PixelSpacing
        else:
            print(f"PixelSpacing not found in file: {dicom_file_path}")
            return None

        # Get SliceThickness (distance between slices)
        slice_thickness = float(ds.SpacingBetweenSlices)
        # if "SliceThickness" in ds:
        #     slice_thickness = float(ds.SliceThickness)
        # elif "SpacingBetweenSlices" in ds:
        #     slice_thickness = float(ds.SpacingBetweenSlices)
        # else:
        #     print(f"SliceThickness or SpacingBetweenSlices not found in file: {dicom_file_path}")
        #     return None

        # Combine to form voxel size
        voxel_size = (pixel_spacing[0], pixel_spacing[1], slice_thickness)
        return voxel_size, np_slice
    except Exception as e:
        print(f"Error reading {dicom_file_path}: {e}")
        return None

def preprocessing(slice):

    return 

# def segmentation_inference(model, device, img_scale, base_dir, case_name, csv_file_path, save_dir = None):
#     infer_main(model, device, base_dir, case_name, img_scale, csv_file_path, save_dir)
#     print(f'Case {case_name} segmentation inference done!')
    
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

# Specify the root directory and threshold
# root_directory = "/path/to/your/folder"  # Replace with your folder path
# threshold = 100  # Replace with your desired threshold
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

import copy
# Count and plot results

if __name__ == "__main__":
    # Specify the root directory for the raw files - DCM files
    root_directory = "../data/AIPFR/raw/AUSTRALIAN_REGISTRY"  # Replace with your folder path
    assert os.path.exists(root_directory), f"Error: The directory '{root_directory}' does not exist."
    
    '''
    Model loading
    '''
    
    # device = 'cuda'
    
    # full_supervised unet model
    # model_path = '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/unet_checkpoints/No1_Real_eps300_bs20/fold5_best_dice_epoch205.pth'
    base_dir = '../data/AIPFR/processed_test' # for saving the preprocessed datasets 
    save_dir = '../data/AIPFR/processed_quant' # for saving the quantification results 
    # model = model_loading(model_path, device)



    threshold = 150  
   


    # Print the total number of cases and the count of cases above the threshold
    interval_statistics, total_cases, interval_count_above_threshold = count_dcm_files_by_intervals(root_directory, threshold=threshold)
    # plot_histogram(interval_statistics)
    print(f"Total number of cases: {total_cases}")
    # print(f"Number of cases with more than {threshold} DCM files: {interval_count_above_threshold+1}")

    # Print results
    count_below_threshold, cases_below_threshold = find_cases_below_threshold(root_directory, threshold)
    print(f"Number of cases with fewer than {threshold} DCM files: {count_below_threshold}")

    count_above_threshold, cases_above_threshold = find_cases_above_threshold(root_directory, threshold)
    print(f"Number of cases with more than {threshold} DCM files: {count_above_threshold}")
    
    case_count = 0
    # print("Cases Above:")
    # for case, dcm_count in cases_above_threshold:
    #     print(f"{case}: {dcm_count} DCM files")
        
    # # 分割计算体积
    # tb_fibrosis = p
    for case, dcm_count in tqdm(cases_above_threshold):
        case_count += 1
        # if case_count == 1:
        #     continue
        # skip loop if the case already exists
        case_name = basename(case)
        case_dir = join(base_dir, case_name)
        

        
        os.makedirs(case_dir, exist_ok=True)
        csv_file_path = join(save_dir, case_name+'_pixel_check.csv')
        
        try:
            df_quantify = pd.read_csv(join(os.path.dirname(csv_file_path), 'quantification.csv'))
            if case_name in df_quantify['Case'].values:
                print(f"Skipping {case} as it already exists in quantification.csv")
                continue
        except FileNotFoundError:
            pass
        # if os.path.exists(case_dir) and glob.glob(os.path.join(case_dir, '*.png')):
        #     print(f"Skipping {case} as it already exists and contains PNG files")
        #     continue
        
        print(f"\nProcessing {case}: {dcm_count} DCM files")
        
        dcm_files = [os.path.join(case, file) for file in os.listdir(case) if file.endswith('.dcm')]

        volume_case = 0

        slices = [pydicom.read_file(s) for s in dcm_files]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        
        
        print(f'Case {case_name} has {len(slices)} slices')
        
        slices_second= slices[1:]
        slices_first = slices[0:-1]
        slice_distance = [(abs(s1.ImagePositionPatient[2] - s2.ImagePositionPatient[2])) for s1, s2 in zip(slices_first,slices_second)]
        print(f'Case {case_name} unique distance between slices: {np.unique(slice_distance)}')
        
        slice_plane = [s.PixelSpacing for s in slices_first]
        slice_thick = [s.SliceThickness for s in slices_first]
        slice_voxel = [xy[0]*xy[1]*z for xy, z in zip(slice_plane, slice_thick)]
        # image = np.stack([s.pixel_array for s in slices])
        # slice_np = [s.pixel_array for s in slices_first]
        error_log_path = join(base_dir, 'error_cases.txt')
        try:
            slices_HU = get_pixels_hu(slices)
        except Exception as e:
            print(f"Error reading {case}")
            with open(error_log_path, 'a') as f:
                f.write(f"Error reading {case}: {str(e)}\n")
            continue
        # slices_HU = [slices_HU[i] for i in range(slices_HU.shape[0])]
        masks_lung = mask.apply(slices_HU) # 0: background, 1: right-lung mask, 2: left-lung mask
        
        # get slices with lung mask bigger than 4 pixels
        # slices_to_process = [i for i in range(masks_lung.shape[0]) if np.sum(masks_lung[i] > 0) > 100] #this may result in break in consecutive slices
        first_slice = None
        last_slice = None
        # TODO: remove pixel threshold
        for i in range(masks_lung.shape[0]):
            if np.sum(masks_lung[i] > 0) > 400:
                if first_slice is None:
                    first_slice = i
                last_slice = i
        if first_slice is None:
            raise ValueError(f"No slice found with lung mask bigger than 400 pixels for case {basename(case)} in shape of 512")
        print(f'Case {case_name} has lung mask from slice {first_slice} to {last_slice}')
        
        # # visual check of lung mask
        # i = np.floor((first_slice+last_slice)/2).astype(int)
        # mask_zero = masks_lung[i] == 0
        # plt.imsave(f'{base_dir}/lung_mask_zero.png', mask_zero, cmap='gray')
        # mask_one = masks_lung[i] == 1
        # plt.imsave(f'{base_dir}/lung_mask_one.png', mask_one, cmap='gray')
        # mask_two = masks_lung[i] == 2
        # plt.imsave(f'{base_dir}/lung_mask_two.png', mask_two, cmap='gray')
        
    
        total_pixel = 0
        
        print("slice preprocessiong Started")
        
        for z in range(first_slice, last_slice):
            slice = slices_HU[z]
            voxel = slice_voxel[z] #slice with voxel info has less one slice
            mask_lung = masks_lung[z]

        
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
            binary_mask_lung = copy.deepcopy(mask_lung)
            binary_mask_lung[binary_mask_lung != 0] = 1
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
                
                np_slice_resized = np.array(img_slice_resized)
                pixel_num_lung = np.sum(np_slice_resized != 0)
            assert slice_masked.max() >= 0.9, f"slice_masked.max()={slice_masked.max()}"
            
            # np_slice_resized = np.array(img_slice_resized)
            # pixel_num_lung = np.sum(np_slice_resized != 0)
            
            
            data = {
                'Case': [case_name],
                'ID': [slice_name],
                'size': [img_slice_resized.size[0]],
                'voxel': [voxel],
            }
            # if pixel_num_lung > 0:
            data['pixel_num_lung'] = [pixel_num_lung]
            
            df = pd.DataFrame(data)
            df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)
        
        print(f'Case {basename(case)} processed and saved successfully')
        break
        
    print("All cases processed and saved successfully")         

