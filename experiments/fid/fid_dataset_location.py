import glob
import shutil
import os
from tqdm import tqdm

# Define the source and destination directories
source_dir = '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/fid/no_fibrosis/exp3_equal_ratio'


# Define the pattern to match files
# manip_pattern = 'manip0.75'
orig_pattern = 'orig'

for i in range(1,13):
    ms = i * 0.25
    if ms == 0.75:
        continue
    manip_pattern = f'manip{ms:.2f}'


    pattern = f'{orig_pattern}*_{manip_pattern}.jpg'
    # Construct the full pattern with the source directory
    full_pattern = os.path.join(source_dir, pattern)

    # Find all files in the source directory matching the pattern
    matching_files = glob.glob(full_pattern)

    tmp_folder = os.path.join(os.path.split(source_dir)[0],'fid_dataset')
    destination_dir = os.path.join(tmp_folder, manip_pattern)
    os.makedirs(destination_dir, exist_ok=True)

    # Copy each matching file to the destination directory
    for file_path in tqdm(matching_files):
        # Determine the filename from the file_path
        filename = os.path.basename(file_path)
        # Construct the destination file path
        destination_path = os.path.join(destination_dir, filename)
        # Copy the file
        shutil.copy(file_path, destination_path)

    print(f"Copied {len(matching_files)} files to {destination_dir}")

# '''
# Get the original images
# '''
# pattern = 'orig*.jpg'
# full_pattern = os.path.join(source_dir, pattern)
# all_files = glob.glob(full_pattern)
# matching_files = [file for file in all_files if '_manip' not in os.path.basename(file)]
# matching_files.sort()
# tmp_folder = os.path.join(os.path.split(source_dir)[0], 'fid_dataset')
# destination_dir = os.path.join(tmp_folder, "orig")
# os.makedirs(destination_dir, exist_ok=True)

# for file_path in tqdm(matching_files):
#     filename = os.path.basename(file_path)
#     destination_path = os.path.join(destination_dir, filename)
#     shutil.copy(file_path, destination_path)

# print(f"Copied {len(matching_files)} files to {destination_dir}")