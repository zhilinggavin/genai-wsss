import os
from pathlib import Path
import torch
import numpy as np
from torchvision import transforms as transforms
from PIL import Image
from torchvision.transforms import functional as trans_fn
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from itertools import islice
from fn_stylegan_osic import prepare_dataset

# Constants
GPU_NUM = '0'
IMAGE_SIZE = 256
EXP_NAME = 'exp3_equal_ratio'
BATCH_SIZE = 8

# Set the dataset name
# to 'no_fibrosis' for processing non-fibrosis images
# or 'fibrosis' for processing fibrosis images
DATANAME = 'no_fibrosis' # or 'fibrosis'

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUM

'''
This script processes fibrosis images from the preprocessed OSIC dataset (size: 350x350).
The processed images will be resized to 256x256 and saved in the directory:
    data/OSIC/processed/no_fibrosis
'''

def get_workspace_dir():
    """Find the workspace directory by traversing up until a .git folder is found."""
    current_file_dir = Path(__file__).resolve().parent
    workspace_dir = current_file_dir
    while not (workspace_dir / ".git").exists() and workspace_dir != workspace_dir.parent:
        workspace_dir = workspace_dir.parent
    return str(workspace_dir)

def format_filename(filename: str) -> str:
    """
    Convert a filename from the format '82_fibrosis_40.png' to '082_fibrosis_040.png'.

    Args:
        filename (str): The original filename.

    Returns:
        str: The formatted filename with zero-padded numeric parts.
    """
    filename = Path(filename)

    # Extract the stem (filename without extension) and extension
    name = filename.stem  # e.g., '82_fibrosis_40'
    extension = filename.suffix  # e.g., '.png'

    # Split the filename into parts
    parts = name.split('_')  # Split into ['82', 'fibrosis', '40']
    if len(parts) == 3 and parts[0].isdigit() and parts[2].isdigit():
        # Zero-pad the first and last numeric parts
        parts[0] = f"{int(parts[0]):03d}"
        parts[2] = f"{int(parts[2]):03d}"

    # Reconstruct the new filename
    new_name = f"{'_'.join(parts)}{extension}"
    return new_name

def process_images():
    """Main function to process fibrosis images."""
    work_dir = get_workspace_dir()
    save_dir = Path(work_dir) / f'data/OSIC/processed/{DATANAME}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    trainset_loader = prepare_dataset(bs=BATCH_SIZE, dataname=DATANAME, train_mode=False, exp=EXP_NAME)

    # Process images
    count = 0
    max_count = len(trainset_loader)
    target_label = 1 if DATANAME == 'fibrosis' else 0

    for image_axial, labels, names in tqdm(islice(trainset_loader, max_count), total=max_count):
        count += 1

        # Validate labels
        if not torch.all(labels == target_label):
            raise ValueError(f"Error: Expected all labels to be 1, but got different labels at count {count}")

        # Process each image in the batch
        for img_id, name in enumerate(names):
            flat_img_id = (count - 1) * BATCH_SIZE + img_id
            orig_img_path = Path(trainset_loader.dataset.img_folder) / name
            new_img_path = save_dir / format_filename(name)

            try:
                orig_img = Image.open(orig_img_path)
                resized_img = trans_fn.resize(orig_img.convert("RGB"), IMAGE_SIZE, InterpolationMode.LANCZOS)
                resized_img.save(new_img_path)
            except Exception as e:
                print(f"Error processing image {orig_img_path}: {e}")

if __name__ == "__main__":
    process_images()