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

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUM

'''
This script processes fibrosis images from the preprocessed OSIC dataset (size: 350x350).
The processed images will be resized to 256x256 and saved in the directory:
    data/OSIC/processed/fibrosis
'''

def get_workspace_dir():
    """Find the workspace directory by traversing up until a .git folder is found."""
    current_file_dir = Path(__file__).resolve().parent
    workspace_dir = current_file_dir
    while not (workspace_dir / ".git").exists() and workspace_dir != workspace_dir.parent:
        workspace_dir = workspace_dir.parent
    return str(workspace_dir)

def process_images():
    """Main function to process fibrosis images."""
    work_dir = get_workspace_dir()
    save_dir = Path(work_dir) / 'data/OSIC/processed/fibrosis'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    trainset_loader = prepare_dataset(bs=BATCH_SIZE, dataname='fibrosis', train_mode=False, exp=EXP_NAME)

    # Process images
    count = 0
    max_count = len(trainset_loader)

    for image_axial, labels, names in tqdm(islice(trainset_loader, max_count), total=max_count):
        count += 1

        # Validate labels
        if not torch.all(labels == 1):
            raise ValueError(f"Error: Expected all labels to be 1, but got different labels at count {count}")

        # Process each image in the batch
        for img_id, name in enumerate(names):
            flat_img_id = (count - 1) * BATCH_SIZE + img_id
            orig_img_path = Path(trainset_loader.dataset.img_folder) / name
            new_img_path = save_dir / name

            try:
                orig_img = Image.open(orig_img_path)
                resized_img = trans_fn.resize(orig_img.convert("RGB"), IMAGE_SIZE, InterpolationMode.LANCZOS)
                resized_img.save(new_img_path)
            except Exception as e:
                print(f"Error processing image {orig_img_path}: {e}")

if __name__ == "__main__":
    process_images()