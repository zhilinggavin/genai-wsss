import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from itertools import islice
import torch
from fn_stylegan_osic import prepare_dataset

# Constants
BATCH_SIZE = 8
EXP_NAME = 'exp3_equal_ratio'
CSV_FILENAME = "origname_record_fibrosis.csv"

def get_workspace_dir():
    """Find the workspace directory by traversing up until a .git folder is found."""
    current_file_dir = Path(__file__).resolve().parent
    workspace_dir = current_file_dir
    while not (workspace_dir / ".git").exists() and workspace_dir != workspace_dir.parent:
        workspace_dir = workspace_dir.parent
    return workspace_dir

def main():
    """Main function to generate random names for fibrosis images."""
    # Set up directories
    work_dir = get_workspace_dir()
    save_dir = work_dir / 'data/OSIC/processed/fibrosis'
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_save_dir = work_dir / 'data/OSIC'

    # Prepare dataset
    try:
        trainset_loader = prepare_dataset(bs=BATCH_SIZE, dataname='fibrosis', train_mode=False, exp=EXP_NAME)
    except Exception as e:
        raise RuntimeError(f"Failed to prepare dataset: {e}")

    # Initialize data storage
    data = defaultdict(list)
    count = 0
    max_count = len(trainset_loader)

    # Process dataset
    for image_axial, labels, names in tqdm(islice(trainset_loader, max_count), total=max_count):
        count += 1

        # Validate labels
        if not torch.all(labels == 1):
            raise ValueError(f"Error: Expected all labels to be 1, but got different labels at count {count}")

        # Generate random names for each image
        for img_id, orig_name in enumerate(names):
            flat_img_id = (count - 1) * BATCH_SIZE + img_id
            random_name = f"orig{flat_img_id:05d}.png"
            data["orig_name"].append(orig_name)
            data["random_name"].append(random_name)

    # Save results to CSV
    df = pd.DataFrame(data)
    csv_path = csv_save_dir / CSV_FILENAME
    try:
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to {csv_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save CSV: {e}")

if __name__ == "__main__":
    main()