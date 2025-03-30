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
DATANAME = 'fibrosis' # or 'no_fibrosis'
CSV_FILENAME = f"origname_{DATANAME}.csv"

def get_workspace_dir():
    """Find the workspace directory by traversing up until a .git folder is found."""
    current_file_dir = Path(__file__).resolve().parent
    workspace_dir = current_file_dir
    while not (workspace_dir / ".git").exists() and workspace_dir != workspace_dir.parent:
        workspace_dir = workspace_dir.parent
    return workspace_dir

def format_filename(filename: str) -> str:
    """
    Convert a filename from the format '82_fibrosis_40.png' to '082_fibrosis_040.png'.
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

def main():
    """Main function to generate random names for fibrosis images."""
    # Set up directories
    work_dir = get_workspace_dir()
    csv_save_dir = work_dir / 'data/OSIC'

    # Prepare dataset
    try:
        trainset_loader = prepare_dataset(bs=BATCH_SIZE, dataname=DATANAME, train_mode=False, exp=EXP_NAME)
    except Exception as e:
        raise RuntimeError(f"Failed to prepare dataset: {e}")

    # Initialize data storage
    data = defaultdict(list)
    count = 0
    max_count = len(trainset_loader)
    target_label = 1 if DATANAME == 'fibrosis' else 0

    # Process dataset
    for image_axial, labels, names in tqdm(islice(trainset_loader, max_count), total=max_count):
        count += 1

        # Validate labels
        if not torch.all(labels == target_label):
            raise ValueError(f"Error: Expected all labels to be 1, but got different labels at count {count}")

        # Generate random names for each image
        for img_id, orig_name in enumerate(names):
            flat_img_id = (count - 1) * BATCH_SIZE + img_id
            random_name = f"orig{flat_img_id:05d}.png"
            orig_name = format_filename(orig_name)
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