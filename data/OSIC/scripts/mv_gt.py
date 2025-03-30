import pandas as pd
from tqdm import tqdm
from pathlib import Path
import shutil

def get_workspace_dir() -> str:
    """Find the workspace directory by traversing up until a .git folder is found."""
    current_file_dir = Path(__file__).resolve().parent
    workspace_dir = current_file_dir
    while not (workspace_dir / ".git").exists() and workspace_dir != workspace_dir.parent:
        workspace_dir = workspace_dir.parent
    return str(workspace_dir)

# Set work_dir, csv_path, img_dir, and save_dir
work_dir = Path(get_workspace_dir())
csv_path = work_dir / 'data/OSIC/origname_fibrosis.csv'

img_dir = Path('/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/fid/fibrosis/fid_dataset/orig_mask')
save_dir = work_dir / 'data/OSIC/processed/fibrosis_gt'
save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the save directory exists

# Load the CSV file for mapping original names
df = pd.read_csv(csv_path)
random_names = df['random_name'].tolist()
df.set_index('random_name', inplace=True)  # Set 'random_name' as the index for easier lookup

# Process each random name
for name in tqdm(random_names, desc="Processing files"):
    # Retrieve the corresponding 'orig_name'
    orig_name = df.at[name, 'orig_name']

    # Construct the source and destination paths
    src_path = img_dir / name.replace('.png', '_mask.png')
    dst_path = save_dir / orig_name.replace('.png', '_mask.png')
    
    # Check if the source file exists
    if not src_path.exists():
        raise FileNotFoundError(f"Source file {src_path} does not exist.")
    
    # Copy the file
    shutil.copy(src_path, dst_path)