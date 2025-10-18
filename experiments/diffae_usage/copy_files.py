import os
import shutil
import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv('experiments/diffae_usage/encoded_shuffled/bs_10_count200/shuffled_filenames_labels.csv')
    filenames = df['filename'].values
    labels = df['label'].values

    fib_names = filenames[labels == 1]
    print(f"Number of fibrosis images: {len(fib_names)}")
    
    load_dir = "data/OSIC/processed/fibrosis"
    save_dir = "experiments/diffae_usage/encoded_shuffled/bs_10_count200/fibrosis_copied"
    os.makedirs(save_dir, exist_ok=True)
    
    for name in fib_names:
        src_path = os.path.join(load_dir, name)
        dst_path = os.path.join(save_dir, name)
        shutil.copyfile(src_path, dst_path)