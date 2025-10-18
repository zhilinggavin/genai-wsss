import torch
from pytorch_fid import fid_score
import os
import pandas as pd

def plot_fid_results(csv_path):
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    manip_names = df['manip_name'].tolist()
    manip_names = [name.replace('_nofib_fib', '') for name in manip_names]
    
    fid_scores = df['fid'].tolist()
    fid_scores = [round(score, 2) for score in fid_scores]
    plt.figure(figsize=(10, 6))
    plt.plot(manip_names, fid_scores, marker='o')
    plt.title('FID Score vs Manipulation Strength')
    plt.xlabel('Manipulation Strength')
    plt.ylabel('FID Score')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    # plt.show()
    plt.savefig(csv_path.replace('.csv', '.png'))

if __name__ == "__main__":
    MANIP_STRENGTH = [0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    NUM_WORKERS = os.cpu_count() or 1
    
    # real_imgs_path = "data/OSIC/processed/fibrosis"
    real_imgs_path = "experiments/diffae_usage/encoded_shuffled/bs_10_count200/fibrosis_copied"
    device = 'cuda'
    
    file_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(file_dir, 'fid_results_bs10_count200.csv')
    
    for ms in MANIP_STRENGTH:
        generated_imgs_path = f"experiments/diffae_usage/encoded_shuffled/bs_10_count200/manip_{ms}_nofib_fib"
        imgs_paths = [real_imgs_path, generated_imgs_path]

        fid_value = fid_score.calculate_fid_given_paths(imgs_paths, batch_size=10, device=device, dims=2048, num_workers=NUM_WORKERS)
        print(f"FID Score: {fid_value}")
        
        df = pd.DataFrame({
            'manip_name': [f"{os.path.basename(generated_imgs_path)}"],
            'fid': [fid_value]
            })

        df.to_csv(save_path, index=False, mode='a', header=not os.path.exists(save_path))
    
    plot_fid_results(save_path)