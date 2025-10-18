import os
import sys
sys.path.append("")
from src.diffae.useage import model_load
from utils.datasets import Dataset_diffae_osic
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

'''
    Load selected dataset for manipulation. Different selection between fibrosis and no_fibrosis.
    Use the csv file generated in encode.py to load the filenames and labels.
    CSV file path: experiments/diffae_usage/encoded_shuffled/bs_{BATCH_SIZE}_count{COUNT}/shuffled_filenames_labels.csv
    Label 0: no_fibrosis
    Label 1: fibrosis
    Label -1: no_fibrosis for manipulation
'''
BATCH_SIZE = 10
COUNT = 200
TYPE: str = 'PCA'  # 'TSNE' or 'PCA'
# MANIP_STRENGTH = [0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0]  # 1.0, 1.5, 2.0
MANIP_STRENGTH = [1.5]
LOAD_ROOT = f'experiments/diffae_usage/encoded_shuffled/bs_{BATCH_SIZE}_count{COUNT}'
SAVE_DIR = f'experiments/tsne/pca/bs_{BATCH_SIZE}_count{COUNT}'
os.makedirs(SAVE_DIR, exist_ok=True)
CLS_CHECK: bool = False  # Whether to do classification model check
CLS_CHECK_value: float = 0.7  # Classification score threshold [0.5, 0.6, 0.7, 0.8, 0.9]
WEIGHTED_TSNE: bool = False  # Whether to do weighted t-SNE visualization
# for CLS_CHECK_value in [0.5, 0.6, 0.7, 0.8, 0.9]:
# load filename list
file_list = os.path.join(LOAD_ROOT, 'shuffled_filenames_labels.csv')
import csv
filenames = {}
with open(file_list, 'r') as f:
    reader = csv.reader(f)
    next(reader, None)  # skip header row if present
    for row in reader:
        filenames[row[0]] = int(row[1])
logging.info(f"Total {len(filenames)} files to decode.")

filenames_label0 = {k: v for k, v in filenames.items() if v == 0}
filenames_label1 = {k: v for k, v in filenames.items() if v == 1}
filenames_label0_manip = {k: v for k, v in filenames.items() if v == -1}
logging.info(f"Total {len(filenames_label0)} files for no_fibrosis (label 0). \n                {len(filenames_label1)} files for fibrosis (label 1). \n                {len(filenames_label0_manip)} files for manipulation (label -1).")


'''
    Get encoded condition space for all data (including manipulated)
'''
for ms in MANIP_STRENGTH:    
    manip_dir = f'experiments/diffae_usage/encoded_shuffled/bs_{BATCH_SIZE}_count{COUNT}/manip_{ms}_nofib_fib'
    dataset_all = Dataset_diffae_osic(list(filenames.keys()), list(filenames.values()), manip_dir=manip_dir)
    dataset_all[500]
    dataloader_all = DataLoader(dataset_all, batch_size=BATCH_SIZE, shuffle=False)

    save_path = os.path.join(SAVE_DIR, f'cond_manip{ms}_bs{BATCH_SIZE}_all600.npy')
    tsne_path = save_path.replace('/pca/', '/tsne/')
    cached_path = next((path for path in (save_path, tsne_path) if os.path.isfile(path)), None)

    if cached_path:
        logging.info(f"Encoded condition space already exists at {save_path}, loading it directly.")
        cond = np.load(cached_path)
        
        device = 'cuda'
        _, model_cls = model_load(device, diff=False, cls=True)
        direction_class_1 = model_cls.direction_class_1.detach().cpu().numpy()
        logging.info("ISBI Classification Models loaded successfully.")
        
    else:
        logging.info(f"Encoded condition space not found at {save_path}, encoding now.")
        '''
            Load ISBI diff model and classification model
        '''
        device = 'cuda'
        model_diff, model_cls = model_load(device, diff=True, cls=True)
        direction_class_1 = model_cls.direction_class_1.detach().cpu().numpy()
        logging.info("ISBI Diffae and Classification Models loaded successfully.")

        '''
            Encode Images into condition space as cond
        '''
        def encode_imgs(model, dataloader, device):
            model.eval()
            all_cond = []
            all_labels = []
            with torch.no_grad():
                for imgs, labels, _ in tqdm(dataloader, total=len(dataloader)):
                    imgs = imgs.to(device)
                    cond = model.encode(imgs)
                    all_cond.append(cond.cpu().numpy())
                    all_labels.append(labels.numpy())
            all_cond = np.concatenate(all_cond)
            all_labels = np.concatenate(all_labels)
            return all_cond, all_labels

        cond, labels = encode_imgs(model_diff, dataloader_all, device)
        np.save(save_path, cond)
        logging.info(f"Encoded condition space saved to {save_path}")


    labels = np.array(list(filenames.values()))
    # classifier check
    list_score = []
    for img_cond, label in zip(cond, labels):
        img_cond = torch.tensor(img_cond, dtype=torch.float32)
        img_cond = img_cond.unsqueeze(0)  # add batch dimension
        img_cond = img_cond.to(device)
        outputs = model_cls(img_cond)
        pred = torch.softmax(outputs, dim=1)
        score = pred.data[:, 1]
        list_score.extend(score.tolist())
    list_score = np.array(list_score)
    logging.info(f"Classification scores computed for all samples.")

    if CLS_CHECK:
        cond_label0 = cond[(labels == 0) & (list_score < 1-CLS_CHECK_value)]
        cond_label1 = cond[(labels == 1) & (list_score > CLS_CHECK_value)]
        cond_label0_manip = cond[(labels == -1) & (list_score > CLS_CHECK_value)]

        labels = np.array([1]*len(cond_label1) + [0]*len(cond_label0) + [-1]*len(cond_label0_manip))
    else:
        cond_label0 = cond[labels==0]
        cond_label1 = cond[labels==1]
        cond_label0_manip = cond[labels==-1]

    logging.info(f"Encoded condition space: {cond.shape}, label0: {cond_label0.shape}, label1: {cond_label1.shape}, label0_manip: {cond_label0_manip.shape}")
    # cond = np.concatenate([cond_label1, cond_label0, cond_label0_manip], axis=0) if CLS_CHECK else cond
    
    # Binary case, only fibrosis and no_fibrosis
    cond = np.concatenate([cond_label1, cond_label0], axis=0)
    labels = np.array([1]*len(cond_label1) + [0]*len(cond_label0))

    '''
        TSNE Visualization
    '''
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    # TSNE_WEIGHT: bool = True  # Whether to weight by classification model weights
    cls_check_str = f'_cls{CLS_CHECK_value}' if CLS_CHECK else ''
    weighted_str = '_weighted' if WEIGHTED_TSNE else ''

    def tsne_and_plot(cond: np.ndarray, labels: np.ndarray, save_path: str, tsne_weight = None):
        """
        cond: (N, 512) condition vectors in the same order as labels
        labels: (N,) array-like of {1,0,-1}
        tsne_weight: optional 1D array of length 512 to scale features (broadcasted)
        """
        cond = np.asarray(cond)
        labels = np.asarray(labels)
        
        # Apply weighting from classification model if provided
        if tsne_weight is not None:
            w = np.asarray(tsne_weight)
            cond = cond * w
        
        mask1 = labels == 1
        mask0 = labels == 0
        maskm = labels == -1

        a1 = int(mask1.sum()); a2 = int(mask0.sum()); a3 = int(maskm.sum())
        N = labels.shape[0]
        logging.info(f'Fibrosis: {a1}, No_Fibrosis: {a2}, Manip: {a3}, Total: {N}')
        
        if any([a1 < 5, a2 < 5]):
            logging.error(f"Not enough samples for t-SNE. Label 1, 0, and -1 must each have at least 5 samples. a1={a1}, a2={a2}, a3={a3}")
            return
        
        # Larger datasets usually require a larger perplexity, normally [5,50]. 
        # A reasonable starting point might be around the square root of the number of data points
        if TYPE == 'TSNE':
            perp_assume = int(np.sqrt(a1+a2+a3)/5)*5
            perp_assume = max(5, min(perp_assume, 50))  # Clamp between 5 and 50
            
            tsne = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=perp_assume, random_state=20)
            data_tsne = tsne.fit_transform(cond)
            data_all = data_tsne
            logging.info(f"TSNE transformed shape: {data_all.shape}")
        elif TYPE == 'PCA':
            pca = PCA(n_components=2, random_state=20)
            data_pca = pca.fit_transform(cond)
            data_all = data_pca
            logging.info(f"PCA transformed shape: {data_all.shape}")
        
        # Plot the embedded data
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        
        ax.scatter(data_all[mask1, 0], data_all[mask1, 1], color='red', label=f'Fibrosis({a1})', s=10)
        ax.scatter(data_all[mask0, 0], data_all[mask0, 1], color='green', label=f'No_Fibrosis({a2})', s=10)
        ax.scatter(data_all[maskm, 0], data_all[maskm, 1], color='blue', label=f'Manipulated({a3})', s=10)

        title = f'No_Fib_Add(a={ms}){weighted_str}{cls_check_str}'
        ax.set_title(title)
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(save_path)
        # plt.savefig(os.path.join(SAVE_DIR, f'manip{ms}_bs{BATCH_SIZE}_all600{weighted_str}.png'))
        plt.close(fig)
    

    save_path = os.path.join(SAVE_DIR, 'imgs', f'{TYPE}_bs{BATCH_SIZE}_binary{weighted_str}{cls_check_str}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # tsne_and_plot(cond, labels, save_path=save_path, tsne_weight=direction_class_1 if WEIGHTED_TSNE else None)
    
    tsne_and_plot(cond, labels, save_path=save_path, tsne_weight=direction_class_1 if WEIGHTED_TSNE else None)
