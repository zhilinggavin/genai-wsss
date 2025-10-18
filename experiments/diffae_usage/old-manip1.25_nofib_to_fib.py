import copy

import os
gpu_num = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num


from torch.utils.data import Dataset
# from train_10montage import *
import torch
import numpy as np
import os
import random
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score
from torchvision.transforms import functional as trans_fn
import matplotlib.pyplot as plt
import cv2
import copy

import sys
sys.path.append("/media/NAS06/gavinyue/disentanglement")
import stylegan_codebase.fns_custom.fn_stylegan_oisc as fns
from fns_custom.dataset import SimpleDataset, get_transform
from fns_custom.fibmask import get_fib_mask, get_intersection_mask
from tqdm import tqdm
import wandb
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')



# %%

''' exp3_equal_ratio: ratio of fib: no_fib = 1:1 '''
expname = 'exp3_equal_ratio'
trainset_loader = fns.prepare_dataset(bs=8, dataname='no_fibrosis', train_mode=False, exp = expname)
cls_model_path = '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/classification_fibrosis/exp3_equal_ratio/model/model_loss_best_499.pt'

ms=1.25
img_save_folder = f'/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/fid/no_fibrosis/fid_dataset/fib_manip{ms:.2f}'
mask_save_folder = f'/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/fid/no_fibrosis/fid_dataset/fib_manip{ms:.2f}_masks_exp2'
mask_overlay_save_folder = f'/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/fid/no_fibrosis/fid_dataset/fib_manip{ms:.2f}_masks_overlay_exp2'
os.makedirs(mask_save_folder,exist_ok=True)
os.makedirs(img_save_folder,exist_ok=True)
os.makedirs(mask_overlay_save_folder,exist_ok=True)
experiment = wandb.init(project='fibrosis_cls', resume='allow', anonymous='must', name=f'orig_ms{ms:.2f}')
logging.info(f'''Starting training:
                exp_name:        {experiment.name}
                experiment_id:   {experiment.id}
                no_fib num: {len(trainset_loader.dataset)}
            ''')

# %%
device = "cuda"
# load classification model
model = nn.Linear(512, 2)
model = model.to(device)
# state_dict = torch.load('/media/NAS04/yyfang/prognostic_result/xai/counterfactual/Diffusion-Explainer/scripts_osic/result_exp/classification_fibrosis/test/model/model_loss_best.pt')['model_state_dict']
state_dict = torch.load(cls_model_path)['model_state_dict']

if any(key.startswith('module.') for key in state_dict.keys()):
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

    model.load_state_dict(state_dict)
    direction_class_0 = model.module.weight[0]
    direction_class_1 = model.module.weight[1]
else:
    model.load_state_dict(state_dict)
    direction_class_0 = model.weight[0]
    direction_class_1 = model.weight[1]

# load diffusion model as model_diff
model_diff, _ = fns.model_load(device, diff=True, cls=False)
model_diff.eval()

logging.info(f'''
classifer:  {cls_model_path}
model_diff: default
''')
# %%
# Manipulation along direction_class_1: fibrosis
count = 0
max_count = len(trainset_loader)
dataset_transform = get_transform()
from itertools import islice
with torch.no_grad():
    for image_axial, labels, names in tqdm(islice(trainset_loader, max_count), total=max_count):
        img_manip = []
        count += 1

        image_axial, labels = image_axial.to(device), labels.to(device)
        
        if not torch.all(labels == 0):
            raise ValueError(f"Error: Expected all labels to be 0, but got different labels at count {count}")
        
        image_diff = model_diff.encode(image_axial.to(device))
        outputs = model(image_diff)
        pred = F.softmax(outputs, dim=1).reshape(outputs.size()[0], -1)
        score_fib = pred.data[:, 1].cpu().numpy()
        
        xT = model_diff.encode_stochastic(image_axial.to(device), image_diff, T=250)
        
        # Manipulation
        add = ms * direction_class_1

        recon = model_diff.render(xT, image_diff + add, T=100)
        recon_show = (recon.permute(0,2,3,1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy() #range[0,1]
        orig = (image_axial.permute(0,2,3,1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        
        for img_id in range(len(recon_show)):
            falt_img_id = (count-1)*8 + img_id
            each_score = score_fib[img_id]
            recon_each = recon_show[img_id]
            I = Image.fromarray(recon_each)
            manip_save_name = f'orig{falt_img_id:04d}_manip{ms:.2f}.png'
            I.save(os.path.join(img_save_folder,manip_save_name))
            
            # fibrosis score for manipulation
            manip_img = dataset_transform(I)
            manip_cond = model_diff.encode(manip_img[None,:].to(device))
            outputs2 = model(manip_cond)
            pred2 = F.softmax(outputs2, dim=1)
            score_fib2 = pred2.data[:, 1].cpu().numpy()
            
            # get seg_mask
            orig_each = orig[img_id]
            orig_img1 = cv2.cvtColor(orig_each, cv2.COLOR_BGR2GRAY) if len(orig_each.shape) == 3 else orig_each
            manip_img1 = cv2.cvtColor(recon_each, cv2.COLOR_BGR2GRAY) if len(recon_each.shape) == 3 else recon_each
            
            output_image,seg_mask,num1,num2 = get_fib_mask(orig_img1, manip_img1, show=False, debug=False)
            
            seg_mask = Image.fromarray(seg_mask)
            seg_mask.save(os.path.join(mask_save_folder,manip_save_name.replace('.png','_mask.png')))
            
            output_image = Image.fromarray(output_image)
            output_image.save(os.path.join(mask_overlay_save_folder,manip_save_name.replace('.png','_mask_overlay.png')))

            experiment.log({
                            'img_id': falt_img_id,
                            'fib score': each_score,
                            'fib score manip': score_fib2[0],
                            'images': wandb.Image(orig_each),
                            'manip_images': wandb.Image(recon_each),
                            'masks': wandb.Image(seg_mask),
                            'masks_overlay': wandb.Image(output_image)
                        })
