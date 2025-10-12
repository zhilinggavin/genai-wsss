# %%
'''
TSNE visualization of original data and manipulated data
Date: 2024-07-11
'''

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
# import cv2
from PIL import Image



# Visualise Mask
def img_show(output,resize=True):
    out = (output.permute(0,2,3,1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    if resize:
        out = cv2.resize(out[0], (256,256))
    else:
        out = out[0]
    return out

def ShowMask_mod(input,mask,alpha=0.5,anno=None):
    if mask.dtype == np.float32:
        mask = (mask * 255).astype(np.uint8)
    assert mask.dtype == np.uint8, "Input tensor is not of type int8"

    if input.shape[-1] > 4:
        input=img_show(input,resize=True)

    if input.dtype == np.float32:
        input = (input * 255).astype(np.uint8) 


    img2 = Image.fromarray(input).convert("RGBA")

        
    tmp=np.zeros([256,256,3]).astype('uint8')
    tmp[:,:,1]=mask
    if anno is not None:
        if anno.shape == tmp.shape[:2]:
            tmp[:,:,0] = anno
            tmp[:,:,1] = (mask / 2).astype(np.uint8)
        else:
            print("Error: Shapes of anno and tmp do not match.")
    grad_mask=Image.fromarray(tmp).resize((256,256)).convert("RGBA")
    blend_img = Image.blend(img2, grad_mask, alpha)
    return blend_img

"""
    Generate mask for osic
"""
class OsicMask:
    def __init__(self):
        # self.load_names = []
        # self.anno_names = []
        # self.all_orig_img = []
        # self.labels = []
        # self.names = []
        print('OsicMask is initiated. To be modified')
    
    def setup_osic(self, trainset_loader, device, max_count=30, save_encode = False, exp = None):
        self.load_names = []
        self.anno_names = []
        self.all_orig_img = []
        self.labels = []
        self.names = []
        if save_encode:
            # import torch.nn as nn
            # import sys
            # sys.path.append("/media/NAS06/gavinyue/disentanglement/scripts_segmentation")
            
            # # load classification model
            # model = nn.Linear(512, 2)
            # model = model.to(device)
            # state_dict = torch.load('/media/NAS04/yyfang/prognostic_result/xai/counterfactual/Diffusion-Explainer/scripts_osic/result_exp/classification_fibrosis/test/model/model_loss_best.pt')['model_state_dict']
            # model.load_state_dict(state_dict)

            # self.direction_class_0 = model.weight[0]
            # self.direction_class_1 = model.weight[1]

            # # load diffusion model
            # from templates import ffhq256_autoenc
            # # from templates_cls import *
            # # from experiment_classifier import ClsModel
            # from experiment import LitModel
            # conf = ffhq256_autoenc()
            # model_diff = LitModel(conf)
            # conf.name = 'test_osic256_cluster_mxl_round2'
            # state = torch.load(f'/media/NAS04/yyfang/prognostic_result/xai/counterfactual/Diffusion-Explainer/checkpoints/{conf.name}/last.ckpt', map_location='cpu')
            # model_diff.load_state_dict(state['state_dict'], strict=False)
            # model_diff.ema_model.eval()
            # model_diff.ema_model.to(device)
            
            
            model_diff,model = model_load(device, diff=True, cls=False)
            self.img_cond = []
            
        count = 0
        for image_axial, labels, names in tqdm(trainset_loader,total = max_count):
            image_axial = image_axial.to(device)
            # labels = labels.to(device)
            count += 1
            if save_encode:
                cond = model_diff.encode(image_axial.to(device))
                self.img_cond.append(cond.cpu().numpy())
            
            tmp = f"{names[0].split('.')[0]}_count{count}.npy"
            anno_names = [name.split('.')[0] + '_anno.png' for name in names]
            self.load_names.append(tmp)
            self.anno_names.append(anno_names)
            self.labels.append(labels)
            self.names.append(names)
            
            orig_img_show = ((image_axial + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
            self.all_orig_img.append(orig_img_show)
            
            if count >= max_count:
                self.all_orig_img = np.stack(self.all_orig_img, axis=0)
                self.labels = np.stack(self.labels, axis=0)
                self.dataname = 'fibrosis' if self.labels[0][0] == 1 else 'no_fibrosis'
                print('data loaded: ',self.dataname)
                print(f'len(load_names): {len(self.load_names)}, all_orig_img.shape: {self.all_orig_img.shape}')

                if save_encode:
                    self.img_cond = np.stack(self.img_cond, axis=0)
                    if exp is None:
                        tmp = f'/media/NAS06/gavinyue/disentanglement/scripts_segmentation/npy/osic/manip_shuffled/{self.dataname}/cond/recon'
                    else:
                        tmp = f'/media/NAS06/gavinyue/disentanglement/scripts_segmentation/npy/osic/manip_shuffled/{self.dataname}/{exp}/cond/recon'
                    os.makedirs(tmp,exist_ok=True)
                    # Check if the file exists
                    if os.path.exists(f'{tmp}/count{count}.npy'):
                        # If the file exists, raise an error
                        raise FileNotFoundError(f"The file '{tmp}/count{count}.npy' already exists.")
                    
                    np.save(f'{tmp}/count{count}.npy',self.img_cond)
                    print(f'{tmp}/count{count}.npy Saved!')
                else:
                    if exp is None:
                        self.img_cond = np.load(f'/media/NAS06/gavinyue/disentanglement/scripts_segmentation/npy/osic/manip_shuffled/{self.dataname}/cond/recon/count{count}.npy')
                    else:
                        try:
                            self.img_cond = np.load(f'/media/NAS06/gavinyue/disentanglement/scripts_segmentation/npy/osic/manip_shuffled/{self.dataname}/{exp}/cond/recon/count{count}.npy')
                        except FileNotFoundError:
                            print('no img_cond file loaded')  
                print(f'img_cond.shape: {self.img_cond.shape}\n')
                break


    
    def generate_mask(self, img_no_symptom, img_heavier_symptom,sigma=0,k=5):
        # Convert images to grayscale
        # gray_no_symptom = cv2.cvtColor(img_no_symptom, cv2.COLOR_BGR2GRAY)
        # gray_heavier_symptom = cv2.cvtColor(img_heavier_symptom, cv2.COLOR_BGR2GRAY)
        
        if img_no_symptom.shape[-1]==3:
            img_no_symptom = img_no_symptom[:,:,0]
            img_heavier_symptom = img_heavier_symptom[:,:,0]
        if img_no_symptom.dtype == np.float32:
            img_no_symptom = (img_no_symptom * 255).astype(np.uint8)
        if img_heavier_symptom.dtype == np.float32:
            img_heavier_symptom = (img_heavier_symptom * 255).astype(np.uint8)
                    
        # Compute the absolute difference between the blurred images
        diff = cv2.absdiff(img_no_symptom, img_heavier_symptom)

        # TODO: this is new code to get the intersection mask using contours
        combined_mask = self.get_contour_masks(img_no_symptom, img_heavier_symptom)
        diff = cv2.bitwise_and(diff, diff, mask=combined_mask)
        # TODO: new code end here
        
        # # Apply initial threshold to remove values lower than 99.5% 'threshold = np.percentile(flattened_image, percentage)'
        # percentile = np.percentile(diff, 99.9)
        # # _, diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_TOZERO)
        # normalise to [0,255] Useless!
        diff = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Apply Gaussian Blur to reduce noise 
        diff = cv2.GaussianBlur(diff, (5, 5), sigma)
        # Apply Otsu's thresholding to get a binary mask
        _, mask1 = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Apply morphological operations to clean up the mask
        kernel = np.ones((k, k), np.uint8)
        mask2 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel) # Smaller is better for keeping details. Opening operation to remove noise
        mask3 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel) # Larger is better for large gaps. Closing operation to fill holes

        return diff,mask1,mask2,mask3
    
    def get_diff_masks(self,img1, img2):
        assert len(img1.shape) == 2 and len(img2.shape) == 2, "Input images must be grayscale" 
                    
        # Compute the absolute difference between the blurred images
        diff = cv2.absdiff(img1, img2)

        # Remove shape artefactes (hollow and decreased edges)
        combined_mask = self.get_contour_masks(img1, img2)
        diff = cv2.bitwise_and(diff, diff, mask=combined_mask)

        # # normalise to [0,255] Useless!
        # diff = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Apply Gaussian Blur to reduce noise
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        # Apply Otsu's thresholding to get a binary mask
        _, diff_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return diff_mask
        

    # TODO: updated thresholding method, increased gaussian blur kernel size
    def get_contour_masks(self,img1, img2):
        # Convert images to grayscale and apply Gaussian blur
        gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
        gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
        gray_img1 = cv2.GaussianBlur(gray_img1, (7, 7), 0)
        gray_img2 = cv2.GaussianBlur(gray_img2, (7, 7), 0)

        # Create a difference map
        difference_map = cv2.absdiff(gray_img1, gray_img2)

        # Apply thresholding to get binary images
        # _, thresh_img1 = cv2.threshold(gray_img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # _, thresh_img2 = cv2.threshold(gray_img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh_img2 = cv2.threshold(gray_img2, 30, 255, cv2.THRESH_BINARY)
        # Create masks from contours
        mask_img1 = np.zeros_like(gray_img1)
        mask_img2 = np.zeros_like(gray_img2)

        # Find contours for the whole image
        contours_img1, _ = cv2.findContours(gray_img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_img2, _ = cv2.findContours(thresh_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours on the original images
        contour_img1 = cv2.drawContours(img1.copy(), contours_img1, -1, (0, 255, 0), thickness=cv2.FILLED)  # Green contours
        contour_img2 = cv2.drawContours(img2.copy(), contours_img2, -1, (0, 255, 0), thickness=cv2.FILLED)  # Green contours
        contour_img1 = cv2.cvtColor(contour_img1, cv2.COLOR_BGR2RGB)
        contour_img2 = cv2.cvtColor(contour_img2, cv2.COLOR_BGR2RGB)
        # Draw contours on masks
        cv2.drawContours(mask_img1, contours_img1, -1, color=255, thickness=cv2.FILLED)
        cv2.drawContours(mask_img2, contours_img2, -1, color=255, thickness=cv2.FILLED)

        # Apply the masks to the difference map
        combined_mask = cv2.bitwise_and(mask_img1, mask_img2)
        masked_difference_map = cv2.bitwise_and(difference_map, difference_map, mask=combined_mask)
        return combined_mask    
    
    
    # mask test per single image
    def save_mask_each(self,sigma,k_range,load_folder,save_folder,test_show=False):
        for count in tqdm(range(0,len(self.load_names))):
            tmp = f'{load_folder}/{self.load_names[count]}'
            img_manip = np.load(tmp)
            print(f'loaded {tmp}')
            orig_img_show = self.all_orig_img[count]

            for img_id in range(len(img_manip)):
            # for sigma in np.arange(0, 1, 0.1):
                for k in k_range:
                    fig,axis = plt.subplots(6,5,figsize=(4*5,4*6),dpi=150)
                    names = [f'recon (img_id {img_id})','manip_0.75','manip_1.5','manip_2.25','manip_3']
                    names_mask = [f'diff (Blurred, sigma={sigma})',"Otsu's thresholding",f'morphological (kernel size {k})',f'Closed Mask (kernel size {k})']
                    fz = 14

                    for i in range(img_manip.shape[1]):
                        orig_img_each = orig_img_show[img_id]
                        img_manip_each = img_manip[img_id,i]
                            
                        # show manipulations in first row
                        axis[0,i].imshow(img_manip_each)
                        axis[0,i].set_title(names[i],fontsize=fz)
                        axis[0,i].axis('off')
                        
                        if i > 0:
                            masks = list(self.generate_mask(orig_img_each, img_manip_each, sigma, k))  # Convert tuple to list
                            for j in range(len(masks)):
                                axis[j+1,0].imshow(img_manip[img_id,0])
                                axis[j+1,0].set_title(names[0],fontsize=fz)
                                axis[j+1,0].axis('off')
                                axis[j+1,i].imshow(masks[j],cmap='hot')
                                axis[j+1,i].set_title(names_mask[j],fontsize=fz)
                                axis[j+1,i].axis('off')
                                
                            closed_mask = masks[-1]
                            if self.labels[count][img_id] == 0:
                                img_heavier_symptom = img_manip_each
                            else:
                                img_heavier_symptom = img_manip[img_id,0] # recon image
                            blend_img=ShowMask_mod(img_heavier_symptom,closed_mask,0.3)
                            
                            axis[len(masks)+1, i].imshow(blend_img)
                            axis[len(masks)+1, i].set_title('blend (stronger fibrosis)', fontsize=fz)
                            axis[len(masks)+1, i].axis('off')
                            # add recon image in last row
                            axis[len(masks)+1,0].imshow(img_manip[img_id,0])
                            axis[len(masks)+1,0].set_title(names[0],fontsize=fz)
                            axis[len(masks)+1,0].axis('off')
                    plt.tight_layout()
                    if test_show:
                        print(f'count: {count+1}, img_id: {img_id}, sigma: {sigma:.1f}, kernel: {k}')
                        return
                    else:
                        os.makedirs(save_folder, exist_ok=True)
                        plt.savefig(f'{save_folder}/count{count+1}_id{img_id}_sigma{sigma:.1f}_kernel{k}.jpg')
                        plt.close('all')
# # Visulisation of 8 images with masks
    def save_mask_all(self,sigma,k_range,load_folder,save_folder,test_show=False,manip_select=[9]):
        for manip_id in manip_select:
            ms = manip_id*3/4
            for count in tqdm(range(0,len(self.load_names))):
                sigma = 0
                tmp = f'{load_folder}/{self.load_names[count]}'
                img_manip = np.load(tmp)
                # print(img_manip.shape)
                orig_img_show = self.all_orig_img[count]

                for k in k_range:        
                    fig,axis = plt.subplots(8,8,figsize=(4*8,4*8),dpi=100)
                    names_mask = [f'diff (Blurred, sigma={sigma})',"Otsu's thresholding",f'morphological (kernel size {k})',f'Closed Mask (kernel size {k})']
                    fz = 14

                    # for i in range(len(img_manip)):
                    for img_id in range(len(img_manip)):
                        i = img_id
                        orig_img_each = orig_img_show[img_id]
                        img_manip_each = img_manip[img_id,manip_id]

                        masks = self.generate_mask(orig_img_each,img_manip_each,sigma,k)
                        closed_mask = masks[-1]
                        if self.labels[count][img_id] == 0:
                            img_heavier_symptom = img_manip_each
                            # img_folder = '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/slice_select/no_fibrosis'
                            img_folder = '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/no_fibrosis/'
                        else:
                            img_heavier_symptom = orig_img_each
                            # img_folder = '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/slice_select/fibrosis'
                            img_folder = '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/fibrosis/'
                        try:
                            img_anno = Image.open(f'{img_folder}/{self.anno_names[count][img_id]}').resize((256,256))
                            img_anno = np.array(img_anno)
                        except FileNotFoundError:
                            img_anno = np.zeros((256,256))
                        blend_img=ShowMask_mod(img_heavier_symptom,closed_mask,0.2)
                        # blend_img_anno = ShowMask_mod(img_heavier_symptom,closed_mask,0.35,anno=img_anno)
                        blend_img_anno = ShowMask_mod(img_heavier_symptom,closed_mask,0.5,anno=img_anno)
                        
                        # show manipulations
                        axis[0,i].imshow(orig_img_each)
                        axis[0,i].set_title(f'orig_img_id {img_id}',fontsize=fz)
                        axis[0,i].axis('off')            
                        axis[1,i].imshow(img_manip_each)
                        axis[1,i].set_title(f'manip_{ms}',fontsize=fz)
                        axis[1,i].axis('off')
                        for j in range(len(masks)):
                            axis[j+2,i].imshow(masks[j],cmap='hot')
                            axis[j+2,i].set_title(names_mask[j],fontsize=fz)
                            axis[j+2,i].axis('off')
                        axis[j+3,i].imshow(blend_img)
                        axis[j+3,i].set_title('blend (stronger fibrosis)',fontsize=fz)
                        axis[j+3,i].axis('off')
                        axis[j+4,i].imshow(blend_img_anno)
                        axis[j+4,i].set_title('blend_img_anno',fontsize=fz)
                        axis[j+4,i].axis('off')

                    plt.tight_layout()

                    if test_show:
                        print(f'count: {count+1}, load: {self.load_names[count]}, manip: {ms}, sigma: {sigma:.1f}, kernel: {k}')
                        return
                    else:
                        os.makedirs(save_folder, exist_ok=True)
                        plt.savefig(f'{save_folder}/count{count+1}_manip{ms}_sigma{sigma:.1f}_kernel{k}.jpg')
                        plt.close('all')
                    

import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as trans_fn

def setup_seed(randomSeed):
    torch.manual_seed(randomSeed)
    torch.cuda.manual_seed_all(randomSeed)
    np.random.seed(randomSeed)
    random.seed(randomSeed)
    torch.backends.cudnn.deterministic = True

class Dataset_fibrosis(Dataset):
    def __init__(self, x, y):
        self.images = x
        self.labels = y
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index):
        _label = self.labels[index]
        # img_folder = '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/slice_select/fibrosis/' if _label == 1 else '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/slice_select/no_fibrosis/'
        
        # Gavin dataset: more data
        
        self.img_folder = '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/fibrosis/' if _label == 1 else '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/no_fibrosis/'
        # img_folder = '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/fibrosis_selected/' if _label == 1 else '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/no_fibrosis_selected/'
        
        
        try:
            I = Image.open(self.img_folder + self.images[index])
        except FileNotFoundError:
            self.img_folder = '/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/no_fibrosis_covid/'
            I = Image.open(self.img_folder + self.images[index])
        from torchvision.transforms import InterpolationMode
        I_resize = trans_fn.resize(I.convert("RGB"), 256, InterpolationMode.LANCZOS)
        _img = self.transform(I_resize)
        return _img, _label, self.images[index]

    def __len__(self):
        return len(self.images)

def prepare_dataset(bs, dataname='fibrosis', train_mode=False, exp=None):
    """
    Performs an action based on the given option.

    Parameters:
    dataname (str):
                  - 'fibrosis': Only use fibrosis data for train and test.
                  - 'no_fibrosis': Only use no_fibrosis data for train and test.
                  - 'all_data': Use both fibrosis and no_fibrosis data for train and test.
    
    Returns:
    trainset_loader
    """
    setup_seed(20)
    # slice_fibrosis = [i for i in os.listdir('/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/slice_select/fibrosis/') if 'anno' not in i]
    # slice_nofibrosis = [i for i in os.listdir('/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/slice_select/no_fibrosis/') if 'anno' not in i]
    
    # Gavin dataset: more data
    if exp is None:
        slice_fibrosis = [i for i in os.listdir('/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/fibrosis/') if 'anno' not in i]
        slice_nofibrosis = [i for i in os.listdir('/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/no_fibrosis/') if 'anno' not in i]
    elif 'exp2' in exp:
            slice_fibrosis = [i for i in os.listdir('/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/fibrosis/') if 'anno' not in i]
            slice_nofibrosis = [i for i in os.listdir('/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/no_fibrosis/') if 'anno' not in i]
            if dataname == 'all_data':
                slice_nofibrosis_covid = [i for i in os.listdir('/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/no_fibrosis_covid/')]
                difference = len(slice_fibrosis) - len(slice_nofibrosis)
                slice_nofibrosis = slice_nofibrosis + slice_nofibrosis_covid[:difference]
            else:
                print('slice_nofibrosis is from no_fibrosis data, no covid data')
    
    elif 'exp3' in exp:
            slice_fibrosis = [i for i in os.listdir('/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/fibrosis_selected/') if 'anno' not in i]
            slice_nofibrosis = [i for i in os.listdir('/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/no_fibrosis_selected/') if 'anno' not in i]
            if dataname == 'all_data':
                slice_nofibrosis_covid = [i for i in os.listdir('/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/no_fibrosis_covid/')]
                difference = len(slice_fibrosis) - len(slice_nofibrosis)
                slice_nofibrosis = slice_nofibrosis + slice_nofibrosis_covid[:difference]
            elif dataname == 'no_fibrosis_covid':
                slice_nofibrosis_covid = [i for i in os.listdir('/media/NAS04/yyfang/prognostic_result/dataset/data_fibrosis/gavin/slice_select/no_fibrosis_covid/')]
                difference = len(slice_fibrosis) - len(slice_nofibrosis)
                slice_nofibrosis = slice_nofibrosis_covid[:difference]
            else:
                print('slice_nofibrosis is from no_fibrosis data, no covid data')
    
    random.shuffle(slice_fibrosis)
    random.shuffle(slice_nofibrosis)

    if dataname == 'fibrosis':
        slices = slice_fibrosis
    elif dataname == 'no_fibrosis' or dataname == 'no_fibrosis_covid':
        slices = slice_nofibrosis
    elif dataname == 'all_data':
        print(f'Total Slices = slice_fibrosis({len(slice_fibrosis)}) + slice_nofibrosis({len(slice_nofibrosis)})')
    
    if dataname == 'all_data':
        train_size1 = int(len(slice_fibrosis) * 4 / 5)
        train_size2 = int(len(slice_nofibrosis) * 4 / 5)
        x_train = slice_fibrosis[:train_size1] + slice_nofibrosis[:train_size2]
        x_test = slice_fibrosis[train_size1:] + slice_nofibrosis[train_size2:]
        y_train = np.ones(train_size1).tolist() + np.zeros(train_size2).tolist()
        y_test = np.ones(len(slice_fibrosis[train_size1:])).tolist() + np.zeros(len(slice_nofibrosis[train_size2:])).tolist()
        print(f'x_train({len(x_train)}) is from 4/5 of fibrosis and no_fibrosis data')
    else:
        train_size = int(len(slices) * 4 / 5)  # 4:1 ratio for train:test
        x_train, x_test = slices[:train_size], slices[train_size:]
        if dataname == 'fibrosis':
            y_train = np.ones(len(x_train)).tolist()
            y_test = np.ones(len(x_test)).tolist()
        else:
            y_train = np.zeros(len(x_train)).tolist()
            y_test = np.zeros(len(x_test)).tolist()
    
    if train_mode:
        trainset = Dataset_fibrosis(x_train, y_train)
        testset  = Dataset_fibrosis(x_test, y_test)
        trainset_loader = DataLoader(trainset, batch_size=bs, shuffle=True)
        testset_loader = DataLoader(testset, batch_size=bs, shuffle=False)
        return trainset_loader, testset_loader
    else:
        trainset = Dataset_fibrosis(x_train+x_test, y_train+y_test)
        trainset_loader = DataLoader(trainset, batch_size=bs, shuffle=False)
        print(f'{dataname} data loaded. prepare_dataset Done!')
        return trainset_loader

def model_load(device, diff=True, cls=True):
    '''
    Load OLD diff model and classification model, trained by Yingying Fang.
    '''
    import torch.nn as nn
    import sys
    sys.path.append("/media/NAS06/gavinyue/disentanglement/scripts_segmentation")
    model_diff = model = None
    if diff:
        from templates import ffhq256_autoenc #type: ignore
        from experiment import LitModel #type: ignore
        conf = ffhq256_autoenc()
        model_diff = LitModel(conf)
        conf.name = 'test_osic256_cluster_mxl_round2'
        state = torch.load(f'/media/NAS04/yyfang/prognostic_result/xai/counterfactual/Diffusion-Explainer/checkpoints/{conf.name}/last.ckpt', map_location='cpu')
        model_diff.load_state_dict(state['state_dict'], strict=False)
        model_diff.ema_model.eval()
        model_diff.ema_model.to(device)
        print('OLD model diff loaded')
    if cls:
        # load classification model
        model = nn.Linear(512, 2)
        model = model.to(device)
        state_dict = torch.load('/media/NAS04/yyfang/prognostic_result/xai/counterfactual/Diffusion-Explainer/scripts_osic/result_exp/classification_fibrosis/test/model/model_loss_best.pt')['model_state_dict']
        model.load_state_dict(state_dict)
        print('OLD classification model loaded')

        # self.direction_class_0 = model.weight[0].detach().cpu().numpy()
        # self.direction_class_1 = model.weight[1].detach().cpu().numpy()
    return model_diff, model


def load_encode_img(device='cuda',dataname='fibrosis',max_count=30,manip_idx=4,cls=False,exp=None):
    # model_diff,model = model_load(device, diff=True, cls=cls)
    if dataname == 'fibrosis':
        data_folder = '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/npy/osic/manip_shuffled/fibrosis'
    elif dataname == 'no_fibrosis':    
        data_folder = '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/npy/osic/manip_shuffled/no_fibrosis'
    elif dataname == 'no_fibrosis_minus': 
        data_folder = '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/npy/osic/manip_shuffled/no_fibrosis_minus'
    else:
        print('No dataname exists!')
        return
    
    if exp is None:
        model_diff,model = model_load(device, diff=True, cls=cls)
        print(f'no exp: data_folder = {data_folder}')
    elif 'exp3' in exp:
        data_folder = os.path.join(data_folder,exp)
        print(f'exp3: data_folder = {data_folder}')
        model_diff,model = model_load(device, diff=True, cls=False)
        if cls:
            model = nn.Linear(512, 2)
            model = model.to(device)
            cls_model_path = '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/result_exp/classification_fibrosis/exp3_equal_ratio/model/model_loss_best_499.pt'
            state_dict = torch.load(cls_model_path)['model_state_dict']
            model = nn.DataParallel(model)
            model.load_state_dict(state_dict)
        
        
    slices = [i for i in os.listdir(data_folder) if '.npy'  in i]
    lables = np.zeros(len(slices)).tolist() if 'no' in dataname else np.ones(len(slices)).tolist()
    trainset = Dataset_npy(slices, lables, data_folder)
    img_cond = []
    
    for count in range(0,max_count):
        img, label, img_name = trainset[count]#load dataset to tensor
        img = img[:,manip_idx]
        cond = model_diff.encode(img.to(device))
        img_cond.append(cond.cpu().numpy())
    img_cond = np.stack(img_cond, axis=0)

    return img_cond, model
       
        
class Dataset_npy(Dataset):
    def __init__(self, x, y, data_folder):
        self.images = x
        self.labels = y
        self.data_folder = data_folder
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __getitem__(self,count):
        _label = self.labels[count]
        # data_folder = '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/npy/osic/manip_shuffled/fibrosis/' if _label == 1 else '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/npy/osic/manip_shuffled/no_fibrosis/'
        I = np.load(f'{self.data_folder}/{self.images[count]}')
        # I = Image.fromarray(I)
        # from torchvision.transforms import InterpolationMode
        # I_resize = trans_fn.resize(I.convert("RGB"), 256, InterpolationMode.LANCZOS)
        
        img1 = [];img2 = []
        for i in range(I.shape[0]):
            img1 = [self.transform(I[i,j]) for j in range(I.shape[1])]
            img2.append(torch.stack(img1))

        _img = torch.stack(img2)
        return _img, _label, self.images[count]

    def __len__(self):
        return len(self.images)

def load_orig_data_tsne(device,max_count=30,exp='exp3_equal_ratio'):
    # max_count#@params[30,236,len(trainset_loader)-1] 
    M = OsicMask()
    trainset_loader =prepare_dataset(bs=8, dataname='fibrosis',exp=exp)
    if max_count == 'alldata':
        count = len(trainset_loader)-1
    else:
        count = max_count
        print(f'{count}')
    # M.setup_osic(trainset_loader, device, max_count=count, model_load = False, save_encode = False)
    # cond_fib = M.img_cond
    cond_fib = np.load(f'/media/NAS06/gavinyue/disentanglement/scripts_segmentation/npy/osic/manip_shuffled/fibrosis/{exp}/cond/recon/count{count}.npy')  

    trainset_loader =prepare_dataset(bs=8, dataname='no_fibrosis',exp=exp)
    if max_count == 'alldata':
        count = len(trainset_loader)-1
    else:
        count = max_count
    # M.setup_osic(trainset_loader, device, max_count=count, model_load = False, save_encode = False)
    # cond_nofib = M.img_cond
    cond_nofib = np.load(f'/media/NAS06/gavinyue/disentanglement/scripts_segmentation/npy/osic/manip_shuffled/no_fibrosis/{exp}/cond/recon/count{count}.npy')

    cond_fib = np.concatenate(cond_fib)
    cond_nofib = np.concatenate(cond_nofib)
    return cond_fib, cond_nofib
    
class SimpleDataset(Dataset):
    def __init__(self, dir1, dir2=None):
        self.dir1 = dir1
        self.dir2 = dir2
        self.files1 = sorted(os.listdir(dir1))
        self.files2 = sorted(os.listdir(dir2)) if dir2 else None

    def __len__(self):
        return len(self.files1)

    def __getitem__(self, idx):
        img1 = Image.open(os.path.join(self.dir1, self.files1[idx]))
        if self.files2:
            img2 = Image.open(os.path.join(self.dir2, self.files2[idx]))
            return img1, img2
        return img1

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
class GenFibMask():
    def get_diff_masks(img1, img2):
        assert len(img1.shape) == 2 and len(img2.shape) == 2, "Input images must be grayscale" 
                    
        # Compute the absolute difference between the blurred images
        diff = cv2.absdiff(img1, img2)

        # Remove shape artefactes (hollow and decreased edges)
        combined_mask = self.get_contour_masks(img1, img2)
        diff = cv2.bitwise_and(diff, diff, mask=combined_mask)

        # # normalise to [0,255] Useless!
        # diff = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Apply Gaussian Blur to reduce noise
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        # Apply Otsu's thresholding to get a binary mask
        _, diff_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return diff_mask

    def get_boxes(img):
        middle_boxes = []
        height, width = img.shape
        left_half, right_half = img[:, :width // 2], img[:, width // 2:]

        left_contour = max(cv2.findContours(left_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea, default=None)
        right_contour = max(cv2.findContours(right_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea, default=None)
        if left_contour is None or right_contour is None:
            print(f"Failed to find contour for input image")
            return

        def draw_box(contour, offset_x=0):
            x, y, w, h = cv2.boundingRect(contour)
            if offset_x == 0:
                mx_start, mx_end = x + offset_x + int(w * 0.45), x + offset_x + int(w * 0.75)
            else:
                mx_start, mx_end = x + offset_x + int(w * 0.25), x + offset_x + int(w * 0.55)
            my_start, my_end = y + int(h * 0.3), y + int(h * 0.75)
            # ax1.add_patch(plt.Rectangle((mx_start, my_start), mx_end - mx_start, my_end - my_start, edgecolor='blue', facecolor='none', linewidth=2))
            return (mx_start, my_start, mx_end - mx_start, my_end - my_start)

        left_middle_box = draw_box(left_contour)
        right_middle_box = draw_box(right_contour, offset_x=width // 2)

        middle_boxes.append((left_middle_box, right_middle_box))
        middle_boxes = np.concatenate(middle_boxes)
        return middle_boxes