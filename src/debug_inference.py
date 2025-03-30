import os
gpu_num = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

import numpy as np
from os.path import basename, dirname, join
from PIL import Image


import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset, Dataset

from tqdm import tqdm
import sys
sys.path.append("/media/NAS06/gavinyue/disentanglement/scripts_segmentation/Pytorch_UNet")
from unet import UNet
# from utils.data_loading import BasicDataset

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from os import listdir
from os.path import splitext, isfile, join
import pandas as pd


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, scale: float = 1.0, total_num: int = None):
        self.images_dir = Path(images_dir)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale


        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        self.ids = sorted(self.ids)
        self.ids = self.ids[:total_num]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')


    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
        img = np.asarray(pil_img)


        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)  # Convert 2D grayscale to 3D RGB
            img = img.transpose((2, 0, 1))  # Change to (3, H, W) format
            # img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))

        if (img > 1).any():
            img = img / 255.0

        return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        # print(f'idx: {idx}, name: {name}')

        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        img = load_image(img_file[0])

        img_resized = self.preprocess(img, self.scale)


        return {
            'image': torch.as_tensor(img_resized.copy()).float().contiguous(),
            'name': name,
            'orig_img': torch.as_tensor(np.asarray(img)).float().contiguous()
        }



# # model parameters setting
# img_scale=0.5 # resize the image of 256 to 128
# model_path = '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/unet_checkpoints/No4.3_Syn_1.50Cov+RealVal_eps400_bs20/fold5_best_dice_epoch45.pth'

# '''
# Create dataset. This step include preprocssing of the images.
# the input of the model is: image value in [0, 1], tensor float32, shape (batch_size, 3, H, W)
# '''
# base_dir = './exp_img'
# case_name = '02_00019'


# case_dir = join(base_dir, case_name)
# save_dir = join(base_dir, case_name + '_pred')
# os.makedirs(save_dir, exist_ok=True)
# csv_file_path = join(base_dir, case_name+'_pixel_infer.csv')


def make_dataloader(case_dir, img_scale=0.5):
    dataset = BasicDataset(case_dir, img_scale)
    loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    data_loader = DataLoader(dataset, shuffle=False, drop_last=False, **loader_args) #type: ignore
    return data_loader

# make_dataloader(case_dir, img_scale)
'''
model loading
'''
def model_loading(model_path,device):
    if device is None:
        device = torch.device('cuda')
    model = UNet(n_channels=3, n_classes=2, bilinear=False)
    # model = model.to()

    model = model.to(memory_format=torch.channels_last) #type: ignore

    state_dict = torch.load(model_path, map_location=device)
    del state_dict['mask_values']
    model.load_state_dict(state_dict)
    logging.info(f'Model loaded from {model_path}')
    model.to(device=device)
    model.eval()
    return model

# device = torch.device('cuda')
# model = model_loading(model_path, device)


def model_inference(model, data_loader, device, save_dir, csv_file_path):
    pixel_num_lung_all = []
    pixel_num_fibrosis_all = []
    ID_check = []
    for batch in tqdm(data_loader, total=len(data_loader), desc='Inference round', unit='batch', leave=False):

        # batch = dataset[0]
        image = batch['image']
        slice_name = batch['name'][0]
        orig_img = batch['orig_img']

        # image = image[np.newaxis, ...]
        image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last) #type: ignore


        '''
        Model inference.
        The output mask has values of 0 or 255 with dtype of uint8, in shape of (batch_size, 2, 128, 128)
        '''
        # predict the mask. 
        mask_pred = model(image)
        mask_pred = (torch.sigmoid(mask_pred) > 0.5).int().squeeze()

        # Convert the tensor to a PIL image and save it
        mask_pred_image = mask_pred[1].cpu().numpy() * 255  # Convert to numpy array and scale to 0-255
        mask_pred_image = Image.fromarray(mask_pred_image.astype(np.uint8))
        mask_pred_reized = mask_pred_image.resize((256, 256), resample=Image.NEAREST)

        mask_pred_reized.save(f'{save_dir}/{slice_name}_mask.png')

        mask_np = np.array(mask_pred_reized)

        pixel_num_lung = torch.sum(orig_img != 0).item()
        pixel_num_fibrosis = np.sum(mask_np == 255)
        
        pixel_num_lung_all.append(pixel_num_lung)
        pixel_num_fibrosis_all.append(pixel_num_fibrosis)
        ID_check.append(slice_name)
        
        # data = {
        #     # 'Case': [case_name],
        #     'ID': [slice_name],
        #     # 'pixel_num_lung': [pixel_num_lung],
        #     'pixel_num_fibrosis': pixel_num_fibrosis
        # }

    # Create a DataFrame
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        df = pd.DataFrame()



    assert df['ID'].tolist() == ID_check, 'ID paired error'

    # Save the DataFrame to a CSV file
    if 'pixel_num_lung' not in df.columns:
        df['pixel_num_lung'] = pixel_num_lung_all
    df['pixel_num_fibrosis'] = pixel_num_fibrosis_all
    # df.to_csv(join(base_dir, case_name+'_pixel_infer.csv'), index=False)
    df.to_csv(csv_file_path, index=False)
    # header_exists = 'pixel_num_fibrosis' in df.columns
    # df.to_csv(csv_file_path, mode='a', header=not header_exists, index=False)


    # print(f'Case {case_name} done!')
    print(f'Pred mask saved to {save_dir}')
    print(f'Pixel number saved to {csv_file_path}')


def infer_main(model, device, base_dir, case_name, img_scale, csv_file_path, save_dir = None):
    case_dir = join(base_dir, case_name)
    # os.makedirs(case_dir, exist_ok=True)
    if save_dir is None:
        save_dir = join(base_dir, case_name + '_pred')
        os.makedirs(save_dir, exist_ok=True)


    data_loader = make_dataloader(case_dir, img_scale)
    model_inference(model, data_loader, device, save_dir, csv_file_path)
    
if __name__ == '__main__': 
    '''
    load model
    '''
    device = 'cuda'
    # full_supervised unet model
    model_path = '/media/NAS06/gavinyue/disentanglement/scripts_segmentation/unet_checkpoints/No1_Real_eps300_bs20/fold5_best_dice_epoch205.pth'
    base_dir = './quantification_result/full_supervised_unet'
    model = model_loading(model_path, device)
    
    # data loading
    base_dir = './exp_img'
    save_base_dir = './quantification_result/full_supervised_unet'
    case_names = listdir(base_dir)
    case_names = [name for name in case_names if ('_pred' not in name) and ('.' not in name)]
    case_names = sorted(case_names)
    
    # for case_name in case_names:
    #     case_dir = join(base_dir, case_name)
    #     data_loader = make_dataloader(case_dir, img_scale=0.5)
        
    #     save_dir = join(save_base_dir,'imgs', case_name+'_pred')
    #     os.makedirs(save_dir, exist_ok=True)
    #     model_inference(model, data_loader, device, save_dir, csv_file_path)
    

    