import os
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from Pytorch_UNet.unet import UNet
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from os import listdir
from os.path import splitext, isfile, join
import pandas as pd


# Constants
GPU_NUM = '0'
IMAGE_SIZE = 256

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUM
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, scale: float = 1.0, total_num: int = None, ids: list = None):
        self.images_dir = Path(images_dir)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        if ids is not None:
            self.ids = [splitext(file)[0] for file in ids]
        else:
            self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
            self.ids = sorted(self.ids)
            self.ids = self.ids[:total_num]

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')



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
        
def make_dataloader(case_dir, img_scale=0.5, ids: list = None):
    dataset = BasicDataset(case_dir, img_scale, ids=ids)
    loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    data_loader = DataLoader(dataset, shuffle=False, drop_last=False, **loader_args) #type: ignore
    return data_loader


def model_loading(model_path,device):
    '''
    model loading
    '''
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

def infer(model, data_loader, device, save_dir, csv_file_path=None):
    pixel_num_lung_all = []
    pixel_num_fibrosis_all = []
    slcie_ID = []
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
        slcie_ID.append(slice_name)
        

    # Create a DataFrame
    columns = ['Case','ID','size','pixel_num_lung','pixel_num_fibrosis']
    df = pd.DataFrame(columns=columns)
    # Save the DataFrame to a CSV file
    df['Case'] = [slice_name.split('_')[0] for slice_name in slcie_ID]
    df['ID'] = slcie_ID
    df['size'] = IMAGE_SIZE
    df['pixel_num_lung'] = pixel_num_lung_all
    df['pixel_num_fibrosis'] = pixel_num_fibrosis_all
    # df.to_csv(join(base_dir, case_name+'_pixel_infer.csv'), index=False)

    header_exists = os.path.exists(csv_file_path)
    df.to_csv(csv_file_path, mode='a', header= not header_exists, index=False)

    # print(f'Case {case_name} done!')
    print(f'Pred mask saved to {save_dir}')
    print(f'Pixel number saved to {csv_file_path}')

if __name__ == '__main__': 
    '''
    load model
    '''
    device = 'cuda'
    # full_supervised unet model
    model_path = '../experiments/wsss_unet/checkpoints/fold5_best_dice_epoch45.pth'
    img_dir = '../data/OSIC/processed/fibrosis'
    save_base_dir = '../experiments/wsss_unet/results/OSIC'
    model = model_loading(model_path, device)
    
    # data loading
    # Load test set case IDs from CSV
    df = pd.read_csv('../data/OSIC/doctor_category.csv')
    caseids_test = df['case_id'][df['test'] == 1].tolist()
    caseids_test = [str(caseid).zfill(3) for caseid in caseids_test]
    
    
    for caseid in caseids_test:
        file_names = sorted(os.listdir(img_dir))
        file_names = [name for name in file_names if name.startswith(caseid) and name.endswith(('.png', '.jpg', '.jpeg'))]
        
        
        data_loader = make_dataloader(img_dir, img_scale=0.5, ids = file_names)
        
        save_dir = join(save_base_dir,'pred_mask', 'fibrosis_pred')
        os.makedirs(save_dir, exist_ok=True)
        csv_file_path = join(save_base_dir, 'quant', 'slice_check.csv')
        infer(model, data_loader, device, save_dir, csv_file_path)