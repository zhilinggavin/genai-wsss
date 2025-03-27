import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as trans_fn, InterpolationMode
from PIL import Image
import numpy as np
import torch
import random
from pathlib import Path
from os.path import splitext, isfile, join
from tqdm import tqdm
from os import listdir
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

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

def setup_seed(randomSeed):
    torch.manual_seed(randomSeed)
    torch.cuda.manual_seed_all(randomSeed)
    np.random.seed(randomSeed)
    random.seed(randomSeed)
    torch.backends.cudnn.deterministic = True
    
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)
    
class MedSAMDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, mask_suffix: str = '',total_num: int = None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        self.ids = sorted(self.ids)
        self.ids = self.ids[:total_num]
        
        self.box = [0, 0, 256, 256]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(img_3c, box_np):
        # image preprocessing
        H, W = img_3c.size
        img_1024 = trans_fn.resize(img_3c, (1024, 1024), InterpolationMode.LANCZOS)
        img_1024 = np.array(img_1024).astype(np.uint8)

        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1)
        
        # transfer box_np t0 1024x1024 scale
        box_1024 = box_np / np.array([W, H, W, H]) * 1024
        
        return img_1024_tensor, box_1024


    def __getitem__(self, idx):
        name = self.ids[idx]
        # print(f'idx: {idx}, name: {name}')
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        img = load_image(img_file[0])
        mask = load_image(mask_file[0]) 
        
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        
        img_1024_tensor, box_1024 = self.preprocess(img, self.box)
        
        # change mask to [0, 1]
        mask_np = np.asarray(mask)
        mask_values = np.unique(mask_np)
        mask_01 = np.zeros_like(mask_np, dtype=np.uint8)
        for i, v in enumerate(mask_values):
            if mask_np.ndim == 2:
                mask_01[mask_np == v] = i
            else:
                mask_01[(mask_np == v).all(-1)] = i
        

        return img_1024_tensor, box_1024, torch.as_tensor(mask_01.copy()).long().contiguous()
    
class MedSAMDatasetOrig(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, mask_suffix: str = '',total_num: int = None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        self.ids = sorted(self.ids)
        self.ids = self.ids[:total_num]
        
        self.bbox_shift = 20
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(img_3c, box_np):
        # image preprocessing
        H, W = img_3c.size
        img_1024 = trans_fn.resize(img_3c, (1024, 1024), InterpolationMode.LANCZOS)
        img_1024 = np.array(img_1024).astype(np.uint8)

        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1)
        
        # transfer box_np t0 1024x1024 scale
        box_1024 = box_np / np.array([W, H, W, H]) * 1024
        
        return img_1024_tensor, box_1024


    def __getitem__(self, idx):
        name = self.ids[idx]
        # print(f'idx: {idx}, name: {name}')
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        img = load_image(img_file[0])
        mask = load_image(mask_file[0]) 
        
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        
        # change mask to [0, 1]
        mask_np = np.asarray(mask)
        mask_values = np.unique(mask_np)
        mask_01 = np.zeros_like(mask_np, dtype=np.uint8)
        for i, v in enumerate(mask_values):
            if mask_np.ndim == 2:
                mask_01[mask_np == v] = i
            else:
                mask_01[(mask_np == v).all(-1)] = i
        
        # TODO original box fn from medsam
        label_ids = np.unique(mask_01)[1:]
        gt2D = np.uint8(
            mask_01 == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        self.box = np.array([x_min, y_min, x_max, y_max])
        
        
        img_1024_tensor, box_1024 = self.preprocess(img, self.box)
        

        

        return img_1024_tensor, box_1024, torch.as_tensor(mask_01.copy()).long().contiguous()    