import os
from io import BytesIO
from pathlib import Path

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, LSUNClass
import torch
import pandas as pd

import torchvision.transforms.functional as Ftrans
from torch.utils import data

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=['jpg'],
        do_augment: bool = True,
        do_transform: bool = True,
        do_normalize: bool = True,
        sort_names=False,
        has_subdir: bool = True,
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size

        # relative paths (make it shorter, saves memory and faster to sort)
        if has_subdir:
            self.paths = [
                p.relative_to(folder) for ext in exts
                for p in Path(f'{folder}').glob(f'**/*.{ext}')
            ]
        else:
            self.paths = [
                p.relative_to(folder) for ext in exts
                for p in Path(f'{folder}').glob(f'*.{ext}')
            ]
        if sort_names:
            self.paths = sorted(self.paths)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.folder, self.paths[index])
        img = Image.open(path)
        # if the image is 'rgba'!
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class SubsetDataset(Dataset):
    def __init__(self, dataset, size):
        assert len(dataset) >= size
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        assert index < self.size
        return self.dataset[index]


class BaseLMDB(Dataset):
    def __init__(self, path, original_resolution, zfill: int = 5):
        self.original_resolution = original_resolution
        self.zfill = zfill
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.original_resolution}-{str(index).zfill(self.zfill)}'.encode(
                'utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        return img


def make_transform(
    image_size,
    flip_prob=0.5,
    crop_d2c=False,
):
    if crop_d2c:
        transform = [
            d2c_crop(),
            transforms.Resize(image_size),
        ]
    else:
        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
    transform.append(transforms.RandomHorizontalFlip(p=flip_prob))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform)
    return transform


class FFHQlmdb(Dataset):
    def __init__(self,
                 path=os.path.expanduser('datasets/ffhq256.lmdb'),
                 image_size=256,
                 original_resolution=256,
                 split=None,
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=5)
        self.length = len(self.data)

        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 10000
            self.offset = 10000
        elif split == 'test':
            # first 10k
            self.length = 10000
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class Crop:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return Ftrans.crop(img, self.x1, self.y1, self.x2 - self.x1,
                           self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2)


def d2c_crop():
    # from D2C paper for CelebA dataset.
    cx = 89
    cy = 121
    x1 = cy - 64
    x2 = cy + 64
    y1 = cx - 64
    y2 = cx + 64
    return Crop(x1, x2, y1, y2)


class CelebAlmdb(Dataset):
    """
    also supports for d2c crop.
    """
    def __init__(self,
                 path,
                 image_size,
                 original_resolution=128,
                 split=None,
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 crop_d2c: bool = False,
                 **kwargs):
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)
        self.crop_d2c = crop_d2c

        if split is None:
            self.offset = 0
        else:
            raise NotImplementedError()

        if crop_d2c:
            transform = [
                d2c_crop(),
                transforms.Resize(image_size),
            ]
        else:
            transform = [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ]

        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class Horse_lmdb(Dataset):
    def __init__(self,
                 path=os.path.expanduser('datasets/horse256.lmdb'),
                 image_size=128,
                 original_resolution=256,
                 do_augment: bool = True,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        print(path)
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class Bedroom_lmdb(Dataset):
    def __init__(self,
                 path=os.path.expanduser('datasets/bedroom256.lmdb'),
                 image_size=128,
                 original_resolution=256,
                 do_augment: bool = True,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        print(path)
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self.data[index]
        img = self.transform(img)
        return {'img': img, 'index': index}


class CelebAttrDataset(Dataset):

    id_to_cls = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(self,
                 folder,
                 image_size=64,
                 attr_path=os.path.expanduser(
                     'datasets/celeba_anno/list_attr_celeba.txt'),
                 ext='png',
                 only_cls_name: str = None,
                 only_cls_value: int = None,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 d2c: bool = False):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.ext = ext

        # relative paths (make it shorter, saves memory and faster to sort)
        paths = [
            str(p.relative_to(folder))
            for p in Path(f'{folder}').glob(f'**/*.{ext}')
        ]
        paths = [str(each).split('.')[0] + '.jpg' for each in paths]

        if d2c:
            transform = [
                d2c_crop(),
                transforms.Resize(image_size),
            ]
        else:
            transform = [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        with open(attr_path) as f:
            # discard the top line
            f.readline()
            self.df = pd.read_csv(f, delim_whitespace=True)
            self.df = self.df[self.df.index.isin(paths)]

        if only_cls_name is not None:
            self.df = self.df[self.df[only_cls_name] == only_cls_value]

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        name = row.name.split('.')[0]
        name = f'{name}.{self.ext}'

        path = os.path.join(self.folder, name)
        img = Image.open(path)

        labels = [0] * len(self.id_to_cls)
        for k, v in row.items():
            labels[self.cls_to_id[k]] = int(v)

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'index': index, 'labels': torch.tensor(labels)}


class CelebD2CAttrDataset(CelebAttrDataset):
    """
    the dataset is used in the D2C paper.
    it has a specific crop from the original CelebA.
    """
    def __init__(self,
                 folder,
                 image_size=64,
                 attr_path=os.path.expanduser(
                     'datasets/celeba_anno/list_attr_celeba.txt'),
                 ext='jpg',
                 only_cls_name: str = None,
                 only_cls_value: int = None,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 d2c: bool = True):
        super().__init__(folder,
                         image_size,
                         attr_path,
                         ext=ext,
                         only_cls_name=only_cls_name,
                         only_cls_value=only_cls_value,
                         do_augment=do_augment,
                         do_transform=do_transform,
                         do_normalize=do_normalize,
                         d2c=d2c)


class CelebAttrFewshotDataset(Dataset):
    def __init__(
        self,
        cls_name,
        K,
        img_folder,
        img_size=64,
        ext='png',
        seed=0,
        only_cls_name: str = None,
        only_cls_value: int = None,
        all_neg: bool = False,
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
        d2c: bool = False,
    ) -> None:
        self.cls_name = cls_name
        self.K = K
        self.img_folder = img_folder
        self.ext = ext

        if all_neg:
            path = f'data/celeba_fewshots/K{K}_allneg_{cls_name}_{seed}.csv'
        else:
            path = f'data/celeba_fewshots/K{K}_{cls_name}_{seed}.csv'
        self.df = pd.read_csv(path, index_col=0)
        if only_cls_name is not None:
            self.df = self.df[self.df[only_cls_name] == only_cls_value]

        if d2c:
            transform = [
                d2c_crop(),
                transforms.Resize(img_size),
            ]
        else:
            transform = [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
            ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        name = row.name.split('.')[0]
        name = f'{name}.{self.ext}'

        path = os.path.join(self.img_folder, name)
        img = Image.open(path)

        # (1, 1)
        label = torch.tensor(int(row[self.cls_name])).unsqueeze(-1)

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'index': index, 'labels': label}


class CelebD2CAttrFewshotDataset(CelebAttrFewshotDataset):
    def __init__(self,
                 cls_name,
                 K,
                 img_folder,
                 img_size=64,
                 ext='jpg',
                 seed=0,
                 only_cls_name: str = None,
                 only_cls_value: int = None,
                 all_neg: bool = False,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 is_negative=False,
                 d2c: bool = True) -> None:
        super().__init__(cls_name,
                         K,
                         img_folder,
                         img_size,
                         ext=ext,
                         seed=seed,
                         only_cls_name=only_cls_name,
                         only_cls_value=only_cls_value,
                         all_neg=all_neg,
                         do_augment=do_augment,
                         do_transform=do_transform,
                         do_normalize=do_normalize,
                         d2c=d2c)
        self.is_negative = is_negative


class CelebHQAttrDataset(Dataset):
    id_to_cls = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(self,
                 path=os.path.expanduser('datasets/celebahq256.lmdb'),
                 image_size=None,
                 attr_path=os.path.expanduser(
                     'datasets/celeba_anno/CelebAMask-HQ-attribute-anno.txt'),
                 original_resolution=256,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        super().__init__()
        self.image_size = image_size
        self.data = BaseLMDB(path, original_resolution, zfill=5)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        with open(attr_path) as f:
            # discard the top line
            f.readline()
            self.df = pd.read_csv(f, delim_whitespace=True)

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.name
        img_idx, ext = img_name.split('.')
        img = self.data[img_idx]

        labels = [0] * len(self.id_to_cls)
        for k, v in row.items():
            labels[self.cls_to_id[k]] = int(v)

        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index, 'labels': torch.tensor(labels)}


class CelebHQAttrFewshotDataset(Dataset):
    def __init__(self,
                 cls_name,
                 K,
                 path,
                 image_size,
                 original_resolution=256,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        super().__init__()
        self.image_size = image_size
        self.cls_name = cls_name
        self.K = K
        self.data = BaseLMDB(path, original_resolution, zfill=5)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        self.df = pd.read_csv(f'data/celebahq_fewshots/K{K}_{cls_name}.csv',
                              index_col=0)

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.name
        img_idx, ext = img_name.split('.')
        img = self.data[img_idx]

        # (1, 1)
        label = torch.tensor(int(row[self.cls_name])).unsqueeze(-1)

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'index': index, 'labels': label}


class Repeat(Dataset):
    def __init__(self, dataset, new_len) -> None:
        super().__init__()
        self.dataset = dataset
        self.original_len = len(dataset)
        self.new_len = new_len

    def __len__(self):
        return self.new_len

    def __getitem__(self, index):
        index = index % self.original_len
        return self.dataset[index]



class FFHQ(Dataset):
    def __init__(self, root, label="gender", perform_one_class=False, classname=None, selectnum=0):
        """
        PyTorch DataSet for the FFHQ-Age dataset.
        :param root: Root folder that contains a directory for the dataset and the csv with labels in the root directory.
        :param label: Label we want to train on, chosen from the csv labels list.
        """
        self.root = root
        self.target_class = label

        # Store image paths
        self.images = [os.path.join(self.root, "ffhq/resized", file)
                       for file in os.listdir(os.path.join(self.root, "ffhq/resized")) if file.endswith('.jpg')]

        # Import labels from a CSV file
        self.labels = pd.read_csv("../datasets/Kaggle_FFHQ_Resized_256px/ffhq_aging_labels.csv")

        # Image transformation
        self.transform = transforms.Compose([
            # transforms.Resize(224),  # Used to be resize 256
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Make a lookup dictionary for the labels
        # Get column names of dataframe
        cols = self.labels.columns.values
        label_ids = {col_name: i for i, col_name in enumerate(cols)}
        self.class_id = label_ids[self.target_class]

        self.one_hot_encoding = {"male": 1,
                                 "female": 0}

        if perform_one_class == True:
            # class_male = dataset.labels[self.labels[self.target_class] == classname]
            # self.im
            self.labels = self.labels[self.labels[self.target_class] == classname]
            images2 = [self.images[i] for i in self.labels.index.tolist()]
            self.images = images2

            if selectnum != 0:
                self.labels = self.labels[0:selectnum]
                self.images = self.images[0:selectnum]


    def __getitem__(self, index):
        _img = self.transform(Image.open(self.images[index]))
        _label = self.one_hot_encoding[self.labels.iloc[index, self.class_id]]
        return _img, _label

    def __len__(self):
        return len(self.images)


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256, labels=False, filter_label=None):
        # print("path=", path)
        # self.env = lmdb.open(
        #     path,
        #     max_readers=32,
        #     readonly=True,
        #     lock=False,
        #     readahead=False,
        #     meminit=False,
        # )

        # if not self.env:
        #     raise IOError('Cannot open lmdb dataset', path)

        # with self.env.begin(write=False) as txn:
        #     self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        # self.data = BaseLMDB(path, original_resolution, zfill=5)
        # self.length = len(self.data)

        # CheXpert
        self.PRED_LABEL = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices"]

        if filter_label:
            if filter_label not in self.PRED_LABEL:
                raise Exception("Unrecognized label")
            self.filter_label = self.PRED_LABEL.index(filter_label)
        else:
            self.filter_label = None

        self.resolution = resolution
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            if self.labels:
                label_key = f"{self.resolution}-{str(index).zfill(5)}-label".encode("utf-8")
                label_bytes = txn.get(label_key)
                # we need labels to be type "ling"
                label = np.frombuffer(label_bytes, dtype=np.uint8).copy().astype(np.float32)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)

        if self.transform is not None:
            img = self.transform(img)

        if self.filter_label is not None:
            label = label[self.filter_label]
            label = np.array([0, 1]) if label == 1.0 else np.array([1, 0])

        if self.labels:
            return img, label
        return img

    def __getlabels__(self):
        with self.env.begin(write=False) as txn:
            labels = []
            for index in range(self.length):
                label_key = f"{self.resolution}-{str(index).zfill(5)}-label".encode("utf-8")
                label_bytes = txn.get(label_key)
                # we need labels to be type "long"
                label = np.frombuffer(label_bytes, dtype=np.uint8).copy().astype(np.float32)
                labels.append(label)
        return labels

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def get_dataset_osic(args, return_labels=False):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    # filter dataset for specific label
    if 'filter_label' in args and args.filter_label and not args.compare_to_healthy:
        dataset = MultiResolutionDataset(args.dataset_path, transform, args.size,
                                         labels=return_labels,
                                         filter_label=args.filter_label)

    # compare to healthy label
    elif 'compare_to_healthy' in args and args.compare_to_healthy:
        dataset = label_vs_no_finding_dataset(args, transform, return_labels)

    # dataset already filtered
    else:
        dataset = MultiResolutionDataset(args.dataset_path, transform, args.size, labels=False, filter_label=args.filter_label)

    return dataset


def get_dataloader_osic(args):
    dataset = get_dataset_osic(args)
    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True
    )

    return dataloader