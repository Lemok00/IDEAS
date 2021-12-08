from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from imutils.paths import list_files


class ResizedLMDBDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
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

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


class OfficalLMDBDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
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

        self.keys = []
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            for idx,(key, _) in enumerate(cursor):
                self.keys.append(key)

        self.length = len(self.keys)
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = self.keys[index]
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer).resize((self.resolution, self.resolution))
        img = self.transform(img)

        return img


IMG_EXTENSIONS = ['webp', '.png', '.jpg', '.jpeg', '.ppm', '.bmp', '.pgm', '.tif', '.tiff']


class NormalDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.files = []
        for file in sorted(list(list_files(path))):
            if any(file.lower().endswith(ext) for ext in IMG_EXTENSIONS):
                self.files.append(file)

        self.resolution = resolution
        self.transform = transform
        self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = Image.open(self.files[index]).resize((self.resolution, self.resolution))
        img = self.transform(img)

        return img


def set_dataset(type, path, transform, resolution):
    datatype = None
    if type == 'offical_lmdb':
        datatype = OfficalLMDBDataset
    elif type == 'resized_lmdb':
        datatype = ResizedLMDBDataset
    else:
        datatype = NormalDataset
    return datatype(path, transform, resolution)
