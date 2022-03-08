from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset

from imutils.paths import list_files


class LMDBDataset(Dataset):
    def __init__(self, path, transform, resolution=256, max_num=70000):
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
            for idx, (key, _) in enumerate(cursor):
                self.keys.append(key)
                if idx > max_num:
                    break

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
    def __init__(self, path, transform, resolution=256, max_num=70000):
        self.files = []
        listed_files = sorted(list(list_files(path)))
        for i in range(min(max_num, len(listed_files))):
            file = listed_files[i]
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
    if type == 'lmdb':
        datatype = LMDBDataset
    elif type == 'normal':
        datatype = NormalDataset
    else:
        raise NotImplementedError
    return datatype(path, transform, resolution)
