from functools import reduce
from pathlib import Path
from typing import Union
import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np


class H5Dataset(Dataset):
    def __init__(self, path: Union[os.PathLike, str], transform=None):
        self.transform = transform

        with h5py.File(path, 'r') as f:
            self.images = f['image'][...]
            self.labels = f['label'][...]
            assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = torch.tensor(self.images[item])
        image = torch.unsqueeze(image, 0)
        label = torch.tensor(self.labels[item])

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return {
            'image': image,
            'label': label,
        }


def make_h5_directory_dataset(directory_path: Union[os.PathLike, str],
                              transform=None) -> Dataset:
    files = Path(directory_path).glob('*.h5')
    return reduce(lambda x, y: x + y, [H5Dataset(f, transform) for f in files])


class NpzDirectoryDataset(Dataset):
    def __init__(self, path: Union[os.PathLike, str], transform=None):
        self.transform = transform
        self.files = list(Path(path).glob('*.npz'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        with np.load(str(self.files[item])) as f:
            image = torch.tensor(f['image'])
            image = torch.unsqueeze(image, 0)
            label = torch.tensor(f['label'])

            if self.transform:
                image = self.transform(image)
                label = self.transform(label)

        return {
            'image': image,
            'label': label,
        }
