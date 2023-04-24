import numpy as np
import torch
from torch.utils.data import Dataset


class NPYDataset(Dataset):
    def __init__(self, images_path, labels_path, indices=None, transform=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.indices = indices
        self.transform = transform

        # Load the labels
        self.labels = np.load(self.labels_path)
        self.labels = torch.from_numpy(self.labels).type(torch.LongTensor)

        # If indices are not provided, use all indices
        if self.indices is None:
            self.indices = np.arange(len(self.labels))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the index corresponding to idx
        index = self.indices[idx]

        # Load the image corresponding to index on-the-fly
        images = np.float32(np.load(self.images_path, mmap_mode='r'))
        image = images[index]

        # Apply the transform, if any
        if self.transform:
            image = self.transform(image)

        label = self.labels[index]
        
        return image, label