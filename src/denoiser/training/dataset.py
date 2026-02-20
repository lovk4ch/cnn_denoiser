from pathlib import Path

import torch
from PIL import Image
from sympy.core.random import shuffle
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize

from denoiser.utils.common import img_to_tensor


class ImageDataset(Dataset):
    def __init__(self, root, crop_size=None, samples_per_epoch=100, transform=img_to_tensor()):
        if crop_size is None: crop_size = [256]

        self.paths = [p for p in Path(root).iterdir()
                       if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
        self.crop_size = crop_size
        self.samples_per_epoch = samples_per_epoch
        self.transform = transform

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        if shuffle:
            path = self.paths[torch.randint(0, len(self.paths), (1,)).item()]
        else:
            path = self.paths[index]

        image = Image.open(path).convert('RGB')
        image = self.transform(image)

        C, H, W = image.shape
        P = self.crop_size[torch.randint(0, len(self.crop_size), (1,)).item()]

        if P > 0:
            if H < P or W < P:
                image = resize(image, [max(W, P), max(H, P)])
                _, H, W = image.shape

            top = torch.randint(0, H - P + 1, (1,)).item()
            left = torch.randint(0, W - P + 1, (1,)).item()
            image = image[:, top:top + P, left:left + P]

        return image