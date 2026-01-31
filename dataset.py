import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, root, crop_size=640, samples_per_epoch=100, transform=transforms.ToTensor()):
        self.paths = [p for p in Path(root).iterdir()
                       if p.suffix.lower() in {'.jpg', '.jpeg'}]
        self.crop_size = crop_size
        self.samples_per_epoch = samples_per_epoch
        self.transform = transform

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        path = random.choice(self.paths)
        image = Image.open(path).convert('RGB')
        image = self.transform(image)

        C, H, W = image.shape
        P = self.crop_size
        if H < P or W < P:
            image = image.resize((max(W, P), max(H, P)))

        top = random.randint(0, H - P)
        left = random.randint(0, W - P)
        crop = image[:, top:top + P, left:left + P]
        return crop