import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import resize


class ImageDataset(Dataset):
    def __init__(self, root, crop_size=512, samples_per_epoch=100, transform=transforms.ToTensor()):
        self.paths = [p for p in Path(root).iterdir()
                       if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
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
            image = resize(image, [max(W, P), max(H, P)])
            _, H, W = image.shape

        top = random.randint(0, H - P)
        left = random.randint(0, W - P)
        crop = image[:, top:top + P, left:left + P]

        crop = crop * 2 - 1
        return crop