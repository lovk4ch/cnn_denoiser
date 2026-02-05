import re
from pathlib import Path

import torch
import winsound
import yaml
from torch import nn
from torchvision.transforms.v2.functional import to_pil_image
from torchvision.utils import make_grid

from noise import add_noise


def load_config(path="config/project.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def beep():
    winsound.PlaySound(
        str(Path("notify.wav").resolve()),
        winsound.SND_FILENAME
    )

def normalize(n):
    m = n.abs().amax(dim=(1, 2), keepdim=True) + 1e-8
    n = (n / m) * 0.5 + 0.5

    return n

def to_image(n, is_1x1=True):
    n = n.detach().cpu().squeeze(0)

    if is_1x1:
        n = (n + 1) / 2
        n = n.clamp(0, 1)

    return n

def tensor_to_jpg(n, name, is_1x1=True):
    n = to_image(n, is_1x1)
    to_pil_image(n).save(name)

def basic_noise_pair(loader, device):
    for image in loader:
        image = image.to(device)
        tensor_to_jpg(image, "basic.jpg", is_1x1=False)
        image = add_noise(image)
        image = image.clamp(0, 1)
        tensor_to_jpg(image, "noisy.jpg", is_1x1=False)
        break
    return

def save_cnn_grid(tensors=[]):
    tiles = []
    for t in tensors:
        t = t.detach().cpu()
        t = t - t.amin(dim=(1, 2), keepdim=True)
        t = t / (t.amax(dim=(1, 2), keepdim=True) + 1e-8)
        t = t.unsqueeze(1)  # [64, 1, H, W]
        tiles.append(t)

    tiles = torch.stack(tiles, dim=1)  # [64, 6, 1, H, W]
    tiles = tiles.flatten(0, 1)  # [64*6, 1, H, W]

    grid = make_grid(tiles, nrow=6)
    img = to_pil_image(grid)
    img.save(f"data/cache/grid.jpg")

def last_file_index(path):
    files = [p for p in Path(path).iterdir() if p.is_file()]
    if not files:
        return 0

    def num_key(p):
        m = re.search(r"\d+$", p.stem)
        return int(m.group()) if m else -1

    last = max(files, key=num_key)
    return num_key(last)

def get_criterion(cfg_loss):
    name = cfg_loss["loss"].lower()

    if name == "l1":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name in ("huber", "smooth_l1"):
        # PyTorch: SmoothL1Loss ~= Huber с параметром beta (в новых версиях)
        delta = cfg_loss.get("huber_delta", 1.0)
        return nn.SmoothL1Loss(beta=delta)

    raise ValueError(f"Unknown loss: {name}")