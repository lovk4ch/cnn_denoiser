import re
import sys
from pathlib import Path

import torch
import torch.nn.functional
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.functional import pad
from torchvision.transforms import transforms as T
from torchvision.transforms.v2.functional import resize, to_pil_image
from torchvision.utils import make_grid


def img_to_tensor(normalize=False, max_size=-1):
    """
    :param normalize: whether the image is converted to the range [-1, 1]
    :param max_size: shrink big images (for low-memory servers)
    """
    transforms = []

    if max_size > 0:
        transforms.append(
            T.Lambda(lambda img: resize_max_size(img, max_size))
        )

    transforms.append(T.ToTensor())

    if normalize:
        transforms.append(T.Normalize(mean=[0.5], std=[0.5]))

    return T.Compose(transforms)

def resize_max_size(img, max_size):
    w, h = img.size
    max_dim = max(w, h)

    if max_dim <= max_size:
        return img

    scale = max_size / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)

    return resize(img, [new_h, new_w])

def to_image(n, is_1x1=True):
    n = n.detach().cpu().squeeze(0)

    if is_1x1:
        n = (n + 1) / 2
        n = n.clamp(0, 1)

    return n

def tensor_to_img(n, is_1x1=True):
    n = to_image(n, is_1x1)
    return to_pil_image(n)

def save_tensor_as_jpg(n, name, is_1x1=True):
    tensor_to_img(n, is_1x1).save(name)
    return

def get_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    return 10 * torch.log10(max_val ** 2 / mse).item()

def beep():
    if sys.platform.startswith('win'):
        import winsound
        winsound.PlaySound(
            str(Path("notify.wav").resolve()),
            winsound.SND_FILENAME
    )
    return

def save_cnn_grid(tensors=None):
    if tensors is None:
        tensors = []
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

def padding_cat(images, border=10, value=1.0):
    """
    images: list of tensors [1, 3, H, W]
    return: [1, 3, H+2b, N*(W+2b)]
    """
    bordered = []
    for img in images:
        # pad = (left, right, top, bottom)
        img_b = pad(
            img,
            pad=(border, border, border, border),
            value=value
        )
        bordered.append(img_b)

    return torch.cat(bordered, dim=3)  # dim=3 → ширина

def last_file_index(path):
    files = [p for p in Path(path).iterdir() if p.is_file()]
    if not files:
        return 0

    def num_key(p):
        m = re.search(r"\d+$", p.stem)
        return int(m.group()) if m else -1

    last = max(files, key=num_key)
    return num_key(last)

def plot_curves(curves, labels=None, title="Metrics",
                xlabel="step", ylabel="value",
                logy=False):
    """
    curves: list[list[float]]
    labels: list[str] | None
    """
    if labels is None:
        labels = [f"curve_{i}" for i in range(len(curves))]

    for y, label in zip(curves, labels):
        if y:
            plt.plot(range(len(y)), y, label=label)

    if logy:
        plt.yscale("log")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"data/plots/{title}.jpg")
    plt.show()

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