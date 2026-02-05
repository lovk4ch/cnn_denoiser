import re
from pathlib import Path

import winsound
from torchvision.transforms.v2.functional import to_pil_image

from noise import add_noise


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
        tensor_to_jpg(image, "test.jpg", is_1x1=False)
        break
    return

def last_file_index(path):
    files = [p for p in Path(path).iterdir() if p.is_file()]
    if not files:
        return 0

    def num_key(p):
        m = re.search(r"\d+$", p.stem)
        return int(m.group()) if m else -1

    last = max(files, key=num_key)
    return num_key(last)