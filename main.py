import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset import ImageDataset
from src.dncnn import DenoiseTrainer
from src.utils import tensor_to_jpg, last_file_index, beep, load_config

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = ImageDataset(
    root="data/train",
    transform=transform,
    crop_size=[512, 768, 1024],
)

test_dataset = ImageDataset(
    root="data/test",
    transform=transform,
    crop_size=[512],
    samples_per_epoch=1,
)

train_loader = DataLoader(
    dataset=train_dataset,
    pin_memory=True,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    pin_memory=True,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    drop_last=False,
)


def main():
    # --- загрузка конфигурации и настройка моделей ---
    cfg = load_config()
    device = torch.device(cfg["global"]["device"] if torch.cuda.is_available() else 'cpu')

    # basic_noise_pair(test_loader, device)
    # return

    model_dir = Path(cfg["data"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    train_dir = Path(cfg["data"]["train_dir"])
    train_dir.mkdir(parents=True, exist_ok=True)

    test_dir = Path(cfg["data"]["test_dir"])
    test_dir.mkdir(parents=True, exist_ok=True)

    res_dir = Path(cfg["data"]["res_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(cfg["data"]["cache_dir"])

    if cfg["global"]["clean_cache"]:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_every = cfg["train"]["cache_every"]

    mode = cfg["train"]["mode"]
    data_loader = test_loader if mode == "inference" else train_loader

    denoise = DenoiseTrainer(cfg, device)

    # --- обучение ---
    last_epoch = last_file_index(res_dir)
    index = 1 + last_epoch * getattr(data_loader.dataset, "samples_per_epoch", 100)

    for epoch in range(last_epoch + 1, cfg["train"]["epochs"] + 1):
        bar = tqdm(data_loader, desc=f"\033[94m{index}")

        for batch in bar:
            batch = batch.to(device, non_blocking=True)
            noisy, clean = denoise(batch)

            bar.set_postfix(
                grad_denoise=f"{denoise.grad:.3e}",
                iter=f"{index}",
                loss=f"{denoise.loss}"
            )

            if ((cache_every > 0 and index % cache_every / data_loader.batch_size == 0)
                    or mode == "inference"):
                rows = []
                for i in range(batch.size(0)):
                    row = torch.cat([batch[i], noisy[i], clean[i]], dim=2)
                    rows.append(row)
                    grid = torch.cat(rows, dim=1)
                    tensor_to_jpg(grid, f'{cfg["data"]["cache_dir"]}step_res_{index}.jpg')

            index += 1
    beep()


if __name__ == "__main__":
    main()