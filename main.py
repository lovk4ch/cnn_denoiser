import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset import ImageDataset
from src.dncnn import DenoiseTrainer
from src.utils import tensor_to_jpg, beep, load_config, get_psnr, padding_cat

transform = transforms.Compose([
    transforms.ToTensor()
])


def main():
    # --- Загрузка конфигурации ---

    cfg = load_config()
    device = torch.device(cfg["global"]["device"] if torch.cuda.is_available() else 'cpu')

    model_dir = Path(cfg["data"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    train_dir = Path(cfg["data"]["train_dir"])
    train_dir.mkdir(parents=True, exist_ok=True)

    test_dir = Path(cfg["data"]["test_dir"])
    test_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(cfg["data"]["cache_dir"])

    if cfg["global"]["clean_cache"]:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_every = cfg["train"]["cache_every"]

    # --- Настройка моделей ---

    train_dataset = ImageDataset(
        root=cfg["data"]["train_dir"],
        transform=transform,
        crop_size=cfg["train"]["train_crop"],
    )

    test_dataset = ImageDataset(
        root=cfg["data"]["test_dir"],
        transform=transform,
        samples_per_epoch=1,
        crop_size=cfg["train"]["test_crop"],
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

    mode = cfg["train"]["mode"]
    epochs = cfg["train"]["epochs"]

    if mode != "inference":
        data_loader = train_loader
    else:
        data_loader = test_loader

    index = 1
    denoise = DenoiseTrainer(cfg, device)

    # --- Обучение / тест ---

    for epoch in range(1, epochs + 1):
        bar = tqdm(data_loader, desc=f"\033[99mepoch {epoch}")

        g_loss = 0
        g_psnr = 0

        for batch in bar:
            batch = batch.to(device, non_blocking=True)
            batch, noisy, clean = denoise(batch)

            loss = denoise.loss
            g_loss += loss or 0

            psnr = get_psnr(batch, clean)
            g_psnr += psnr or 0

            bar.set_postfix(
                grad_denoise=f"{denoise.grad:.3f}",
                iter=f"{index}",
                loss=f"{loss or 0:.3f}",
                psnr=f"{psnr or 0:.3f}"
            )

            if ((cache_every > 0 and index % cache_every / data_loader.batch_size == 0)
                    or mode == "inference"):
                out = [batch, noisy, clean]
                out = [t for t in out if t is not None]
                rows = []
                for i in range(batch.size(0)):
                    border = int(batch.size(3) * 10 // 1024)
                    row = padding_cat(out, border=border)
                    rows.append(row)
                    grid = torch.cat(rows, dim=1)
                    tensor_to_jpg(grid, f'{cfg["data"]["cache_dir"]}step_res_{index}.jpg')

            index += 1
        print()

        if mode != "inference":
            denoise.save_weights()

        g_loss /= bar.total
        g_psnr /= bar.total
        print(f"epoch {epoch}: loss={g_loss:.3f}, psnr={g_psnr:.3f}")

    beep()


if __name__ == "__main__":
    main()