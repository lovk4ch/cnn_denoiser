import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset import ImageDataset
from src.dncnn import Denoise
from src.noise import add_noise
from src.unet_restore import UNetRestore
from src.utils import tensor_to_jpg, last_file_index, beep, load_config, get_criterion, grad

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

    denoise_checkpoint = model_dir / cfg["data"]["denoise_checkpoint"]
    denoise_path = model_dir / cfg["data"]["denoise_path"]
    denoise = Denoise(cfg["denoise"]).to(device)
    load_weights(denoise, device, denoise_path)
    optim_d = torch.optim.Adam(denoise.parameters(), lr=cfg["train"]["lr"])

    restore_checkpoint = model_dir / cfg["data"]["restore_checkpoint"]
    restore_path = model_dir / cfg["data"]["restore_path"]
    restore = UNetRestore(cfg["restore"]).to(device)
    load_weights(restore, device, restore_path)
    optim_r = torch.optim.Adam(restore.parameters(), lr=cfg["train"]["lr"])

    criterion = get_criterion(cfg["train"])
    g_loss = sys.float_info.max
    mode = cfg["train"]["mode"]

    # --- настройка режимов ---
    if mode not in {"denoise", "restore", "inference"}:
        raise ValueError(f"Unknown mode: {mode}")

    data_loader = None

    if mode == "denoise":
        data_loader = train_loader
        denoise.train()

    if mode == "restore":
        data_loader = train_loader
        denoise.eval()
        for p in denoise.parameters():
            p.requires_grad_(False)
        restore.train()

    if mode == "inference":
        data_loader = test_loader
        denoise.eval()
        for p in denoise.parameters():
            p.requires_grad_(False)
        restore.eval()
        for p in restore.parameters():
            p.requires_grad_(False)

    # --- обучение ---
    last_epoch = last_file_index(res_dir)
    index = 1 + last_epoch * getattr(data_loader.dataset, "samples_per_epoch", 100)

    for epoch in range(last_epoch + 1, cfg["train"]["epochs"] + 1):
        bar = tqdm(data_loader, desc=f"\033[94m{index}")

        for batch in bar:
            batch = batch.to(device)    # [0..1]
            noisy = add_noise(batch)    # [0..1]
            clean = (batch * 2 - 1)     # [-1..1]
            noisy = (noisy * 2 - 1)     # [-1..1]

            denoised = None
            restored = None

            if mode == "denoise":
                base_noise = noisy - batch
                res_noise = denoise(noisy)
                denoised = (noisy - res_noise).clamp(-1, 1)
                loss = criterion(base_noise, res_noise)
                print(denoised.shape)

                optim_d.zero_grad()
                loss.backward()
                optim_d.step()

            if mode == "restore":
                with torch.no_grad():
                    res_noise = denoise(noisy)
                    denoised = (noisy - res_noise).clamp(-1, 1)

                restored = restore(denoised)
                loss = criterion(restored, clean)

                optim_r.zero_grad()
                loss.backward()
                optim_r.step()

            if mode == "inference":
                with torch.no_grad():
                    res_noise = denoise(noisy)
                    denoised = (noisy - res_noise).clamp(-1, 1)
                    restored = restore(denoised)
                    loss = criterion(restored, clean)

            bar.set_postfix(
                grad_denoise=f"{grad(denoise):.3e}",
                grad_restore=f"{grad(restore):.3e}",
                iter=f"{index}",
                loss=f"{loss.item()}"
            )

            if index % 25 == 0 or mode == "inference":
                cache = [clean[0], noisy[0]]
                if denoised is not None:
                    cache.append(denoised[0])
                if restored is not None:
                    cache.append(restored[0])
                if cache_every > 0 and index % cache_every == 0:
                    step_res = torch.cat(cache, dim=2)
                    tensor_to_jpg(step_res, f'{cfg["data"]["cache_dir"]}step_res_{index}.jpg')

            index += 1

        '''
        loss = evaluate(denoise, device, cfg, criterion, epoch)
        denoise.train()
        print()
        print(f"evaluate: epoch {epoch}, loss: {loss:.4f}")
        '''

        torch.save(denoise.state_dict(), denoise_path)
        torch.save(restore.state_dict(), restore_path)
        print("✔️️ Model checkpoint updated successfully.")

    beep()

def evaluate(model, device, cfg, criterion=nn.L1Loss(), epoch=1):
    clean = transform(Image.open("basic.jpg").convert("RGB")).unsqueeze(0).to(device)
    noisy = transform(Image.open("noisy.jpg").convert("RGB")).unsqueeze(0).to(device)

    clean = clean * 2 - 1
    noisy = noisy * 2 - 1

    model.eval()
    with torch.no_grad():
        pred_noise = model(noisy)
        denoised = noisy - pred_noise

        loss = criterion(denoised, clean)

        res = denoised.clamp(-1, 1)
        # res = torch.cat([clean, noisy, res], dim=3)
        tensor_to_jpg(res, f'{cfg["data"]["res_dir"]}res_ep_{epoch}.jpg')

    return loss.item()

def load_weights(model, device, path):
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
        print(f"✔️️ Model {path.name} weights loaded from disk.")
    else:
        print(f"⚠️ Model {path.name} config doesn't exist, start training from scratch.")


if __name__ == "__main__":
    main()