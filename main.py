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
from src.dncnn import Denoiser
from src.noise import add_noise
from src.utils import tensor_to_jpg, last_file_index, beep, load_config, get_criterion

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = ImageDataset(
    root="data/train",
    transform=transform,
    crop_size=[128, 256, 384],
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

    model_file = model_dir / cfg["data"]["model_file"]
    checkpoint_file = model_dir / cfg["data"]["model_checkpoint"]

    model = Denoiser(cfg["model"])
    if model_file.exists():
        model.load_state_dict(torch.load(model_file, weights_only=True, map_location=device))
        print("✔️️ Model weights loaded from disk.")
    else:
        print("⚠️ Model config doesn't exist, start training from scratch.")
    model.to(device)

    criterion = get_criterion(cfg["train"])
    g_loss = sys.float_info.max
    optim = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    if cfg["train"]["train_mode"] is True:
        model.train()
        last_epoch = last_file_index(res_dir)
        index = 1 + last_epoch * train_dataset.samples_per_epoch

        for epoch in range(last_epoch + 1, cfg["train"]["epochs"] + 1):
            train_bar = tqdm(train_loader, desc=f"\033[94m{index}")

            for batch in train_bar:
                batch = batch.to(device)
                noisy = add_noise(batch)
                batch = (batch * 2 - 1)
                noisy = (noisy * 2 - 1)

                base_noise = noisy - batch
                res_noise = model(noisy)

                # --- сумма квадратов разностей между пикселями картинок ---
                loss = criterion(base_noise, res_noise)

                # --- baseline: насколько noisy хуже clean ---
                with torch.no_grad():
                    mse_noisy = criterion(noisy, batch).item()

                # --- основной loss ---
                mse_pred = loss.item()

                # --- gain: во сколько раз модель лучше noisy ---
                gain = mse_noisy / (mse_pred + 1e-8)

                optim.zero_grad()
                loss.backward()

                # --- градиенты ---
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.detach().pow(2).sum().item()
                grad_norm = grad_norm ** 0.5

                optim.step()

                train_bar.set_postfix(
                    mse_pred=f"{mse_pred:.4f}",
                    gain=f"{gain:.2f}",
                    grad_norm=f"{grad_norm:.3e}",
                    iter=f"{index}"
                )

                if cache_every > 0 and index % cache_every == 0:
                    step_res = torch.cat([batch[0], noisy[0], (res_noise - base_noise)[0]], dim=2)
                    tensor_to_jpg(step_res, f'{cfg["data"]["cache_dir"]}step_res_{index}.jpg')

                index += 1

            loss = evaluate(model, device, cfg, criterion, epoch)
            model.train()
            print()
            print(f"evaluate: epoch {epoch}, loss: {loss:.4f}")

            # лучшая модель
            if loss < g_loss:
                torch.save(model.state_dict(), model_file)
                g_loss = loss
                print("🌟 Best state")

            # последняя модель
            torch.save(model.state_dict(), checkpoint_file)
            print("✔️️ Model checkpoint updated successfully.")

    else:
        for _ in range(1, cfg["train"]["test_images"] + 1):
            test_bar = tqdm(test_loader, desc=f"\033[94miteration {_}")
            model.eval()

            for batch in test_bar:
                batch = batch.to(device)
                batch = batch * 2 - 1

                with torch.no_grad():
                    pred_noise = model(batch)
                    denoised = batch - pred_noise

                    res = denoised.clamp(-1, 1)
                    tensor_to_jpg(res, f'{cfg["data"]["res_dir"]}res_{_}.jpg')

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
        res = torch.cat([clean, noisy, res], dim=3)
        tensor_to_jpg(res, f'{cfg["data"]["res_dir"]}res_ep_{epoch}.jpg')

    return loss.item()


if __name__ == "__main__":
    main()