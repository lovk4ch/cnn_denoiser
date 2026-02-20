from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ImageDataset
from ..models.denoiser import Denoiser
from ..utils.model import save_weights, load_weights, load_config, get_device
from ..utils.noise import add_noise
from ..utils.common import get_criterion, get_psnr, save_tensor_as_jpg, padding_cat, beep, \
    img_to_tensor


class Trainer:
    def __init__(self):
        super().__init__()

        self.cfg = load_config() or {}
        self.device = get_device()
        self.loss = None

        self.model = Denoiser(self.cfg["denoise"]).to(self.device)
        self.criterion = get_criterion(self.cfg["train"])
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg["train"]["lr"])

        load_weights(self.model, self.cfg, self.device)
        self.model.train()

    @property
    def grad(self):
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.detach().pow(2).sum().item()
        return grad_norm ** 0.5

    def __call__(self, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # +[0..1]
        with torch.no_grad():
            noisy = add_noise(batch)

        # [-1..1]
        batch = (batch * 2 - 1)
        # [-1..1]
        noisy = (noisy * 2 - 1)

        base_noise = noisy - batch
        res_noise = self.model(noisy)
        clean = (noisy - res_noise).clamp(-1, 1)
        loss = self.criterion(base_noise, res_noise)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        self.optim.step()
        self.loss = loss.item()

        return batch, noisy, clean

    def run(self):
        # --- TRAIN ---

        train_dir = Path(self.cfg["data"]["train_dir"])
        train_dir.mkdir(parents=True, exist_ok=True)

        cache_dir = Path(self.cfg["data"]["cache_dir"])
        if self.cfg["global"]["clean_cache"]:
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = ImageDataset(
            crop_size=self.cfg["train"]["train_crop"],
            root=train_dir,
            transform=img_to_tensor(),
        )

        train_loader = DataLoader(
            pin_memory=True,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            drop_last=False,
            dataset=train_dataset,
        )

        index = 1
        epochs = self.cfg["train"]["epochs"]
        cache_every = self.cfg["train"]["cache_every"]

        for epoch in range(1, epochs + 1):
            bar = tqdm(train_loader, desc=f"\033[99mepoch {epoch}")

            g_loss = 0
            g_psnr = 0

            for batch in bar:
                batch = batch.to(self.device, non_blocking=True)
                batch, noisy, clean = self(batch)

                psnr = get_psnr(batch, clean)
                g_psnr += psnr or 0

                g_loss += self.loss or 0

                bar.set_postfix(
                    grad_denoise=f"{self.grad:.3f}",
                    iter=f"{index}",
                    loss=f"{self.loss or 0:.3f}",
                    psnr=f"{psnr or 0:.3f}"
                )

                if cache_every > 0 and index % cache_every / train_loader.batch_size == 0:
                    out = [batch, noisy, clean]
                    out = [t for t in out if t is not None]
                    rows = []
                    for i in range(batch.size(0)):
                        border = int(batch.size(3) * 10 // 1024)
                        row = padding_cat(out, border=border)
                        rows.append(row)
                        grid = torch.cat(rows, dim=1)
                        save_tensor_as_jpg(grid, f'{self.cfg["data"]["cache_dir"]}step_res_{index}.jpg')

                index += 1
            print()

            save_weights(self.model, self.cfg)
            g_loss /= bar.total
            g_psnr /= bar.total
            print(f"epoch {epoch}: loss={g_loss:.3f}, psnr={g_psnr:.3f}")

        beep()