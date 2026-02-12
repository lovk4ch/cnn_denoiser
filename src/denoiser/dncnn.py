from pathlib import Path
from typing import Tuple

import torch
from torch import nn, Tensor

from noise import add_noise
from utils import get_criterion


class ResBlock(nn.Module):
    def __init__(self, use_norm: bool = False, dilation: int = 1, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or {}

        channels = int(cfg.get("channels", 64))
        k = int(cfg.get("kernel_size", 3))

        norm_groups = int(cfg.get("norm_groups", 8))

        layers = [
            nn.Conv2d(channels, channels, k, dilation=dilation, padding=dilation * (k - 1) // 2, bias=True),
        ]
        if use_norm:
            layers.append(nn.GroupNorm(norm_groups, channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(channels, channels, k, dilation=dilation, padding=dilation * (k - 1) // 2, bias=True),
        )
        if use_norm:
            layers.append(nn.GroupNorm(norm_groups, channels))

        self.net = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))   # skip + activation


class Denoise(nn.Module):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or {}

        in_channels: int = int(cfg.get("in_channels", 3))
        out_channels: int = int(cfg.get("out_channels", 3))
        features: int = int(cfg.get("channels", 64))
        depth: int = int(cfg.get("depth", 5))
        k = int(cfg.get("kernel_size", 3))
        use_norm: bool = bool(cfg.get("use_norm", False))
        norm_groups: int = int(cfg.get("norm_groups", 8))


        # dilation: список или число
        dil = cfg.get("dilation", 1)
        if isinstance(dil, (list, tuple)):
            dilation_list = [int(x) for x in dil]
        else:
            dilation_list = [int(dil)] * max(depth + 2, 2)  # +2: in/out

        # гарантируем длину: [in] + depth + [out]
        need_len = depth + 2
        if len(dilation_list) < need_len:
            dilation_list = dilation_list + [dilation_list[-1]] * (need_len - len(dilation_list))

        # --- входной слой ---
        # переводит изображение в пространство признаков
        d = dilation_list[0]
        self.conv_in = nn.Conv2d(
            in_channels,
            features,
            kernel_size=k,
            dilation=d,
            padding=d * (k - 1) // 2,
            bias=True,
        )

        # --- общая активация ---
        # храним как поле, чтобы не создавать каждый раз
        self.act = nn.ReLU(inplace=True)

        # --- тело сети ---
        # регистрация слоёв в PyTorch
        # последовательность одинаковых conv-блоков
        blocks = []
        for i in range(depth):
            d = dilation_list[i + 1]
            block = [
                nn.Conv2d(
                    features,
                    features,
                    kernel_size=k,
                    dilation=d,
                    padding=d * (k - 1) // 2,
                    bias=False if use_norm else True,
                )
            ]
            if use_norm:
                block.append(nn.GroupNorm(norm_groups, features))
            block.append(nn.ReLU(inplace=True))
            blocks.append(nn.Sequential(*block))

        self.blocks = nn.ModuleList(blocks)

        # --- выходной слой ---
        # возвращает тензор той же формы, что и вход
        d = dilation_list[-1]
        self.conv_out = nn.Conv2d(
            features,
            out_channels,
            kernel_size=k,
            dilation=d,
            padding=d * (k - 1) // 2,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv_in(x))
        for block in self.blocks:
            x = block(x)
        return self.conv_out(x)

class DenoiseTrainer:
    def __init__(self, cfg: dict | None = None, device: torch.device = None):
        super().__init__()
        self.cfg = cfg or {}

        self._grad = None
        self._loss = None

        train = cfg.get("train")
        data = cfg.get("data")
        model_dir = Path(data.get("model_dir"))

        self.TRAIN_MODE = train.get("mode") == "denoise"

        self.model_checkpoint = model_dir / data.get("denoise_checkpoint")
        self.model_path = model_dir / data.get("denoise_path")
        self.model = Denoise(cfg.get("denoise")).to(device)
        self.criterion = get_criterion(cfg["train"])
        self.optim = torch.optim.Adam(self.model.parameters(), lr=train.get("lr"))

        self.load_weights(device)

        if self.TRAIN_MODE:
            self.model.train()
        else:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    @property
    def loss(self):
        return self._loss

    @property
    def grad(self):
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.detach().pow(2).sum().item()
        return grad_norm ** 0.5

    def __call__(self, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        noisy = None

        if self.TRAIN_MODE:
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
            self._loss = loss.item()

        else:
            with torch.no_grad():
                # [-1..1]
                batch = (batch * 2 - 1)
                res_noise = self.model(batch)
                clean = (batch - res_noise).clamp(-1, 1)

        return batch, noisy, clean

    def load_weights(self, device):
        if not self.model_path.exists():
            print(f"⚠️ Model {self.model_path.name} not found, training from scratch.")
            return False

        try:
            state = torch.load(
                self.model_path,
                weights_only=True,
                map_location=device
            )
            self.model.load_state_dict(state)
            print(f"✔️ Model {self.model_path.name} weights loaded.")
            return True

        except Exception as e:
            print(f"❌ Failed to load weights: {e}")
            print(f"⚠️ Training from scratch.")
            return False

    def save_weights(self):
        try:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)
            print(f"✔️ Model {self.model_path.name} checkpoint updated successfully.")
        except Exception as e:
            print(f"❌ Failed to save model {self.model_path.name} : {e}")