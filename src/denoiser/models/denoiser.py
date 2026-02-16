import torch
from torch import nn


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


class Denoiser(nn.Module):
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