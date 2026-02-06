import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, use_norm: bool = False, cfg = None):
        super().__init__()
        layers = [
            nn.Conv2d(cfg["channels"], cfg["channels"], cfg["kernel_size"], padding=cfg["dilation"], bias=True),
        ]
        if use_norm:
            layers.append(nn.GroupNorm(8, cfg["channels"]))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(cfg["channels"], cfg["channels"], 3, padding=1, bias=True))
        if use_norm:
            layers.append(nn.GroupNorm(8, cfg["channels"]))

        self.net = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))   # skip + активация


class Denoiser(nn.Module):
    def __init__(
        self, cfg
    ):
        super().__init__()

        in_channels: int = 3
        out_channels: int = 3
        features: int = cfg["channels"]
        depth: int = cfg["depth"]
        use_norm: bool = False

        # --- входной слой ---
        # переводит изображение в пространство признаков
        self.conv_in = nn.Conv2d(
            in_channels,
            features,
            kernel_size=cfg["kernel_size"],
            padding=cfg["dilation"][0],
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

            block = [
                nn.Conv2d(
                    features,
                    features,
                    kernel_size=cfg["kernel_size"],
                    dilation=cfg["dilation"][i],
                    padding=cfg["dilation"][i],
                )
            ]

            if use_norm:
                block.append(nn.GroupNorm(8, features))

            block.append(self.act)
            blocks.append(nn.Sequential(*block))

        self.blocks = nn.ModuleList(blocks)

        '''
        self.blocks = nn.ModuleList([
            ResBlock(use_norm, cfg) for _ in range(depth)
        ])
        '''

        # --- выходной слой ---
        # возвращает тензор той же формы, что и вход
        self.conv_out = nn.Conv2d(
            features,
            out_channels,
            kernel_size=cfg["kernel_size"],
            padding=cfg["dilation"][-1],
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv_in(x))
        for block in self.blocks:
            x = block(x)
        return self.conv_out(x)