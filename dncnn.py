import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, features: int, use_norm: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(features, features, 3, padding=1, bias=True),
        ]
        if use_norm:
            layers.append(nn.GroupNorm(8, features))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(features, features, 3, padding=1, bias=True))
        if use_norm:
            layers.append(nn.GroupNorm(8, features))

        self.net = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))   # skip + активация


class Denoiser(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: int = 64,
        num_blocks: int = 5,
        use_norm: bool = False,
    ):
        super().__init__()

        # --- входной слой ---
        # переводит изображение в пространство признаков
        self.conv_in = nn.Conv2d(
            in_channels,
            features,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # --- общая активация ---
        # храним как поле, чтобы не создавать каждый раз
        self.act = nn.ReLU(inplace=True)

        # --- тело сети ---
        # регистрация слоёв в PyTorch
        # последовательность одинаковых conv-блоков
        blocks = []
        for i in range(num_blocks):
            block = [
                nn.Conv2d(
                    features,
                    features,
                    kernel_size=3,
                    padding=2 if i == 2 else 1,
                    dilation=2 if i == 2 else 1)
            ]

            if use_norm:
                block.append(nn.GroupNorm(8, features))

            block.append(self.act)
            blocks.append(nn.Sequential(*block))

        self.blocks = nn.ModuleList(blocks)

        '''
        self.blocks = nn.ModuleList([
            ResBlock(features, use_norm) for _ in range(num_blocks)
        ])
        '''

        # --- выходной слой ---
        # возвращает тензор той же формы, что и вход
        self.conv_out = nn.Conv2d(
            features,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv_in(x))
        for block in self.blocks:
            x = block(x)
        return self.conv_out(x)