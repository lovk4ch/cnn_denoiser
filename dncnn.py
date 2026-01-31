import torch
from torch import nn


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
        )

        # --- общая активация ---
        # храним как поле, чтобы не создавать каждый раз
        self.act = nn.ReLU(inplace=True)

        # --- тело сети ---
        # последовательность одинаковых conv-блоков
        blocks = []
        for i in range(num_blocks):
            block = [
                nn.Conv2d(
                    features,
                    features,
                    kernel_size=3,
                    padding=1)
            ]

            if use_norm:
                block.append(nn.GroupNorm(8, features))

            block.append(self.act)
            blocks.append(nn.Sequential(*block))

        # регистрация слоёв в PyTorch
        self.blocks = nn.ModuleList(blocks)

        # --- выходной слой ---
        # возвращает тензор той же формы, что и вход
        self.conv_out = nn.Conv2d(
            features,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv_in(x))
        for block in self.blocks:
            x = block(x)
        x = self.conv_out(x)
        return x