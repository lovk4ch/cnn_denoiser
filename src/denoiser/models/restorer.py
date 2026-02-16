import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """(Conv3x3 -> ReLU) x2 - два уровня для понимания паттерна"""
    def __init__(self, in_channels: int, out_channels: int, k: int = 3, d: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, dilation=d, padding=d * (k - 1) // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=k, dilation=d, padding=d * (k - 1) // 2, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downsample: stride-2 conv (предпочтительнее maxpool для restoration)."""
    def __init__(self, in_channels: int, out_channels: int, k: int = 3, s: int = 2):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, stride=s, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.down(x))


class Up(nn.Module):
    """
    Upsample: bilinear upsample + conv, чтобы избежать 'шахматки' от ConvTranspose.
    Потом concat со skip и ConvBlock.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, k: int = 3, d: int = 1):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=d * (k - 1) // 2, bias=True)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        # --- upsample ---
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # --- чуть "поджать" каналы ---
        x = F.relu(self.reduce(x), inplace=True)

        # --- на всякий случай выровнять размер (если из-за нечётных размеров чуть не сошлось) ---
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        # --- concat по каналам ---
        x = torch.cat([x, skip], dim=1)

        # --- обучаемый смеситель, склеивает детали ---
        return self.block(x)


class Restorer(nn.Module):
    """
    Мини-U-Net на 3 уровня.
    base=32 (или 48).
    ВЫХОД: pred_clean (3 канала).
    Можно добавить global skip: return x_in + pred_residual (опционально).
    """
    def __init__(self, cfg: dict | None = None):
        super().__init__()

        base: int = int(cfg.get("channels", 32))
        k = int(cfg.get("kernel_size", 3))
        in_channels = 3
        out_channels = 3
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        # --- Encoder ---
        self.enc1 = ConvBlock(in_channels, c1)  # H
        self.down1 = Down(c1, c2)               # H/2

        self.enc2 = ConvBlock(c2, c2)
        self.down2 = Down(c2, c3)               # H/4

        self.enc3 = ConvBlock(c3, c3)
        self.down3 = Down(c3, c4)               # H/8

        # --- Bottleneck ("горлышко") — самое низкое разрешение ---
        self.bottleneck = ConvBlock(c4, c4)

        # --- Decoder ---
        self.up3 = Up(in_channels=c4, skip_channels=c3, out_channels=c3)  # H/4
        self.up2 = Up(in_channels=c3, skip_channels=c2, out_channels=c2)  # H/2
        self.up1 = Up(in_channels=c2, skip_channels=c1, out_channels=c1)  # H

        self.out = nn.Conv2d(c1, out_channels, kernel_size=k, padding=1, bias=True)

    def forward(self, x):
        # --- Encoder (сохраняем skip'ы) ---
        s1 = self.enc1(x)
        x = self.down1(s1)
        s2 = self.enc2(x)

        x = self.down2(s2)
        s3 = self.enc3(x)

        x = self.down3(s3)

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # --- Decoder ---
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        # --- Clean ---
        return self.out(x)