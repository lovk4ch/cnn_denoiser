import torch
import torch.nn.functional as F


def _blur(noise, k=5):
    # простой box blur без зависимостей
    w = torch.ones(1, 1, k, k, device=noise.device) / (k * k)
    # depthwise: применяем к каждому каналу отдельно
    c = noise.shape[1]
    w = w.repeat(c, 1, 1, 1)
    return F.conv2d(noise, w, padding=k//2, groups=c)

def add_shot_noise(x, lam=30.0):
    # имитация фотонов: чем меньше света, тем хуже относительный шум
    # x in [0,1], lam регулирует “количество фотонов”
    xq = (x * lam).clamp(min=0)
    y = torch.poisson(xq) / lam
    return y.clamp(0,1)

def add_gaussian_noise(x):
    return x + torch.rand_like(x) * 0.4 - 0.2

def add_canon_like_noise(x):
    # x: [B,3,H,W] in [0,1]
    B, C, H, W = x.shape
    assert C == 3

    # 1) яркость (luma) для маски теней
    # (приближённо, можно и среднее по каналам)
    luma = (0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3])

    # маска: в тенях ближе к 1, в светах ближе к 0
    # степень регулирует “насколько резко” шум растёт в тенях
    shadow = (1.0 - luma).clamp(0, 1)
    shadow = shadow ** torch.empty(1, device=x.device).uniform_(1.2, 2.2).item()

    # 2) базовый sigma (будет меняться от патча к патчу)
    sigma = torch.empty(1, device=x.device).uniform_(0.05, 0.19).item()

    # 3) мелкий “белый” шум
    n_white = torch.randn_like(x)

    # 4) “крупинка” (коррелированный шум)
    n_corr = _blur(torch.randn_like(x), k=int(torch.randint(3, 7, (1,)).item()))
    # нормируем приблизительно
    n_corr = n_corr / (n_corr.std(dim=(2,3), keepdim=True) + 1e-6)
    n_corr = n_corr[..., :x.shape[-2], :x.shape[-1]]

    # 5) смешиваем мелкий и крупный
    mix = torch.empty(1, device=x.device).uniform_(0.15, 0.45).item()
    n = (1 - mix) * n_white + mix * n_corr

    # 6) делаем “цветность”: разные sigma на каналы (chroma)
    # у Canon/компактов часто цветной шум заметнее в тенях
    rgb_scale = torch.tensor([1.0, 1.0, 1.0], device=x.device).view(1,3,1,1)
    rgb_scale = rgb_scale * torch.tensor(
        [torch.empty(1, device=x.device).uniform_(0.9, 1.3).item(),
         torch.empty(1, device=x.device).uniform_(0.9, 1.2).item(),
         torch.empty(1, device=x.device).uniform_(1.0, 1.4).item()],
        device=x.device
    ).view(1,3,1,1)

    # 7) итог: шум сильнее в тенях + немного остаётся в светах
    strength = sigma * (0.35 + 0.65 * shadow)  # не ноль в светах
    noisy = x + n * strength * rgb_scale

    '''
    if torch.rand(1) < 0.3:  # не всегда!
        noisy = add_shot_noise(noisy)
    '''

    return noisy.clamp(0, 1)