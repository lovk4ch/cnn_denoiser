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
    x = x + torch.rand_like(x) * 0.6 - 0.3
    return x.clamp(-1,1)

def blotch_noise_like(x, scale_range=(0.06, 0.18), blur_k_range=(0, 5)):
    """
    x: [B,3,H,W] (нужен только device/shape)
    Возвращает blotch-компоненту шума [B,3,H,W] примерно N(0,1) по масштабу.
    """
    B, C, H, W = x.shape
    device = x.device

    # 1) создаём шум на низком разрешении
    s = float(torch.empty(1, device=device).uniform_(*scale_range).item())  # доля от размера
    h2 = max(8, int(H * s))
    w2 = max(8, int(W * s))

    n_lr = torch.randn(B, C, h2, w2, device=device)

    # 2) апсемплим до исходного размера (получаем крупные пятна)
    n = F.interpolate(n_lr, size=(H, W), mode="bilinear", align_corners=False)

    # 3) опционально слегка сгладим (делает пятна более "естественными")
    k = int(torch.randint(blur_k_range[0], blur_k_range[1] + 1, (1,), device=device).item())
    if k >= 3 and (k % 2 == 1):
        # depthwise blur через avgpool (быстро и достаточно)
        # (можно заменить на gauss blur, но для шума этого хватает)
        pad = k // 2
        n = F.avg_pool2d(n, kernel_size=k, stride=1, padding=pad)

    # 4) нормируем по каждой картинке/каналу
    n = n / (n.std(dim=(2,3), keepdim=True) + 1e-6)
    return n

def add_noise_mix(x):
    # x: [B,3,H,W] in [0,1]
    device = x.device

    # мелкий
    n_white = torch.randn_like(x)

    # коррелированный (мелко-средний)
    n_corr = torch.randn_like(x)
    n_corr = F.avg_pool2d(n_corr, kernel_size=5, stride=1, padding=2)
    n_corr = n_corr / (n_corr.std(dim=(2,3), keepdim=True) + 1e-6)

    # крупные пятна
    n_blotch = blotch_noise_like(x)

    # веса смеси (можно фиксировать или рандомить)
    a = float(torch.empty(1, device=device).uniform_(0.40, 0.70).item())  # white
    b = float(torch.empty(1, device=device).uniform_(0.10, 0.35).item())  # corr
    c = float(torch.empty(1, device=device).uniform_(0.10, 0.35).item())  # blotch
    s = a + b + c
    a, b, c = a/s, b/s, c/s

    n = a*n_white + b*n_corr + c*n_blotch

    # сила шума (пример)
    sigma = float(torch.empty(1, device=device).uniform_(0.01, 0.06).item())

    noisy = (x + sigma * n).clamp(0, 1)

    # квантование (как артефакт)
    if torch.rand(1, device=device) < 0.5:
        q = int(torch.randint(64, 160, (1,), device=device).item())
        noisy = torch.round(noisy * q) / q

    return noisy

# для деградации (low-res)
def _gauss_blur(x, k=5, sigma=1.2):
    # x: [B,C,H,W]
    if k % 2 == 0:
        k += 1
    device = x.device
    coords = torch.arange(k, device=device) - k // 2
    g = torch.exp(-(coords**2) / (2*sigma*sigma))
    g = g / g.sum()
    # separable
    g1 = g.view(1,1,k,1)
    g2 = g.view(1,1,1,k)
    C = x.shape[1]
    x = F.conv2d(x, g1.repeat(C,1,1,1), padding=(k//2,0), groups=C)
    x = F.conv2d(x, g2.repeat(C,1,1,1), padding=(0,k//2), groups=C)
    return x

def _down_up(x, scale=0.5, mode="bilinear"):
    B,C,H,W = x.shape
    h2 = max(8, int(H*scale))
    w2 = max(8, int(W*scale))
    y = F.interpolate(x, size=(h2,w2), mode=mode, align_corners=False if mode=="bilinear" else None)
    y = F.interpolate(y, size=(H,W), mode=mode, align_corners=False if mode=="bilinear" else None)
    return y

def _rgb_to_ycbcr(x):
    # x [0,1]
    r,g,b = x[:,0:1], x[:,1:2], x[:,2:3]
    y  = 0.299*r + 0.587*g + 0.114*b
    cb = (b - y) * 0.564 + 0.5
    cr = (r - y) * 0.713 + 0.5
    return torch.cat([y,cb,cr], dim=1)

def _ycbcr_to_rgb(x):
    y,cb,cr = x[:,0:1], x[:,1:2]-0.5, x[:,2:3]-0.5
    r = y + 1.403*cr
    b = y + 1.773*cb
    g = y - 0.714*cr - 0.344*cb
    return torch.cat([r,g,b], dim=1)

def add_noise(x):
    """
    x: [B,3,H,W] in [0,1]
    return: degraded in [0,1]
    """
    assert x.dim() == 4 and x.size(1) == 3
    B,_,H,W = x.shape
    device = x.device

    # 3) шум: мелкий + коррелированный, сильнее в тенях, чуть цветной
    luma = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).clamp(0, 1)
    shadow = (1.0 - luma).clamp(0, 1)
    shadow = shadow ** float(torch.empty(1, device=device).uniform_(1.2, 2.2).item())
    mix = float(torch.empty(1, device=device).uniform_(.45, 1.15).item())

    n_white = torch.randn_like(x)
    n_corr = torch.randn_like(x)
    # делаем "пятнистость"
    k2 = int(torch.randint(5, 15, (1,), device=device).item())
    s2 = float(torch.empty(1, device=device).uniform_(0.8, 2.4).item())
    n_corr = _gauss_blur(n_corr, k=k2, sigma=s2)
    n_corr = n_corr / (n_corr.std(dim=(2, 3), keepdim=True) + 1e-6)

    n = (1 - mix) * n_white + mix * n_corr

    # цветность шума (слегка разные каналы)
    rgb_scale = torch.tensor(
        [float(torch.empty(1, device=device).uniform_(0.9, 1.3).item()),
         float(torch.empty(1, device=device).uniform_(0.9, 1.2).item()),
         float(torch.empty(1, device=device).uniform_(1.0, 1.4).item())],
        device=device
    ).view(1, 3, 1, 1)

    # общий уровень
    sigma = float(torch.empty(1, device=device).uniform_(0.01, 0.05).item())
    # в тенях сильнее
    strength = sigma * (0.35 + 0.65 * shadow)
    x = x + n * strength * rgb_scale

    # 4) лёгкое квантование (бандинг/“компрессия” без JPEG)
    if torch.rand(1, device=device) < 0.5:
        q = int(torch.randint(64, 160, (1,), device=device).item())  # чем меньше, тем хуже
        x = torch.round(x * q) / q

    x = add_noise_mix(x)
    return x.clamp(0, 1)

def add_compression(x):
    """
    x: [B,3,H,W] in [0,1]
    return: degraded in [0,1]
    """
    assert x.dim() == 4 and x.size(1) == 3
    B,_,H,W = x.shape
    device = x.device

    x = (x + 1) / 2

    # 1) лёгкая потеря деталей: blur + down/up (имитирует ресайз+компрессию)
    if torch.rand(1, device=device) < 0.7:
        k = int(torch.randint(3, 9, (1,), device=device).item())
        sigma = float(torch.empty(1, device=device).uniform_(0.6, 1.8).item())
        x = _gauss_blur(x, k=k, sigma=sigma)

    if torch.rand(1, device=device) < 0.7:
        scale = float(torch.empty(1, device=device).uniform_(0.45, 0.85).item())
        x = _down_up(x, scale=scale, mode="bilinear")

    # 2) хрома-сабсэмплинг (как у JPEG/видео): портит цвет, но не яркость
    if torch.rand(1, device=device) < 0.9:
        ycc = _rgb_to_ycbcr(x)
        Y  = ycc[:,0:1]
        CbCr = ycc[:,1:3]
        # сильнее сжимаем цвет
        scale_c = float(torch.empty(1, device=device).uniform_(0.25, 0.6).item())
        CbCr = _down_up(CbCr, scale=scale_c, mode="bilinear")
        ycc = torch.cat([Y, CbCr], dim=1)
        x = _ycbcr_to_rgb(ycc)

    x = x * 2 - 1
    return x.clamp(-1,1)