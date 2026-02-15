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

def blotch_noise_like(x, scale_range=(0.09, 0.35), blur_k_range=(0, 5)):
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
    sigma = float(torch.empty(1, device=device).uniform_(0.1, 0.15).item())

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

def add_luminance_iso_noise(
    x,
    sigma_range=(0.005, 0.06),          # сила luma-шума (основной)
    corr_prob=0.35,                     # вероятность сделать luma шум коррелированным (мелко-средним)
    corr_k_range=(3, 11),               # ядро для корреляции
    corr_sigma_range=(0.6, 2.2),        # sigma гаусс-блюра для корреляции
    shadow_boost_range=(0.7, 2.0),      # насколько сильнее шум в тенях (степень)
    shadow_mix_range=(0.35, 0.75),      # доля "усиления в тенях"
    chroma_prob=0.20,                   # вероятность добавить chroma шум (редко)
    chroma_strength_range=(0.15, 0.45), # доля от sigma (насколько слабее chroma относительно luma)
    chroma_corr_prob=0.25,              # вероятность коррелировать chroma шум
    quant_prob=0.01,                    # вероятность квантования (как артефакт камеры/JPEG)
    quant_levels_range=(96, 200),       # уровни квантования
    clamp=True
):
    """
    ISO-подобный шум:
      - основной шум по яркости (luminance), почти монохромный
      - сильнее в тенях
      - иногда коррелированный
      - редко добавляется слабый chroma (CbCr) шум
      - опционально квантование

    x: [B,3,H,W] in [0,1]
    return: [B,3,H,W] in [0,1]
    """
    assert x.dim() == 4 and x.size(1) == 3
    device = x.device

    # --- 1) Переводим в YCbCr, чтобы шумить в яркости ---
    ycc = _rgb_to_ycbcr(x)          # [0,1]
    Y = ycc[:, 0:1]                # [B,1,H,W]
    CbCr = ycc[:, 1:3]             # [B,2,H,W]

    # --- 2) Карта теней: в темных областях шум сильнее ---
    # shadow in [0,1], 1 = тень
    shadow = (1.0 - Y).clamp(0, 1)
    p = float(torch.empty(1, device=device).uniform_(*shadow_boost_range).item())
    shadow = shadow ** p

    mix = float(torch.empty(1, device=device).uniform_(*shadow_mix_range).item())
    sigma = float(torch.empty(1, device=device).uniform_(*sigma_range).item())

    # strength_map: базовый sigma + усиление в тенях
    # (1-mix) даёт "везде немного", mix добавляет "в тенях больше"
    strength = sigma * ((1.0 - mix) + mix * shadow)

    # --- 3) Luma-шум: белый или коррелированный ---
    n = torch.randn_like(Y)

    if torch.rand(1, device=device) < corr_prob:
        k = int(torch.randint(corr_k_range[0], corr_k_range[1] + 1, (1,), device=device).item())
        s = float(torch.empty(1, device=device).uniform_(*corr_sigma_range).item())
        # используем твою _gauss_blur (она работает на [B,C,H,W])
        n = _gauss_blur(n, k=k, sigma=s)
        n = n / (n.std(dim=(2, 3), keepdim=True) + 1e-6)

    # добавляем шум только в Y
    Y_noisy = (Y + n * strength).clamp(0, 1) if clamp else (Y + n * strength)

    # --- 4) Редко добавляем chroma шум (обычно камера его подавляет) ---
    if torch.rand(1, device=device) < chroma_prob:
        # chroma слабее luma: chroma_sigma = sigma * factor
        factor = float(torch.empty(1, device=device).uniform_(*chroma_strength_range).item())
        chroma_sigma = sigma * factor

        n_c = torch.randn_like(CbCr)

        if torch.rand(1, device=device) < chroma_corr_prob:
            # лёгкая корреляция chroma-шума (обычно слабее, чем luma)
            k = int(torch.randint(3, 9, (1,), device=device).item())
            s = float(torch.empty(1, device=device).uniform_(0.5, 1.6).item())
            n_c = _gauss_blur(n_c, k=k, sigma=s)
            n_c = n_c / (n_c.std(dim=(2, 3), keepdim=True) + 1e-6)

        # добавляем chroma шум (меньше амплитуда, можно тоже слегка усилить в тенях)
        # слегка коррелируем с тенями, но слабее чем Y
        strength_c = chroma_sigma * (0.65 + 0.35 * shadow)
        CbCr_noisy = (CbCr + n_c * strength_c).clamp(0, 1) if clamp else (CbCr + n_c * strength_c)
    else:
        CbCr_noisy = CbCr

    # --- 5) Собираем обратно RGB ---
    ycc_noisy = torch.cat([Y_noisy, CbCr_noisy], dim=1)
    rgb = _ycbcr_to_rgb(ycc_noisy)

    # --- 6) Квантование как артефакт (редко) ---
    if torch.rand(1, device=device) < quant_prob:
        q = int(torch.randint(quant_levels_range[0], quant_levels_range[1] + 1, (1,), device=device).item())
        rgb = torch.round(rgb * q) / q

    if clamp:
        rgb = rgb.clamp(0, 1)

    return rgb

def add_noise(x):
    """
    Noise-роутер.
    x: [B,3,H,W] in [0,1]
    return: [B,3,H,W] in [0,1]
    """
    assert x.dim() == 4 and x.size(1) == 3
    device = x.device

    # Вероятности режимов
    p_iso  = 0.82   # основной ISO-like luminance noise
    p_mix  = 0.13   # твой add_noise_mix (кодек/артефакты/пятна)
    p_hard = 0.05   # жёсткий режим (ISO + mix + усиление артефактов)

    r = float(torch.rand(1, device=device).item())

    # --- 1) ISO-luma (реалистичный шум камеры) ---
    if r < p_iso:
        x = add_luminance_iso_noise(
            x,
            sigma_range=(0.01, 0.06),
            corr_prob=0.35,
            chroma_prob=1.00,              # цветной шум редко
            chroma_strength_range=(0.12, 0.40),
        )
        return x.clamp(0, 1)

    # --- 2) Mix-artifacts (реже) ---
    if r < p_iso + p_mix:
        # важно: add_noise_mix ожидает x в [0,1]
        x = add_noise_mix(x)
        return x.clamp(0, 1)

    # --- 3) Hard-artifacts (редко, чтобы сеть не "замылилась" на лёгком) ---
    # ISO + mix + доп. пятна/корреляция
    x = add_luminance_iso_noise(
        x,
        sigma_range=(0.03, 0.14),          # сильнее ISO
        corr_prob=0.05,                    # чаще коррелируем
        corr_k_range=(5, 21),
        corr_sigma_range=(1.0, 4.0),
        chroma_prob=0.30,                  # цвет может полезть сильнее
        chroma_strength_range=(0.20, 0.60),
        quant_levels_range=(48, 140)
    )

    # изредка mix как "кодек/грязь"
    if torch.rand(1, device=device) < 0.1:
        x = add_noise_mix(x)

    # иногда добавим дополнительные крупные пятна (blotch) поверх
    if torch.rand(1, device=device) < 0.1:
        n_b = blotch_noise_like(x, scale_range=(0.10, 0.35), blur_k_range=(0, 7))
        sigma_b = float(torch.empty(1, device=device).uniform_(0.03, 0.10).item())
        x = (x + sigma_b * n_b).clamp(0, 1)

    return x.clamp(0, 1)