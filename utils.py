def to_image(n, bias=False):
    n = n.detach().cpu().squeeze(0)
    if bias:
        m = n.abs().amax(dim=(1, 2), keepdim=True) + 1e-8
        n = (n / m) * 0.5 + 0.5
    return n