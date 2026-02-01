from torchvision.transforms.v2.functional import to_pil_image


def to_image(n, normalize=False):
    n = n.detach().cpu().squeeze(0)
    n = (n + 1) / 2
    n = n.clamp(0, 1)

    if normalize:
        m = n.abs().amax(dim=(1, 2), keepdim=True) + 1e-8
        n = (n / m) * 0.5 + 0.5

    return n

def tensor_to_jpg(n, name, normalize=False):
    n = to_image(n, normalize)
    to_pil_image(n).save(name)