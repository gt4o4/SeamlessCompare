import numpy as np
import torch
from einops import rearrange


def sort_palette(rgbs, palette_rgb):
    dist = rearrange(rgbs, 'N C -> N 1 C') - rearrange(palette_rgb, 'P C -> 1 P C')
    dist = np.linalg.norm(dist, axis=-1)
    dist = np.argmin(dist, axis=-1)
    dist = np.argsort(np.bincount(dist))

    # bg = np.ones(3) if dataset.white_bg else np.zeros(3)
    # palette_rgb = [tuple(a.tolist()) for a in palette_rgb[dist.cpu().numpy()] if not np.allclose(a, bg)]
    # palette_rgb.append(tuple(bg.tolist()))
    palette_rgb = [tuple(a) for a in palette_rgb[dist].tolist()]
    return torch.Tensor(palette_rgb)