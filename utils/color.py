import numpy as np
import torch
from einops import rearrange


def sort_palette(rgbs, palette_rgb, bg=None, K=384000000):
    N, M = rgbs.shape[0], palette_rgb.shape[0]
    if N * M > K:
        K = K // M
        rgbs = rgbs[torch.randperm(N)[:K]]
    dist = np.linalg.norm(rearrange(rgbs, 'N C -> N 1 C') - rearrange(palette_rgb, 'P C -> 1 P C'), axis=-1)
    dist = np.bincount(np.argmin(dist, axis=-1))
    if bg is not None:
        idx = np.argmin(np.linalg.norm(palette_rgb - bg, axis=-1))
        dist[idx] = np.iinfo(dist.dtype).max
    return palette_rgb[np.argsort(dist)]
