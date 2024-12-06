import numpy as np
import torch

def normal(loc=0.0, scale=1.0, size=None, dtype=None, device=None):
    return torch.randn(size, dtype=dtype, device=device) * scale + loc

def uniform(a=0.0, b=1.1, size=None, dtype=None, device=None):
    size = size if size is not None else ()
    return torch.rand(size, dtype=dtype, device=device) * (b - a) + a