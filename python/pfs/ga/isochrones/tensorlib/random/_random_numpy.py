import numpy as np
import scipy

def normal(loc=0.0, scale=1.0, size=None, dtype=None, device=None):
    return np.random.normal(loc=loc, scale=scale, size=size).astype(dtype)

def uniform(a=0.0, b=1.0, size=None, dtype=None, device=None):
    return np.random.uniform(a, b, size).astype(dtype)