import numpy as np

int32 = np.int32
int64 = np.int64
float32 = np.float32
float64 = np.float64

nan = np.nan
newaxis = None

def tensor(data, dtype=None, device=None):
    return np.array(data, dtype=dtype)

def cpu(data):
    return data

shape = np.shape
size = np.size
ndim = np.ndim

def arange(*data, dtype=None, device=None):
    return np.arange(*data, dtype=dtype)

def linspace(*data, dtype=None, device=None):
    return np.linspace(*data, dtype=dtype)

def zeros(shape, dtype=None, device=None):
    return np.zeros(shape, dtype=dtype)

def ones(shape, dtype=None, device=None):
    return np.ones(shape, dtype=dtype)

def one_hot(indices, depth, dtype=None):
    return np.stack([indices == i for i in range(depth)], axis=-1).astype(dtype)

def eye(N, M=None, dtype=None, device=None):
    return np.eye(N, M, dtype=dtype)

def empty(shape, dtype=None, device=None):
    return np.empty(shape, dtype=dtype)

def full(shape, fill_value, dtype=None, device=None):
    return np.full(shape, fill_value, dtype=dtype)

zeros_like = np.zeros_like
ones_like = np.ones_like
full_like = np.full_like
empty_like = np.empty_like

reshape = np.reshape
stack = np.stack
concat = np.concatenate

where = np.where
tile = np.tile
searchsorted = np.searchsorted
# gather