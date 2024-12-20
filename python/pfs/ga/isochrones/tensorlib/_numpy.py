import numpy as np
import scipy

int32 = np.int32
int64 = np.int64
float32 = np.float32
float64 = np.float64

nan = np.nan
inf = np.inf
neginf = -np.inf
newaxis = None

def tensor(data, dtype=None, device=None):
    return np.array(data, dtype=dtype)

def cpu(data):
    return data

def cast(data, dtype):
    return data.astype(dtype)

def istensor(data):
    return isinstance(data, np.ndarray)

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

atleast_1d = np.atleast_1d
atleast_2d = np.atleast_2d

reshape = np.reshape
stack = np.stack
concat = np.concatenate

where = np.where
repeat = np.repeat
tile = np.tile

def gather(data, indices, axis=None):
    indices = np.array(indices)
    axis = axis if axis is not None else 0

    if axis < 0:
        axis = data.ndim + axis

    # Output shape will be the index shape + the input shape except for the axis dimension
    output_shape = list(indices.shape) + list(data.shape[:axis]) + list(data.shape[axis + 1:])

    idx = indices.flatten()

    if axis == 0:
        idx = (idx,)
    elif axis > 0:
        idx = axis * (slice(None),) + (idx,)
    elif axis < 0:
        idx = (Ellipsis, idx,) + (axis + 1) * (slice(None),)
    
    return np.reshape(data[idx], output_shape)

def gather_nd(data, indices):
    indices = tuple(np.array(indices).T)
    return data[indices]

searchsorted = np.searchsorted

abs = np.abs
exp = np.exp
log = np.log
log10 = np.log10

isnan = np.isnan
isinf = np.isinf
isneginf = np.isneginf
isposinf = np.isposinf

def any(data, axis=None):
    return np.any(data, axis=axis)

def all(data, axis=None):
    return np.all(data, axis=axis)

def sum(data, axis=None):
    return np.sum(data, axis=axis)

def mean(data, axis=None):
    return np.mean(data, axis=axis)

def logsumexp(data, axis=None):
    return scipy.special.logsumexp(data, axis=axis)

def min(data, axis=None):
    return np.min(data, axis=axis)

def max(data, axis=None):
    return np.max(data, axis=axis)

def count_nonzero(data, axis=None, dtype=None):
    res = np.count_nonzero(data, axis=axis)
    if dtype is not None:
        res = res.astype(dtype)
    return res