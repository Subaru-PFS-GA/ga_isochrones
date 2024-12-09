import numpy as np
import torch

int32 = torch.int32
int64 = torch.int64
float32 = torch.float32
float64 = torch.float64

nan = np.nan
inf = np.inf
neginf = -np.inf
newaxis = None

def tensor(data, dtype=None, device=None):
    return torch.tensor(data, dtype=dtype, device=device, requires_grad=False)

def cast(data, dtype):
    return data.to(dtype)

def cpu(data):
    return data.detach().cpu().numpy()

def istensor(data):
    return isinstance(data, torch.Tensor)

def shape(data):
    return data.shape

def size(data):
    return data.numel()

def ndim(data):
    return data.dim()

arange = torch.arange
linspace = torch.linspace

zeros = torch.zeros
ones = torch.ones
one_hot = torch.nn.functional.one_hot
eye = torch.eye
empty = torch.empty
full = torch.full
zeros_like = torch.zeros_like
ones_like = torch.ones_like
full_like = torch.full_like
empty_like = torch.empty_like

atleast_1d = torch.atleast_1d
atleast_2d = torch.atleast_2d

def reshape(data, shape):
    return data.reshape(shape)

def stack(data, axis=0):
    return torch.stack(data, dim=axis)

def concat(data, axis=0):
    return torch.cat(data, dim=axis)

where = torch.where
repeat = torch.repeat_interleave
tile = torch.tile

def gather(data, indices, axis=None):
    indices = torch.tensor(indices, device=data.device)
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
    
    return torch.reshape(data[idx], output_shape)

def gather_nd(data, indices):
    indices = torch.tensor(indices, device=data.device)

    # Rotate the indices to the first dimension, followed by the batch shape
    dims = list(range(-1, indices.ndim - 1))
    indices = tuple(indices.permute(*dims))

    return data[indices]

def gather_nd_safe(data, indices):   
    indices = torch.tensor(indices, device=data.device)

    # Last index should match the shape of the tensor. Do a bound check here and
    # return nan for out-of-bounds indices.
    mask = torch.full(indices.shape[:-1], False, dtype=torch.bool, device=indices.device)
    for i in range(indices.shape[-1]):
        mask |= (indices[..., i] < 0) | (indices[..., i] >= data.shape[i])

    # Replace out of bound indices with zero
    indices = torch.where(mask[..., newaxis], torch.zeros_like(indices), indices)

    # Rotate the indices to the first dimension, followed by the batch shape
    dims = list(range(-1, indices.ndim - 1))
    indices = tuple(indices.permute(*dims))

    res = torch.where(mask, nan, data[indices])
    return res

searchsorted = torch.searchsorted

abs = torch.abs
exp = torch.exp
log = torch.log
log10 = torch.log10

isnan = torch.isnan
isinf = torch.isinf
isneginf = torch.isneginf
isposinf = torch.isposinf

def any(data, axis=None):
    return torch.any(data) if axis is None else torch.any(data, dim=axis)

def all(data, axis=None):
    return torch.all(data) if axis is None else torch.all(data, dim=axis)

def sum(data, axis=None):
    return torch.sum(data) if axis is None else torch.sum(data, dim=axis)

def mean(data, axis=None):
    return torch.mean(data) if axis is None else torch.mean(data, dim=axis)

def logsumexp(data, axis=None):
    dim = axis if axis is not None else tuple(range(data.ndim))
    return torch.logsumexp(data, dim=dim)

def min(data, axis=None):
    if axis is None:
        values = torch.min(data)
    else:
        values, _ = torch.min(data, dim=axis)
    return values

def max(data, axis=None):
    if axis is None:
        values = torch.max(data)
    else:
        values, _ = torch.max(data, dim=axis)
    return values

def count_nonzero(data, axis=None, dtype=None):
    res = torch.count_nonzero(data, dim=axis)
    if dtype is not None:
        res = res.to(dtype)
    return res