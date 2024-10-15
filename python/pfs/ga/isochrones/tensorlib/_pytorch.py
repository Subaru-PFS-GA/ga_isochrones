import numpy as np
import torch

int32 = torch.int32
int64 = torch.int64
float32 = torch.float32
float64 = torch.float64

nan = np.nan
newaxis = None

tensor = torch.tensor

def cpu(data):
    return data.detach().cpu().numpy()

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

def reshape(data, shape):
    return data.reshape(shape)

stack = torch.stack
concat = torch.concatenate

where = torch.where
tile = torch.tile
searchsorted = torch.searchsorted