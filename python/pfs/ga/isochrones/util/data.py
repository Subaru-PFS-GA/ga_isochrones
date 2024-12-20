import h5py
import numpy as np
from numbers import Number

from .. import tensorlib as tt

#region HDF5 utility functions

def to_numpy(data):
    if tt.istensor(data):
        data = tt.cpu(data)
    return data

def merge_tensors(a, b, axis=0):
    if tt.istensor(a) and tt.istensor(b):
        r = tt.stack([a, b], axis=axis)
    elif isinstance(a, Number) and isinstance(b, Number):
        r = a
    else:
        raise TypeError("Merged types must match.")
    return r

def merge_dict(a, b, axis=0):
    r = {}
    for i, k in enumerate(a):
        r[k] = merge_tensors(a[k], b[k], axis=axis)
    return r
                
def merge_dict_of_lists(a, b, axis=0):
    r = {}
    for i, k in enumerate(a):
        r[k] = []
        for j, (aa, bb) in enumerate(zip(a[k], b[k])):
            if aa is not None and bb is not None:
                rr = merge_tensors(aa, bb, axis=axis)
                r[k].append(rr)
    return r

def save_dict_h5(grp, d, s=None):
    if d is not None:
        for k in d:
            save_data_h5(grp, k, d[k], s=s)

def save_data_h5(grp, name, data, maxshape=None, chunks=None, s=None):
    if data is not None:
        if s is not None:
            data = to_numpy(data[s])
        else:
            data = to_numpy(data)
        ds = grp.create_dataset(name, data=data, maxshape=maxshape, chunks=chunks)
        return ds
    else:
        return None

def append_data_h5(grp, name, data):
    if data is not None:
        data = to_numpy(data)
        ds = grp[name]
        
        # Find dimension that needs to be increased
        newsize = None
        newshape = []
        newslice = []
        for i in range(len(ds.maxshape)):
            if ds.maxshape[i] is None:
                if newsize is not None:
                    raise Exception('Only one dimension can be increased.')
                newsize = ds.shape[i] + 1
                newshape.append(newsize)
                newslice.append(ds.shape[i])
            else:
                newshape.append(ds.shape[i])
                newslice.append(slice(None))
        ds.resize(tuple(newshape))
        ds[tuple(newslice)] = data

def load_dict_h5(grp, format='tt'):
    # Load a dictionary from a HDF5 file, either as numpy arrays or gpu tensors
    d = {}
    for k in grp:
        d[k] = load_data_h5(grp, k, format=format)
    return d

def load_data_h5(grp, name, format='tt'):
    # load an array from a HDF5 file as a numpy array or a gpu tensor
    if name in grp:
        data = grp[name][()]
        if format == 'tt':
            data = tt.tensor(data)
        elif format == 'np':
            pass
        else:
            raise NotImplementedError()
        return data
    else:
        return None
