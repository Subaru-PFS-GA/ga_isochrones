import numpy as np
import tensorflow as tf

# Initialize eager mode and dynamic memory allocation

# tf.enable_v2_behavior()
gpus = tf.config.list_physical_devices('GPU') 
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
try:
    tf.compat.v1.enable_eager_execution()
except ValueError:
    pass

# Define the tensorlib API

int32 = tf.int32
int64 = tf.int64
float32 = tf.float32
float64 = tf.float64

nan = np.nan
inf = np.inf
neginf = -np.inf
newaxis = None

def tensor(data, dtype=None, device=None):
    return tf.convert_to_tensor(data, dtype=dtype)

def cpu(data):
    return data.numpy()

cast = tf.cast

def shape(data):
    return data.shape

size = tf.size
ndim = tf.rank

arange = tf.range
linspace = tf.linspace

zeros = tf.zeros
ones = tf.ones
one_hot = tf.one_hot
eye = tf.eye
empty = tf.zeros

def full(shape, fill_value, dtype=None, device=None):
    return tf.fill(shape, tf.convert_to_tensor(fill_value, dtype=dtype))

zeros_like = tf.zeros_like
ones_like = tf.ones_like

def full_like(data, value):
    return tf.fill(data.shape, value)

empty_like = tf.zeros_like

atleast_1d = tf.experimental.numpy.atleast_1d
atleast_2d = tf.experimental.numpy.atleast_2d

reshape = tf.reshape
stack = tf.stack
concat = tf.concat

where = tf.where

def repeat(data, repeats, axis):
    return tf.repeat(data, repeats, axis=axis)

def tile(data, repeats):
    if len(repeats) < len(data.shape):
        repeats = (1,) * (len(data.shape) - len(repeats)) + repeats
    return tf.tile(data, repeats)

def gather(data, indices, axis=None):
    return tf.gather(data, indices, axis=axis)

def gather_nd(data, indices):
    return tf.gather_nd(data, indices)

searchsorted = tf.searchsorted

abs = tf.math.abs
exp = tf.math.exp
log = tf.math.log

def log10(x):
    return tf.math.log(x) / tf.math.log(10.0)

isnan = tf.math.is_nan
isinf = tf.math.is_inf

def isneginf(data):
    return tf.math.is_inf(data) & (data < 0)

def isposinf(data):
    return tf.math.is_inf(data) & (data > 0)

def any(data, axis=None):
    return tf.reduce_any(data, axis=axis)

def all(data, axis=None):
    return tf.reduce_all(data, axis=axis)

def sum(data, axis=None):
    return tf.reduce_sum(data, axis=axis)

def mean(data, axis=None):
    return tf.reduce_mean(data, axis=axis)

def logsumexp(data, axis=None):
    return tf.reduce_logsumexp(data, axis=axis)

def min(data, axis=None):
    return tf.reduce_min(data, axis=axis)

def max(data, axis=None):
    return tf.reduce_max(data, axis=axis)