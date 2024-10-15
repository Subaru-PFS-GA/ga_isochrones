import numpy as np
import tensorflow as tf

int32 = tf.int32
int64 = tf.int64
float32 = tf.float32
float64 = tf.float64

nan = np.nan,
newaxis = None,

tensor = tf.convert_to_tensor

def cpu(data):
    return data.numpy()

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
full = tf.fill
zeros_like = tf.zeros_like
ones_like = tf.ones_like

def full_like(data, value):
    return tf.fill(data.shape, value)

empty_like = tf.zeros_like

reshape = tf.reshape
stack = tf.stack
concat = tf.concat

where = tf.where

def tile(data, reps):
    if len(reps) < len(data.shape):
        reps = (1,) * (len(data.shape) - len(reps)) + reps
    return tf.tile(data, reps)

searchsorted = tf.searchsorted
