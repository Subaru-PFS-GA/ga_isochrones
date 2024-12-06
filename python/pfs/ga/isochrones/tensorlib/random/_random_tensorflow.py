import numpy as np
import tensorflow as tf

def normal(loc=0.0, scale=1.0, size=None, dtype=tf.float32, device=None):
    return tf.random.normal(size, mean=loc, stddev=scale, dtype=dtype)

def uniform(a=0.0, b=1.1, size=None, dtype=tf.float32, device=None):
    size = size if size is not None else ()
    return tf.random.uniform(size, minval=a, maxval=b, dtype=dtype)