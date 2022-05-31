import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.framework.ops import EagerTensor

def mag_to_flux(mag):
    """Convert AB mags to flux"""
    flux = 10**(-0.4 * (mag + 48.6))
    return flux

def flux_to_mag(flux):
    """Convert flux to AB mags"""
    if isinstance(flux, EagerTensor):
        mag = tf.constant(-2.5, dtype=flux.dtype) * \
            tf.math.log(flux) / \
            tf.math.log(tf.constant(10.0, dtype=flux.dtype)) - tf.constant(48.6, dtype=flux.dtype)
        return mag
    else:
        mag = -2.5 * np.log10(flux) - 48.6
        return mag

def sdss_to_hsc(sdss_g, sdss_r, sdss_i, sdss_z):
    hsc_g = sdss_g - 0.00816446 - 0.08366937 * (sdss_g - sdss_r) - 0.00726883 * (sdss_g - sdss_r)**2
    hsc_i = sdss_i + 0.00130204 - 0.16922042 * (sdss_i - sdss_z) - 0.01374245 * (sdss_i - sdss_z)**2
    return hsc_g, hsc_i

def dm_to_kpc(dm):
    if isinstance(dm, EagerTensor):
        return tf.constant(1e-3, dtype=dm.dtype) * \
            10**((dm + tf.constant(5, dtype=dm.dtype)) / tf.constant(5, dtype=dm.dtype))
    else:
        return 1e-3 * 10**((dm + 5.0) / 5.0)

def kpc_to_dm(kpc):
    if isinstance(kpc, EagerTensor):
        return tf.constant(5, dtype=kpc.dtype) * \
                           tf.math.log(kpc * 1e3) / tf.math.log(tf.constant(10, dtype=kpc.dtype)) - \
                           tf.constant(5, dtype=kpc.dtype)
    else:
        return 5 * np.log(kpc * 1e3) / np.log(10) - 5