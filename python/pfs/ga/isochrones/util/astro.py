import numpy as np
from tensorflow.python.framework.ops import EagerTensor

from ..util.astro import *
from ..util.data import *

def sdss_to_hsc(sdss_g, sdss_r, sdss_i, sdss_z):
    """
    Convert SDSS photometry to HSC using formulae from Komiyama et al. ApJ 853 29K (2018)
    """

    hsc_g = sdss_g - 0.00816446 - 0.08366937 * (sdss_g - sdss_r) - 0.00726883 * (sdss_g - sdss_r)**2
    hsc_i = sdss_i + 0.00130204 - 0.16922042 * (sdss_i - sdss_z) - 0.01374245 * (sdss_i - sdss_z)**2
    return hsc_g, hsc_i
