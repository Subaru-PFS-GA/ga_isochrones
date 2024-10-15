import numpy as np

from ..util.astro import *
from ..util.data import *

def sdss_to_hsc(sdss_g, sdss_r, sdss_i, sdss_z):
    """
    Convert SDSS photometry to HSC using formulae from Komiyama et al. ApJ 853 29K (2018)
    """

    # https://hsc.mtk.nao.ac.jp/pipedoc/pipedoc_7/colorterms.html
    # https://hsc.mtk.nao.ac.jp/pipedoc/pipedoc_8/colorterms.html
    # https://python.hotexamples.com/site/file?hash=0x84cbf64609a6d47c06c3209b08b3186662df8606a83b9cab81665be727fabc57

    # hscPipe 4 colorterms
    hsc_g = sdss_g - 0.00816446 - 0.08366937 * (sdss_g - sdss_r) - 0.00726883 * (sdss_g - sdss_r)**2
    hsc_r = sdss_r + 0.00231810 + 0.01284177 * (sdss_r - sdss_i) - 0.03068248 * (sdss_r - sdss_i)**2
    hsc_i = sdss_i + 0.00130204 - 0.16922042 * (sdss_i - sdss_z) - 0.01374245 * (sdss_i - sdss_z)**2

    # hscPipe 8 colorterms
    # hsc_g = sdss_g - 0.009777 - 0.077235 * (sdss_g - sdss_r) - 0.013121 * (sdss_g - sdss_r)**2
    # hsc_r = sdss_r - 0.000711 - 0.006847 * (sdss_r - sdss_i) - 0.035110 * (sdss_r - sdss_i)**2
    # hsc_i = sdss_i + 0.000357 - 0.153290 * (sdss_i - sdss_z) - 0.009277 * (sdss_i - sdss_z)**2

    return { 'hsc_g': hsc_g, 'hsc_r': hsc_r, 'hsc_i': hsc_i }