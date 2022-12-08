import os

import tensorflow.compat.v2 as tf

from .test_base import TestBase
from pfs.ga.isochrones.constants import Constants
from pfs.ga.isochrones import Dartmouth

class TestDartmouth(TestBase):
    def load_grid(self):
        grid = Dartmouth()
        grid.load(os.path.join(self.ISOCHRONES_DATA, 'dartmouth/import/afep0_cfht_sdss_hsc/isochrones.h5'))
        return grid

    def test_interp3d_problematic(self):
        grid = self.load_grid()

        iso_M_ini = tf.linspace(tf.constant(0.6, dtype=Constants.TF_PRECISION), tf.constant(0.8, dtype=Constants.TF_PRECISION), 20)
        iso_Fe_H = tf.fill(iso_M_ini.shape, tf.constant(-2.0, dtype=Constants.TF_PRECISION))
        iso_log_t = tf.fill(iso_M_ini.shape, tf.constant(10.160, dtype=Constants.TF_PRECISION))    # 10.175

        iso = grid.interp3d(iso_Fe_H, iso_log_t, iso_M_ini, values=[grid.values['hsc_g'], grid.values['hsc_i']])