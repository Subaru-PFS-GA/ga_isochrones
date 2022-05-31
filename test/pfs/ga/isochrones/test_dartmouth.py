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

    def test_find_EEP_problematic(self):
        grid = self.load_grid()

        Fe_H = tf.convert_to_tensor([-0.97249065, -2.35758372, -2.39997938, -2.45752465, -0.99786944,
                                     -2.01355258, -1.6906547 , -1.48351109, -1.52686941, -0.82903916], dtype=Constants.TF_PRECISION)
        log_t = tf.convert_to_tensor([ 9.56559188, 10.146128  , 10.146128  , 10.146128  ,  9.82951612,
                                      10.041392  , 10.041392  , 10.041392  , 10.041392  ,  9.61747047], dtype=Constants.TF_PRECISION)
        M_ini = tf.convert_to_tensor([1.2596913304062323, 1.2795316312572347, 0.41909814, 0.6158603], dtype=Constants.TF_PRECISION)

        # lo_EEP = [214., 240.]
        # hi_EEP = [218., 244.]

        # generate index, which would normally be done by the class itself
        x = tf.stack([Fe_H, log_t, tf.zeros_like(Fe_H)], axis=-1)
        grid._ip._create_index(x)
        
        mi_EEP, mi_M_ini, mask = grid._find_EEP(Fe_H, log_t, M_ini)
        diff = tf.abs(M_ini - mi_M_ini)
        self.assertTrue(tf.math.reduce_max(diff) < 1.0e-5)

    def test_interp3d_problematic(self):
        grid = self.load_grid()

        iso_M_ini = tf.linspace(tf.constant(0.6, dtype=Constants.TF_PRECISION), tf.constant(0.8, dtype=Constants.TF_PRECISION), 20)
        iso_Fe_H = tf.fill(iso_M_ini.shape, tf.constant(-2.0, dtype=Constants.TF_PRECISION))
        iso_log_t = tf.fill(iso_M_ini.shape, tf.constant(10.160, dtype=Constants.TF_PRECISION))    # 10.175

        iso = grid.interp3d(iso_Fe_H, iso_log_t, iso_M_ini, values=[grid.values['hsc_g'], grid.values['hsc_i']])