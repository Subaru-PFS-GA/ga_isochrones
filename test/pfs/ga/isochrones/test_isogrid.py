import os

import tensorflow.compat.v2 as tf

from .test_base import TestBase
from pfs.ga.isochrones.constants import Constants
from pfs.ga.isochrones import Dartmouth

class TestIsoGrid(TestBase):
    def load_grid(self):
        grid = Dartmouth()
        grid.load(os.path.join(self.ISOCHRONES_DATA, 'dartmouth/import/afep0_cfht_sdss_hsc/isochrones.h5'))
        return grid
        
    def test_load(self):
        grid = self.load_grid()
        self.assertIsNotNone(grid.axes)
        self.assertIsNotNone(grid.values)

    def test_interp3d_EEP(self):
        def interp3d_EEP(shape):
            Fe_H = tf.random.uniform(shape, grid.Fe_H[0] + 0.05, grid.Fe_H[-1] - 0.05, dtype=Constants.TF_PRECISION)
            log_t = tf.random.uniform(shape, 8, 9.0, dtype=Constants.TF_PRECISION)
            EEP = tf.random.uniform(shape, 150, 350, dtype=Constants.TF_PRECISION)

            [M_ini] = grid._interp3d_EEP(Fe_H, log_t, EEP, [grid.M_ini])
            self.assertEqual(shape, M_ini.shape)

            [M_ini] = grid._interp3d_EEP(Fe_H, log_t, EEP, [grid.M_ini])
            self.assertEqual(shape, M_ini.shape)

        grid = self.load_grid()
        interp3d_EEP((10,))

        grid = self.load_grid()
        interp3d_EEP((10, 3))

    def test_bracket_EEP(self):
        def bracket_EEP(shape):
            Fe_H = tf.random.uniform(shape, grid.Fe_H[0] + 0.05, grid.Fe_H[-1] - 0.05, dtype=Constants.TF_PRECISION)
            log_t = tf.random.uniform(shape, 8, 9.0, dtype=Constants.TF_PRECISION)

            x = tf.stack([Fe_H, log_t, tf.zeros_like(Fe_H)], axis=-1)
            grid._ip._create_index(x)
            lo_EEP, hi_EEP = grid._bracket_EEP()

            self.assertEqual(shape, lo_EEP.shape)
            self.assertEqual(shape, hi_EEP.shape)

        grid = self.load_grid()
        bracket_EEP((10,))

        grid = self.load_grid()
        bracket_EEP((10, 3))

    def test_find_EEP(self):
        def find_EEP(shape):
            # The MIST grid has a hole along [Fe/H] = -2.5, index 3
            Fe_H = tf.random.uniform(shape, -1.95, grid.Fe_H[-1] - 0.05, dtype=Constants.TF_PRECISION)
            log_t = tf.random.uniform(shape, 8, 9.0, dtype=Constants.TF_PRECISION)
            M_ini = tf.random.uniform(shape, 0.1, 3.0, dtype=Constants.TF_PRECISION)

            x = tf.stack([Fe_H, log_t, M_ini], axis=-1)
            grid._ip._create_index(x)
            mi_EEP, mi_M_ini, mask = grid._find_EEP(Fe_H, log_t, M_ini)

            diff = tf.abs(M_ini - mi_M_ini)
            diff = tf.math.reduce_max(tf.where(mask, 0, diff))
            self.assertTrue(diff < 1.0e-3)

        grid = self.load_grid()
        find_EEP((10,))

        grid = self.load_grid()
        find_EEP((10, 3))

    def test_find_EEP_limits(self):
        Fe_H = tf.convert_to_tensor([-0.09211389, -0.09211389, -0.18223292, -0.18223292,  0.00416489,
                                      0.00416489, -0.07610813, -0.07610813, -0.06870603, -0.06870603,
                                     -0.80026388, -0.80026388, -0.51160281, -0.51160281, -0.73004026,
                                     -0.73004026, -1.94617729, -1.94617729], dtype=Constants.TF_PRECISION)
        log_t = tf.convert_to_tensor([ 8.60093254,  8.60093254,  9.09404259,  9.09404259,  9.42434881,
                                       9.42434881,  9.66365287,  9.66365287,  9.80777486,  9.80777486,
                                       9.98890987,  9.98890987, 10.041392  , 10.041392  , 10.146128  ,
                                      10.146128  , 10.175     , 10.175     ], dtype=Constants.TF_PRECISION)
        M_ini = tf.convert_to_tensor([0.12229782, 0.34620701, 0.10798566, 0.113546  , 0.10026138,
                                      0.11675613, 0.31158672, 0.10588064, 0.12232815, 0.13681453,
                                      0.12524728, 0.16292408, 0.10486671, 0.1227991 , 0.12355714,
                                      0.15054357, 0.6735583 , 0.69743672], dtype=Constants.TF_PRECISION)
        
        grid = self.load_grid()

        x = tf.stack([Fe_H, log_t, tf.zeros_like(M_ini)], axis=-1)
        grid._ip._create_index(x)
        
        lo_EEP, hi_EEP = grid._bracket_EEP()

        [lo_M_ini] = grid._interp3d_EEP(Fe_H, log_t, lo_EEP + 0.01, [grid.M_ini])
        [hi_M_ini] = grid._interp3d_EEP(Fe_H, log_t, hi_EEP - 0.01, [grid.M_ini])

        self.assertEqual(0, tf.reduce_sum(tf.cast(tf.math.is_nan(lo_M_ini), dtype=tf.int32)))
        self.assertEqual(0, tf.reduce_sum(tf.cast(tf.math.is_nan(hi_M_ini), dtype=tf.int32)))

    def test_interp3d(self):
        def interp3d(shape):
            # The MIST grid has a hole along [Fe/H] = -2.5, index 3
            Fe_H = tf.random.uniform(shape, -1.95, grid.Fe_H[-1] - 0.05, dtype=Constants.TF_PRECISION)
            log_t = tf.random.uniform(shape, 8, 9.0, dtype=Constants.TF_PRECISION)
            M_ini = tf.random.uniform(shape, 0.1, 3.0, dtype=Constants.TF_PRECISION)

            eep, _, [hsc_g, hsc_r], mask = grid.interp3d(Fe_H, log_t, M_ini, [grid.values['hsc_g'], grid.values['hsc_i']])
            self.assertEqual(shape, hsc_g.shape)
            self.assertEqual(shape, hsc_r.shape)
    
        grid = self.load_grid()

        interp3d((1,))
        interp3d((10,))
        interp3d((10,3))

    def test_interp3d_nan(self):
        Fe_H = tf.convert_to_tensor([0.01329973, -0.00103306, 0.0048698, -0.01950695], dtype=Constants.TF_PRECISION)
        log_t = tf.convert_to_tensor([10.34432429, 10.33866003, 10.30724244, 10.30065941], dtype=Constants.TF_PRECISION)
        M_ini = tf.convert_to_tensor([0.10254606, 0.40250976, 0.12998558, 0.148707], dtype=Constants.TF_PRECISION)

        grid = self.load_grid()

        eep, _, [hsc_g, hsc_r], mask = grid.interp3d(Fe_H, log_t, M_ini, [grid.values['hsc_g'], grid.values['hsc_i']])

        pass

    
"""
    def test_bracket_EEP_problematic(self):
        # In this situation, an RGB star is interpolated with a mass that's
        # higher than some up the mass upper limits of the surrounding
        # isochrones. Interpolation still should be possible since the EEP
        # range is nicely covered

        Fe_H = tf.fill((1,), tf.constant(-1.75, dtype=Constants.TF_PRECISION))
        log_t = tf.fill((1,), tf.constant(10.02, dtype=Constants.TF_PRECISION))
        M_ini = tf.fill((1,), tf.constant(0.85, dtype=Constants.TF_PRECISION))

        grid = self.load_grid()

        # generate index, which would normally be done by the class itself
        x = tf.stack([Fe_H, log_t, tf.zeros_like(Fe_H)], axis=-1)
        grid._ip._create_index(x)

        lo_EEP, hi_EEP = grid._bracket_EEP(Fe_H, log_t, M_ini)
        self.assertTrue(hi_EEP > 0)

    def test_find_EEP_problematic(self):
        grid = self.load_grid()

        Fe_H = tf.convert_to_tensor([0.001899077227174164, 0.014318513225709326, -1.0283096, -3.491832], dtype=Constants.TF_PRECISION)
        log_t = tf.convert_to_tensor([9.150067065169749, 9.203306146482129, 8.350221, 8.301319], dtype=Constants.TF_PRECISION)
        M_ini = tf.convert_to_tensor([1.2596913304062323, 1.2795316312572347, 0.41909814, 0.6158603], dtype=Constants.TF_PRECISION)

        # lo_EEP = [214., 240.]
        # hi_EEP = [218., 244.]

        # generate index, which would normally be done by the class itself
        x = tf.stack([Fe_H, log_t, tf.zeros_like(Fe_H)], axis=-1)
        grid._ip._create_index(x)
        
        mi_EEP, mi_M_ini, mask = grid._find_EEP(Fe_H, log_t, M_ini)
        diff = tf.abs(M_ini - mi_M_ini)
        self.assertTrue(tf.math.reduce_max(diff) < 1.0e-5)

    def test_interp3d_eep_problematic(self):
        grid = self.load_grid()
            
        eep = tf.range(450, 550, dtype=Constants.TF_PRECISION)

        fe_h = tf.fill(eep.shape, grid.Fe_H[5])
        log_t = tf.fill(eep.shape, grid.log_t[100])
        x = tf.stack([fe_h, log_t, eep], axis=-1)
        grid._ip._create_index(x)
        [hsc_g, hsc_i] = grid._interp3d_EEP(fe_h, log_t, eep, [grid.values['hsc_g'], grid.values['hsc_i']])
        self.assertEqual(0, tf.reduce_max(tf.math.abs(hsc_g - grid.values['hsc_g'][5, 100, 450: 550])))
        
        fe_h = tf.fill(eep.shape, grid.Fe_H[5] - 0.125)
        log_t = tf.fill(eep.shape, grid.log_t[100])
        x = tf.stack([fe_h, log_t, eep], axis=-1)
        grid._ip._create_index(x)
        [hsc_g, hsc_i] = grid._interp3d_EEP(fe_h, log_t, eep, [grid.values['hsc_g'], grid.values['hsc_i']])
        self.assertEqual(True, tf.reduce_all(
            tf.math.logical_and(
                tf.math.less(hsc_g, grid.values['hsc_g'][5, 100, 450: 550]),
                tf.math.greater(hsc_g, grid.values['hsc_g'][4, 100, 450: 550]))))
        self.assertEqual(True, tf.reduce_all(
            tf.math.logical_and(
                tf.math.less(hsc_i, grid.values['hsc_i'][5, 100, 450: 550]),
                tf.math.greater(hsc_i, grid.values['hsc_i'][4, 100, 450: 550]))))

        fe_h = tf.fill(eep.shape, grid.Fe_H[5])
        log_t = tf.fill(eep.shape, grid.log_t[100] - 0.025)
        x = tf.stack([fe_h, log_t, eep], axis=-1)
        grid._ip._create_index(x)
        [hsc_g, hsc_i] = grid._interp3d_EEP(fe_h, log_t, eep, [grid.values['hsc_g'], grid.values['hsc_i']])
        self.assertEqual(True, tf.reduce_all(
            tf.math.logical_and(
                tf.math.greater(hsc_g, grid.values['hsc_g'][5, 99, 450: 550]),
                tf.math.less(hsc_g, grid.values['hsc_g'][5, 100, 450: 550]))))
        self.assertEqual(True, tf.reduce_all(
            tf.math.logical_and(
                tf.math.greater(hsc_i, grid.values['hsc_i'][5, 99, 450: 550]),
                tf.math.less(hsc_i, grid.values['hsc_i'][5, 100, 450: 550]))))

        pass
        
    

    def test_interp3d_ordering(self):
        grid = self.load_grid()

        M_ini = tf.linspace(tf.constant(0.4, dtype=Constants.TF_PRECISION), tf.constant(0.8, dtype=Constants.TF_PRECISION), 100)
        fe_h = tf.fill(M_ini.shape, grid.Fe_H[5])
        log_t = tf.fill(M_ini.shape, grid.log_t[100])

        eep, [hsc_g, hsc_i] = grid.interp3d(fe_h, log_t, M_ini, [grid.values['hsc_g'], grid.values['hsc_i']])

        pass

    def test_interp3d_problematic(self):
        grid = self.load_grid()

        # These cases are at the edge of the MIST isochrones dues to very high
        # age log t > 9.3. The solution is to mark extrapolations with a NaN

        Fe_H = tf.convert_to_tensor([-3.1474695, -4.0014763, -3.1543455, -3.2279196, -3.9314253,
                                     -3.2643065, -3.1226394, -3.253033, -3.2979946, -3.1741328], dtype=Constants.TF_PRECISION)
        log_t = tf.convert_to_tensor([9.309808, 9.263662, 9.309559, 9.338262, 9.300449, 9.307342,
                                      9.303298, 9.3072605, 9.322571, 9.313253], dtype=Constants.TF_PRECISION)
        M_ini = tf.convert_to_tensor([1.3958763, 1.3805327, 1.383056 , 1.3865354, 1.3958708, 1.3992141,
                                      1.3874997, 1.3892338, 1.3959047, 1.3907031], dtype=Constants.TF_PRECISION)

        eep, [hsc_g, hsc_r] = grid.interp3d(Fe_H, log_t, M_ini, [grid.values['hsc_g'], grid.values['hsc_r']])

        self.assertEqual(10, tf.reduce_sum(tf.cast(tf.math.is_nan(hsc_g), tf.int32)))
        self.assertEqual(10, tf.reduce_sum(tf.cast(tf.math.is_nan(hsc_r), tf.int32)))

    def test_interp3d_problematic_end_of_MS(self):
        grid = self.load_grid()

        # There result in MS stars with too small mass

        Fe_H = tf.convert_to_tensor([-3.2 ], dtype=Constants.TF_PRECISION)
        log_t = tf.convert_to_tensor([9.1], dtype=Constants.TF_PRECISION)
        M_ini = tf.convert_to_tensor([1.65], dtype=Constants.TF_PRECISION)

        eep, [hsc_g, hsc_r] = grid.interp3d(Fe_H, log_t, M_ini, [grid.values['hsc_g'], grid.values['hsc_r']])

        self.assertEqual(1, tf.reduce_sum(tf.cast(tf.math.is_nan(hsc_g), tf.int32)))
        self.assertEqual(1, tf.reduce_sum(tf.cast(tf.math.is_nan(hsc_r), tf.int32)))

    def test_interp3d_problematic_middle(self):
        grid = self.load_grid()

        Fe_H = tf.convert_to_tensor([-3.9], dtype=Constants.TF_PRECISION)
        log_t = tf.convert_to_tensor([10.0], dtype=Constants.TF_PRECISION)
        M_ini = tf.convert_to_tensor([0.860], dtype=Constants.TF_PRECISION)

        eep, [hsc_g, hsc_r] = grid.interp3d(Fe_H, log_t, M_ini, [grid.values['hsc_g'], grid.values['hsc_r']])

        self.assertEqual(1, tf.reduce_sum(tf.cast(tf.math.is_nan(hsc_g), tf.int32)))
        self.assertEqual(1, tf.reduce_sum(tf.cast(tf.math.is_nan(hsc_r), tf.int32)))
"""