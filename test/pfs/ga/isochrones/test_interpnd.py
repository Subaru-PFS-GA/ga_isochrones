import os
import numpy as np
import numpy.testing

from .test_base import TestBase
from pfs.ga.isochrones import tensorlib as tt
from pfs.ga.isochrones.constants import Constants
from pfs.ga.isochrones.interpnd import InterpNd
from pfs.ga.isochrones import Dartmouth

class InterpNdTest(TestBase):
    def load_grid(self):
        grid = Dartmouth()
        grid.load(os.path.join(self.ISOCHRONES_DATA, 'dartmouth/import/afep0_cfht_sdss_hsc/isochrones.h5'))
        return grid

    def get_extreme(self, grid):
        log_t = tt.tensor([grid.log_t[0] - 1.0, grid.log_t[0], grid.log_t[grid.log_t.shape[0] // 2], grid.log_t[grid.log_t.shape[0] // 2 + 1], grid.log_t[-1], grid.log_t[-1] + 1])
        return log_t

    def get_random(self, grid, shape):
        log_t = tt.random.uniform(size=shape, a=grid.log_t[0] + 0.5, b=grid.log_t[-1] - 0.5, dtype=Constants.TT_PRECISION)
        return log_t

    def test_digitize(self):
        def digitize(log_t):
            ip = InterpNd([grid.Fe_H, grid.log_t, grid.EEP])
            idx, mask = ip._digitize(log_t, grid.log_t)
            mask = (log_t <= grid.log_t[0]) | (log_t > grid.log_t[-1])
            self.assertEqual(idx.shape, log_t.shape)

            idx_n = idx.numpy()
            idx_n[mask.numpy()] = -99
            ref_n = np.digitize(log_t.numpy(), grid.log_t.numpy(), right=False)
            ref_n[mask.numpy()] = -99
            numpy.testing.assert_array_equal(idx_n, ref_n)

        grid = self.load_grid()
        digitize(self.get_extreme(grid))
        digitize(self.get_random(grid, (10,)))
        digitize(self.get_random(grid, (11,)))
        digitize(self.get_random(grid, (10, 7)))
        digitize(self.get_random(grid, (11, 7)))

    def test_find_nearby(self):
        def find_nearby(log_t):
            ip = InterpNd([grid.Fe_H, grid.log_t, grid.EEP])
            idx, mask = ip._find_nearby(grid.log_t, log_t)
            self.assertEqual(idx.shape, log_t.shape + (2,))

        grid = self.load_grid()
        find_nearby(self.get_extreme(grid))
        find_nearby(self.get_random(grid, (10,)))
        find_nearby(self.get_random(grid, (10, 7)))

    def test_interpNd(self):
        def get_extreme():
            Fe_H = tt.tensor([grid.Fe_H[0] - 1.0, grid.Fe_H[0], grid.Fe_H[grid.Fe_H.shape[0] // 2], grid.Fe_H[grid.Fe_H.shape[0] // 2 + 1], grid.Fe_H[-1], grid.Fe_H[-1] + 1])
            return Fe_H

        def interpNd(Fe_H):
            #Fe_H = tt.random.uniform(log_t.shape, grid.Fe_H[0] + 0.01, grid.Fe_H[-1], dtype=Constants.TT_PRECISION)
            #EEP = tt.random.uniform(log_t.shape, 202.0, 605.0, dtype=Constants.TT_PRECISION)
            #Fe_H = tt.full(Fe_H.shape, -0.125, dtype=Constants.TT_PRECISION)
            log_t = tt.full(Fe_H.shape, 8.5, dtype=Constants.TT_PRECISION)
            EEP = tt.full(Fe_H.shape, 202.4, dtype=Constants.TT_PRECISION)

            x = tt.stack([Fe_H, log_t, EEP], axis=-1)
            ip = InterpNd([grid.Fe_H, grid.log_t, grid.EEP])
            [M_ini] = ip(x, [grid.M_ini])
            self.assertEqual(log_t.shape, M_ini.shape)

        grid = self.load_grid()
        interpNd(get_extreme())
        