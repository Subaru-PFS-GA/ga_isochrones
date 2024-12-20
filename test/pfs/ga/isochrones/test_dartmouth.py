import os

from .test_base import TestBase
from pfs.ga.isochrones import tensorlib as tt
from pfs.ga.isochrones.constants import Constants
from pfs.ga.isochrones import Dartmouth

class DartmouthTest(TestBase):
    def load_grid(self):
        grid = Dartmouth()
        grid.load(os.path.join(self.ISOCHRONES_DATA, 'dartmouth/import/afep0_cfht_sdss_hsc/isochrones.h5'))
        return grid

    def test_interp3d_problematic(self):
        grid = self.load_grid()

        iso_M_ini = tt.linspace(0.6, 0.8, 20, dtype=Constants.TT_PRECISION)
        iso_Fe_H = tt.full(iso_M_ini.shape, -2.0, dtype=Constants.TT_PRECISION)
        iso_log_t = tt.full(iso_M_ini.shape, 10.160, dtype=Constants.TT_PRECISION)

        iso = grid.interp3d(iso_Fe_H, iso_log_t, iso_M_ini, values=[grid.values['hsc_g'], grid.values['hsc_i']])