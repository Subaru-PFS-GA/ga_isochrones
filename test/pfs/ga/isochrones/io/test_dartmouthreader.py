import os

import tensorflow.compat.v2 as tf

from ..test_base import TestBase
from pfs.ga.isochrones.constants import Constants
from pfs.ga.isochrones import Dartmouth
from pfs.ga.isochrones.io import DartmouthReader

class TestDartmouthReader(TestBase):
    def _create_grid(self):
        grid = Dartmouth()
        grid._input_path = os.path.join(self.ISOCHRONES_DATA, 'isochrones/dartmouth/isochrones/SDSSugriz/')
        return grid

    def test_read_file(self):
        grid = self._create_grid()
        df = grid._read_dartmouth('p00', 'p0', '')
        self.assertEqual((371073, 14), df.shape)

    def test_read_all(self):
        grid = self._create_grid()

        dartmouth = grid._read_all('p0', '', read_young=True)
        self.assertEqual(9, len(dartmouth))
        self.assertEqual((447483, 14), dartmouth['p00'].shape)

        dartmouth = grid._read_all('p0', '', read_young=False)
        self.assertEqual(9, len(dartmouth))
        self.assertEqual((371073, 14), dartmouth['p00'].shape)

    def test_build_grid(self):
        grid = self._create_grid()
        
        dartmouth = grid._read_all('p0', '', read_young=True)
        self.assertEqual(9, len(dartmouth))
        self.assertEqual((447483, 14), dartmouth['p00'].shape)

        grid._build_grid(dartmouth)

        grid.save('/scratch/ceph/dobos/data/isochrones/dartmouth/test/dartmouth_sdss.h5')

    def test_convert_to_hsc(self):
        grid = Dartmouth()
        grid.load('/scratch/ceph/dobos/data/isochrones/dartmouth/test/dartmouth_sdss.h5', format='np')
        grid._convert_to_hsc()
        grid.save('/scratch/ceph/dobos/data/isochrones/dartmouth/test/dartmouth_hsc.h5')