import os

import tensorflow.compat.v2 as tf

from ..test_base import TestBase
from pfs.ga.isochrones import Constants
from pfs.ga.isochrones import MIST
from pfs.ga.isochrones.io import MISTReader

class TestMISTReader(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        os.makedirs(cls.ISOCHRONES_TEST, exist_ok=True)

    def _create_reader(self):
        reader = MISTReader()
        reader._in = os.path.join(self.ISOCHRONES_DATA, 'mist/source')
        return reader

    def test_read_file(self):
        reader = self._create_reader()
        df = reader._read_file('p0.00', 'p0.0', '0.4', photometry='SDSSugriz')
        self.assertEqual((105594, 15), df.shape)
        self.assertEqual(10, df['EEP'].min())
        self.assertEqual(1710, df['EEP'].max())

    def test_read_all(self):
        reader = self._create_reader()

        mist = reader._read_all('SDSSugriz', 'p0.0', '0.4')
        self.assertEqual(15, len(mist))
        self.assertEqual((105594, 15), mist['p0.00'].shape)

    def test_build_grid(self):
        reader = self._create_reader()
        reader._grid = MIST()
        
        mist = reader._read_all('SDSSugriz', 'p0.0', '0.4')
        self.assertEqual(15, len(mist))
        self.assertEqual((105594, 15), mist['p0.00'].shape)

        reader._build_grid(mist, photometry='SDSSugriz')
        fn = os.path.join(self.ISOCHRONES_TEST, 'mist_sdss.h5')
        reader._grid.save(os.path.join(self.ISOCHRONES_TEST, 'mist_sdss.h5'))
        