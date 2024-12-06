import os

from ..test_base import TestBase
from pfs.ga.isochrones import Constants
from pfs.ga.isochrones import Dartmouth
from pfs.ga.isochrones.io import DartmouthReader
from pfs.ga.isochrones.io.dartmouthreader import _convert_to_hsc

class DartmouthReaderTest(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        os.makedirs(cls.ISOCHRONES_TEST, exist_ok=True)

    def _create_reader(self):
        reader = DartmouthReader()
        reader._in = os.path.join(self.ISOCHRONES_DATA, 'dartmouth/source/isochrones')
        return reader

    def test_read_file(self):
        reader = self._create_reader()
        df = reader._read_file('p00', 'p0', '', photometry='SDSSugriz')
        self.assertEqual((10029, 14), df.shape)
        self.assertEqual(2, df['EEP'].min())
        self.assertEqual(279, df['EEP'].max())

    def test_read_all(self):
        reader = self._create_reader()

        dartmouth = reader._read_all('SDSSugriz', 'p0', '', read_young=True)
        self.assertEqual(9, len(dartmouth))
        self.assertEqual((14845, 14), dartmouth['p00'].shape)

        dartmouth = reader._read_all('SDSSugriz', 'p0', '', read_young=False)
        self.assertEqual(9, len(dartmouth))
        self.assertEqual((10029, 14), dartmouth['p00'].shape)

    def test_build_grid(self):
        reader = self._create_reader()
        reader._grid = Dartmouth()
        
        dartmouth = reader._read_all('SDSSugriz', 'p0', '', read_young=True)
        self.assertEqual(9, len(dartmouth))
        self.assertEqual((14845, 14), dartmouth['p00'].shape)

        reader._build_grid(dartmouth, photometry='SDSSugriz')
        fn = os.path.join(self.ISOCHRONES_TEST, 'dartmouth_sdss.h5')
        reader._grid.save(os.path.join(self.ISOCHRONES_TEST, 'dartmouth_sdss.h5'))

    def test_convert_to_hsc(self):
        grid = Dartmouth()
        grid.load(os.path.join(self.ISOCHRONES_TEST, 'dartmouth_sdss.h5'))
        
        reader = self._create_reader()
        _convert_to_hsc(grid)
        grid.save(os.path.join(self.ISOCHRONES_TEST, 'dartmouth_hsc.h5'))