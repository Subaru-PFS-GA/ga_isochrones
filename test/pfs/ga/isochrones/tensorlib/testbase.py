import os
from importlib import reload
import numpy as np
import numpy.testing as npt
from unittest import TestCase

import pfs.ga.isochrones.tensorlib.config as config

class TestBase(TestCase):
    def get_lib(self):
        raise NotImplementedError()
    
    def get_tensortype():
        raise NotImplementedError()

    def get_tl(self):
        config.TENSORLIB = self.get_lib()
        import pfs.ga.isochrones.tensorlib as tl
        reload(tl)
        return tl

    def test_tensor(self):
        tl = self.get_tl()        
        t = tl.tensor([1, 2, 3 ])
        self.assertEqual(self.get_tensortype(), type(t))

    def test_cpu(self):
        tl = self.get_tl()
        t = tl.tensor([1, 2, 3 ])
        self.assertEqual(np.ndarray, type(tl.cpu(t))) 

    def test_shape(self):
        tl = self.get_tl()
        t = tl.tensor([1, 2, 3 ])
        tl.shape(t)

    def test_size(self):
        tl = self.get_tl()
        t = tl.tensor([1, 2, 3 ])
        self.assertEqual(3, tl.size(t))

    def test_ndim(self):
        tl = self.get_tl()
        t = tl.tensor([1, 2, 3 ])
        self.assertEqual(1, tl.ndim(t))

    def test_arange(self):
        tl = self.get_tl()

        t = tl.arange(3)
        npt.assert_equal(np.array([0, 1, 2]), tl.cpu(t))

        t = tl.arange(1, 3)
        npt.assert_equal(np.array([1, 2]), tl.cpu(t))

        t = tl.arange(1, 3, 0.5)
        npt.assert_equal(np.array([1, 1.5, 2, 2.5]), tl.cpu(t))

    def test_linspace(self):
        tl = self.get_tl()

        t = tl.linspace(1, 2, 3)
        npt.assert_equal(np.array([1, 1.5, 2]), tl.cpu(t))

    def test_zeros(self):
        tl = self.get_tl()
        t = tl.zeros((3,))
        self.assertEqual(self.get_tensortype(), type(t))
        npt.assert_equal([0, 0, 0], tl.cpu(t))

    def test_ones(self):
        tl = self.get_tl()
        t = tl.ones((3,))
        self.assertEqual(self.get_tensortype(), type(t))
        npt.assert_equal([1, 1, 1], tl.cpu(t))

    def test_one_hot(self):
        tl = self.get_tl()
        t1 = tl.tensor([0, 1, 2])
        t2 = tl.one_hot(t1, 3)
        npt.assert_equal([[1, 0, 0], [0, 1, 0], [0, 0, 1]], tl.cpu(t2))

    def test_eye(self):
        tl = self.get_tl()
        t = tl.eye(3)
        npt.assert_equal([[1, 0, 0], [0, 1, 0], [0, 0, 1]], tl.cpu(t))

    def test_empty(self):
        tl = self.get_tl()
        t = tl.ones((3,))
        self.assertEqual(self.get_tensortype(), type(t))
        self.assertEqual((3,), tl.shape(t))

    def test_full(self):
        tl = self.get_tl()
        t = tl.full((3,), 2)
        self.assertEqual(self.get_tensortype(), type(t))
        npt.assert_equal([2, 2, 2], tl.cpu(t))

    def test_zeros_like(self):
        tl = self.get_tl()
        t = tl.tensor([1, 2, 3 ])
        t2 = tl.zeros_like(t)
        npt.assert_equal([0, 0, 0], tl.cpu(t2))

    def test_ones_like(self):
        tl = self.get_tl()
        t = tl.tensor([1, 2, 3 ])
        t2 = tl.ones_like(t)
        npt.assert_equal([1, 1, 1], tl.cpu(t2))

    def test_full_like(self):
        tl = self.get_tl()
        t = tl.tensor([1, 2, 3 ])
        t2 = tl.full_like(t, 2)
        npt.assert_equal([2, 2, 2], tl.cpu(t2))

    def test_empty_like(self):
        tl = self.get_tl()
        t = tl.tensor([1, 2, 3 ])
        t2 = tl.empty_like(t)
        self.assertEqual((3,), tl.shape(t2))
        
    def test_reshape(self):
        tl = self.get_tl()
        t = tl.tensor([1, 2, 3 ])
        t2 = tl.reshape(t, (3, 1))
        self.assertEqual((3, 1), tl.shape(t2))

    def test_stack(self):
        tl = self.get_tl()
        t = tl.tensor([1, 2, 3 ])
        t2 = tl.tensor([4, 5, 6 ])
        t3 = tl.stack([t, t2], axis=-1)
        self.assertEqual((3, 2), tl.shape(t3))

    def test_concat(self):
        tl = self.get_tl()
        t = tl.tensor([1, 2, 3 ])
        t2 = tl.tensor([4, 5, 6 ])
        t3 = tl.concat([t, t2], axis=0)
        self.assertEqual((6,), tl.shape(t3))

    def test_where(self):
        tl = self.get_tl()
        t = tl.tensor([1, 2, 3 ])
        t2 = tl.tensor([4, 5, 6 ])
        t3 = tl.where(t > 2, t, t2)
        npt.assert_equal([4, 5, 3], tl.cpu(t3))

    def test_tile(self):
        tl = self.get_tl()

        t = tl.tensor([1, 2, 3 ])
        t2 = tl.tile(t, (2,))
        npt.assert_equal([1, 2, 3, 1, 2, 3], tl.cpu(t2))

        t = tl.tensor([[1, 2], [3, 4]])
        t2 = tl.tile(t, (2,))
        npt.assert_equal([[1, 2, 1, 2], [3, 4, 3, 4]], tl.cpu(t2))

        t = tl.tensor([[1, 2], [3, 4]])
        t2 = tl.tile(t, (1, 2,))
        npt.assert_equal([[1, 2, 1, 2], [3, 4, 3, 4]], tl.cpu(t2))

        t = tl.tensor([[1, 2], [3, 4]])
        t2 = tl.tile(t, (2, 1,))
        npt.assert_equal([[1, 2], [3, 4], [1, 2], [3, 4]], tl.cpu(t2))

    def test_searchsorted(self):
        tl = self.get_tl()

        t = tl.tensor([1, 2, 3, 4, 5])
        t2 = tl.tensor([0, 1, 2, 3, 4, 5, 6])
        t3 = tl.searchsorted(t2, t)
        npt.assert_equal([1, 2, 3, 4, 5], tl.cpu(t3))