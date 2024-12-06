import os
from importlib import reload
import numpy as np
import numpy.testing as npt
from unittest import TestCase

import pfs.ga.isochrones.tensorlib.config as config

class TensorlibTestBase():
    def get_lib(self):
        raise NotImplementedError()
    
    def get_tensortype():
        raise NotImplementedError()

    def get_tensorlib(self):
        config.TENSORLIB = self.get_lib()
        import pfs.ga.isochrones.tensorlib as tt
        reload(tt)
        return tt
    
    def get_tensorlib_random(self):
        config.TENSORLIB = self.get_lib()
        import pfs.ga.isochrones.tensorlib.random as tr
        reload(tr)
        return tr

    def test_tensor(self):
        tt = self.get_tensorlib()        
        t = tt.tensor([1, 2, 3 ])
        self.assertEqual(self.get_tensortype(), type(t))

        t = tt.tensor([1, 2, 3 ], dtype=tt.float64)
        self.assertEqual(self.get_tensortype(), type(t))

        t = tt.tensor([1, 2, 3 ], device='cpu')
        self.assertEqual(self.get_tensortype(), type(t))

    def test_cast(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1, 2, 3 ])
        t2 = tt.cast(t, tt.float32)
        self.assertEqual(tt.float32, t2.dtype)

    def test_cpu(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1, 2, 3 ])
        self.assertEqual(np.ndarray, type(tt.cpu(t))) 

    def test_shape(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1, 2, 3 ])
        tt.shape(t)

    def test_size(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1, 2, 3 ])
        self.assertEqual(3, tt.size(t))

    def test_ndim(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1, 2, 3 ])
        self.assertEqual(1, tt.ndim(t))

    def test_arange(self):
        tt = self.get_tensorlib()

        t = tt.arange(3)
        npt.assert_equal(np.array([0, 1, 2]), tt.cpu(t))

        t = tt.arange(1, 3)
        npt.assert_equal(np.array([1, 2]), tt.cpu(t))

        t = tt.arange(1, 3, 0.5)
        npt.assert_equal(np.array([1, 1.5, 2, 2.5]), tt.cpu(t))

    def test_linspace(self):
        tt = self.get_tensorlib()

        t = tt.linspace(1, 2, 3)
        npt.assert_equal(np.array([1, 1.5, 2]), tt.cpu(t))

    def test_zeros(self):
        tt = self.get_tensorlib()
        t = tt.zeros((3,))
        self.assertEqual(self.get_tensortype(), type(t))
        npt.assert_equal([0, 0, 0], tt.cpu(t))

    def test_ones(self):
        tt = self.get_tensorlib()
        t = tt.ones((3,))
        self.assertEqual(self.get_tensortype(), type(t))
        npt.assert_equal([1, 1, 1], tt.cpu(t))

    def test_one_hot(self):
        tt = self.get_tensorlib()
        t1 = tt.tensor([0, 1, 2])
        t2 = tt.one_hot(t1, 3)
        npt.assert_equal([[1, 0, 0], [0, 1, 0], [0, 0, 1]], tt.cpu(t2))

    def test_eye(self):
        tt = self.get_tensorlib()
        t = tt.eye(3)
        npt.assert_equal([[1, 0, 0], [0, 1, 0], [0, 0, 1]], tt.cpu(t))

    def test_empty(self):
        tt = self.get_tensorlib()
        t = tt.ones((3,))
        self.assertEqual(self.get_tensortype(), type(t))
        self.assertEqual((3,), tt.shape(t))

    def test_full(self):
        tt = self.get_tensorlib()

        t = tt.full((3,), 2)
        self.assertEqual(self.get_tensortype(), type(t))
        npt.assert_equal([2, 2, 2], tt.cpu(t))

        t = tt.full((3,), 2, dtype=tt.float64)
        self.assertEqual(self.get_tensortype(), type(t))
        npt.assert_equal([2, 2, 2], tt.cpu(t))

    def test_zeros_like(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1, 2, 3 ])
        t2 = tt.zeros_like(t)
        npt.assert_equal([0, 0, 0], tt.cpu(t2))

    def test_ones_like(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1, 2, 3 ])
        t2 = tt.ones_like(t)
        npt.assert_equal([1, 1, 1], tt.cpu(t2))

    def test_full_like(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1, 2, 3 ])
        t2 = tt.full_like(t, 2)
        npt.assert_equal([2, 2, 2], tt.cpu(t2))

    def test_empty_like(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1, 2, 3 ])
        t2 = tt.empty_like(t)
        self.assertEqual((3,), tt.shape(t2))
        
    def test_reshape(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1, 2, 3 ])
        t2 = tt.reshape(t, (3, 1))
        self.assertEqual((3, 1), tt.shape(t2))

    def test_stack(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1, 2, 3 ])
        t2 = tt.tensor([4, 5, 6 ])

        t3 = tt.stack([t, t2])
        self.assertEqual((2, 3), tt.shape(t3))

        t3 = tt.stack([t, t2], axis=-1)
        self.assertEqual((3, 2), tt.shape(t3))

    def test_concat(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1, 2, 3 ])
        t2 = tt.tensor([4, 5, 6 ])
        t3 = tt.concat([t, t2], axis=0)
        self.assertEqual((6,), tt.shape(t3))

    def test_where(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1, 2, 3 ])
        t2 = tt.tensor([4, 5, 6 ])
        t3 = tt.where(t > 2, t, t2)
        npt.assert_equal([4, 5, 3], tt.cpu(t3))

    def test_repeat(self):
        tt = self.get_tensorlib()

        t = tt.tensor([1, 2, 3])
        t2 = tt.repeat(t, 2, axis=0)
        npt.assert_equal([1, 1, 2, 2, 3, 3], tt.cpu(t2))
        t2 = tt.repeat(t, 2, axis=-1)
        npt.assert_equal([1, 1, 2, 2, 3, 3], tt.cpu(t2))

        t = tt.tensor([[1, 2], [3, 4]])
        t2 = tt.repeat(t, 2, axis=0)
        npt.assert_equal([[1, 2], [1, 2], [3, 4], [3, 4]], tt.cpu(t2))
        t2 = tt.repeat(t, 2, axis=-1)
        npt.assert_equal([[1, 1, 2, 2], [3, 3, 4, 4]], tt.cpu(t2))

    def test_tile(self):
        tt = self.get_tensorlib()

        t = tt.tensor([1, 2, 3 ])
        t2 = tt.tile(t, (2,))
        npt.assert_equal([1, 2, 3, 1, 2, 3], tt.cpu(t2))

        t = tt.tensor([[1, 2], [3, 4]])
        t2 = tt.tile(t, (2,))
        npt.assert_equal([[1, 2, 1, 2], [3, 4, 3, 4]], tt.cpu(t2))

        t = tt.tensor([[1, 2], [3, 4]])
        t2 = tt.tile(t, (1, 2,))
        npt.assert_equal([[1, 2, 1, 2], [3, 4, 3, 4]], tt.cpu(t2))

        t = tt.tensor([[1, 2], [3, 4]])
        t2 = tt.tile(t, (2, 1,))
        npt.assert_equal([[1, 2], [3, 4], [1, 2], [3, 4]], tt.cpu(t2))

    def test_gather(self):
        tt = self.get_tensorlib()

        t = tt.tensor([1, 2, 3])
        t2 = tt.gather(t, [0, 2])
        npt.assert_equal([1, 3], tt.cpu(t2))

        t = tt.tensor([[1, 2], [3, 4]])
        t2 = tt.gather(t, [0, 1])
        npt.assert_equal([[1, 2], [3, 4]], tt.cpu(t2))

        t = tt.tensor([[1, 2], [3, 4]])
        t2 = tt.gather(t, [0, 1], axis=1)
        npt.assert_equal([[1, 2], [3, 4]], tt.cpu(t2))

        t = tt.tensor([[1, 2], [3, 4]])
        t2 = tt.gather(t, [0, 1], axis=-1)
        npt.assert_equal([[1, 2], [3, 4]], tt.cpu(t2))

        #

        t = tt.tensor([[1, 2], [3, 4]])
        t2 = tt.gather(t, [[0, 1], [1, 0]])
        npt.assert_equal([[[1, 2], [3, 4]], [[3, 4], [1, 2]]], tt.cpu(t2))

        t = tt.tensor([[1, 2], [3, 4]])
        t2 = tt.gather(t, [[0, 1], [1, 0]], axis=1)
        npt.assert_equal([[[1, 2], [2, 1]], [[3, 4], [4, 3]]], tt.cpu(t2))

        t = tt.tensor([[1, 2], [3, 4]])
        t2 = tt.gather(t, [[0, 1], [1, 0]], axis=-1)
        npt.assert_equal([[[1, 2], [2, 1]], [[3, 4], [4, 3]]], tt.cpu(t2))

    def test_gather_nd(self):
        tt = self.get_tensorlib()

        t = tt.tensor([[1, 2], [3, 4]])
        t2 = tt.gather_nd(t, [[0, 0], [1, 1]])
        npt.assert_equal([1, 4], tt.cpu(t2))

    def test_searchsorted(self):
        tt = self.get_tensorlib()

        t = tt.tensor([1, 2, 3, 4, 5])
        t2 = tt.tensor([0, 1, 2, 3, 4, 5, 6])
        t3 = tt.searchsorted(t2, t)
        npt.assert_equal([1, 2, 3, 4, 5], tt.cpu(t3))

    def test_any(self):
        tt = self.get_tensorlib()
        t = tt.tensor([True, True, True])
        self.assertTrue(tt.any(t))

        t = tt.tensor([False, False, False])
        self.assertFalse(tt.any(t))

        t = tt.tensor([[True, False], [False, False]])
        self.assertTrue(tt.any(t, axis=1)[0])
        self.assertFalse(tt.any(t, axis=1)[1])

    def test_all(self):
        tt = self.get_tensorlib()
        t = tt.tensor([True, True, True])
        self.assertTrue(tt.all(t))

        t = tt.tensor([True, False, False])
        self.assertFalse(tt.all(t))

        t = tt.tensor([[True, True], [False, True]])
        self.assertTrue(tt.all(t, axis=1)[0])
        self.assertFalse(tt.all(t, axis=1)[1])

    def test_sum(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1, 2, 3 ])
        self.assertEqual(6, tt.sum(t))

        t = tt.tensor([[1, 2], [3, 4]])
        self.assertEqual(10, tt.sum(t))

        t = tt.tensor([[1, 2], [3, 4]])
        self.assertTrue(tt.all(tt.tensor([3, 7]) == tt.sum(t, axis=1)))

    def test_mean(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1.0, 2.0, 3.0 ])
        self.assertEqual(2, tt.mean(t))

        t = tt.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(2.5, tt.mean(t))

        t = tt.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue(tt.all(tt.tensor([1.5, 3.5]) == tt.mean(t, axis=1)))

    def test_logsumexp(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1.0, 2.0, 3.0 ])
        v = tt.logsumexp(t)
        self.assertTrue(tt.all(tt.abs(v - 3.4076059644443806) < 1e-6))

        t = tt.tensor([[1.0, 2.0], [3.0, 4.0]])
        v = tt.logsumexp(t)
        self.assertTrue(tt.abs(v - 4.440189698561196) < 1e-6)

        t = tt.tensor([[1.0, 2.0], [3.0, 4.0]])
        v = tt.logsumexp(t, axis=1)
        e = tt.tensor([2.31326169, 4.31326169])
        self.assertTrue(tt.all(tt.abs(v - e) < 1e-6))

    def test_min(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1, 2, 3])
        self.assertEqual(1, tt.min(t))

        t = tt.tensor([[1, 2], [3, 4]])
        self.assertEqual(1, tt.min(t))

        t = tt.tensor([[1, 2], [3, 4]])
        self.assertTrue(tt.all(tt.tensor([1, 3]) == tt.min(t, axis=1)))

    def test_max(self):
        tt = self.get_tensorlib()
        t = tt.tensor([1, 2, 3])
        self.assertEqual(3, tt.max(t))

        t = tt.tensor([[1, 2], [3, 4]])
        self.assertEqual(4, tt.max(t))

        t = tt.tensor([[1, 2], [3, 4]])
        self.assertTrue(tt.all(tt.tensor([2, 4]) == tt.max(t, axis=1)))

    def test_random_normal(self):
        tt = self.get_tensorlib()
        tr = self.get_tensorlib_random()
        
        r = tr.normal(0, 1, (3,))
        self.assertEqual((3,), tt.shape(r))

    def test_random_uniform(self):
        tt = self.get_tensorlib()
        tr = self.get_tensorlib_random()
        
        r = tr.uniform(0, 1, (3,))
        self.assertEqual((3,), tt.shape(r))