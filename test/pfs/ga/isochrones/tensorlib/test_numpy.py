import os
from importlib import reload
from unittest import TestCase

from .tensorlibtestbase import TensorlibTestBase

class NumpyTest(TensorlibTestBase, TestCase):
    def get_lib(self):
        return 'numpy'
    
    def get_tensortype(self):
        from numpy import ndarray
        return ndarray