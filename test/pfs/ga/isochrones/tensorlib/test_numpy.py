import os
from importlib import reload

from .testbase import TestBase

class NumpyTest(TestBase):
    def get_lib(self):
        return 'numpy'
    
    def get_tensortype(self):
        from numpy import ndarray
        return ndarray