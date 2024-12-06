import os
from importlib import reload
from unittest import TestCase

from .tensorlibtestbase import TensorlibTestBase

class PytorchTest(TensorlibTestBase, TestCase):
    def get_lib(self):
        return 'pytorch'
    
    def get_tensortype(self):
        from torch import Tensor
        return Tensor