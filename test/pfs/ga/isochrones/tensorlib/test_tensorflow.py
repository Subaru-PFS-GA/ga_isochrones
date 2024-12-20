import os
from importlib import reload
from unittest import TestCase

from .tensorlibtestbase import TensorlibTestBase

class TensorflowTest(TensorlibTestBase, TestCase):
    def get_lib(self):
        return 'tensorflow'
    
    def get_tensortype(self):
        from tensorflow.python.framework.ops import EagerTensor
        return EagerTensor