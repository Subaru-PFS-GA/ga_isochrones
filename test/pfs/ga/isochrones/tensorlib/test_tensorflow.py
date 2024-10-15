import os
from importlib import reload

from .testbase import TestBase

class TensorflowTest(TestBase):
    def get_lib(self):
        return 'tensorflow'
    
    def get_tensortype(self):
        from tensorflow.python.framework.ops import EagerTensor
        return EagerTensor