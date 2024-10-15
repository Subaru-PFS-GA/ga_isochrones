import os
from importlib import reload

from .testbase import TestBase

class PytorchTest(TestBase):
    def get_lib(self):
        return 'pytorch'
    
    def get_tensortype(self):
        from torch import Tensor
        return Tensor