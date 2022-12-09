import os
from unittest import TestCase
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf

class TestBase(TestCase):
    @classmethod
    def setUpClass(cls):
        tf.enable_v2_behavior()
        gpus = tf.config.list_physical_devices('GPU') 
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        try:
            tf.compat.v1.enable_eager_execution()
        except ValueError:
            pass

        cls.ISOCHRONES_ROOT = os.environ['ISOCHRONES_ROOT'].strip('"') if 'ISOCHRONES_ROOT' in os.environ else ''
        cls.ISOCHRONES_DATA = os.environ['ISOCHRONES_DATA'].strip('"') if 'ISOCHRONES_DATA' in os.environ else ''
        cls.ISOCHRONES_TEST = os.environ['ISOCHRONES_TEST'].strip('"') if 'ISOCHRONES_TEST' in os.environ else ''

    def setUp(self):
        plt.figure(figsize=(10, 6))

    def get_filename(self, ext):
        filename = type(self).__name__[4:] + '_' + self._testMethodName[5:] + ext
        return filename

    def save_fig(self, f=None, filename=None):
        if f is None:
            f = plt
        if filename is None:
            filename = self.get_filename('.png')
        f.savefig(os.path.join(self.ISOCHRONES_TEST, filename))