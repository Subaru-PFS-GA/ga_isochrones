import logging
import numpy as np
import h5py
import pandas as pd
from shutil import copyfile
from itertools import permutations 
import re
import logging

from pfs.ga.isochrones.dartmouth import Dartmouth
from ..util.astro import *
from ..util.data import *

# TODO: Implement similarly to Dartmouth, use notebook implementation as base

class MistReader():
    def __init__(self):
        pass

    def add_args(self, parser):
        pass

    def parse_args(self, args):
        pass

    def run(self):
        raise NotImplementedError()