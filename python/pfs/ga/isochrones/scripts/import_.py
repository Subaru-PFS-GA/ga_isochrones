#!/usr/bin/env python3

import os
import argparse
import logging
from shutil import copyfile
import h5py
import numpy as np
from tqdm import tqdm

from .script import Script
from ..io import DartmouthReader
from ..io import MISTReader
from ..util.data import *

class Import(Script):
    """
    Implements function to import various kinds of data
    """

    MODES = {
        'dartmouth': DartmouthReader,
        'mist': MISTReader
    }

    def __init__(self):
        super(Import, self).__init__()

        self._mode = None
        self._outdir = None
        self._reader = None

    def _enum_modes(self):
        return enumerate(Import.MODES.keys())

    def add_args(self, parser):
        super(Import, self).add_args(parser)

        mode_parsers = parser.add_subparsers(dest='mode', required=True)

        parsers = {}
        for i, mode in self._enum_modes():
            p = mode_parsers.add_parser(mode)
            super(Import, self).add_args(p)
            p.add_argument('--in', type=str, nargs='+', required=True, help='Input data.\n')
            p.add_argument('--out', type=str, required=True, help='Output directory.\n')
            self.add_mode_args(mode, p)
            parsers[mode] = p     

    def add_mode_args(self, mode, parser):
        script = Import.MODES[mode]()
        script.add_args(parser)

    def parse_args(self):
        super(Import, self).parse_args()

        self._outdir = self.args['out']
        self._mode = self.args['mode']
        self._reader = Import.MODES[self._mode]()
        self.parse_mode_args(self._mode, self.args)

    def parse_mode_args(self, mode, args):
        self._reader.parse_args(args)

    def prepare(self):
        super(Import, self).prepare()

        self.create_output_dir(self._outdir)
        self.init_logging(self._outdir)
        self.init_tensorflow()

    def run(self):
        self._reader.run()

    def execute_notebooks(self):
        super(Import, self).execute_notebooks()
        self._reader.execute_notebooks(self)
        
def main():
    script = Import()
    script.execute()

if __name__ == "__main__":
    main()