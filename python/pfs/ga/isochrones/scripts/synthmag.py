#!/usr/bin/env python3

import os
import sys
import numpy as np
import argparse
from tqdm import tqdm

from .script import Script
from .. import IsoGrid
from pfs.ga.pfsspec.core import Filter
from pfs.ga.pfsspec.core.grid import ArrayGrid, RbfGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.bosz import Bosz
from pfs.ga.pfsspec.stellar.grid.phoenix import Phoenix

class SynthMag(Script):
    """
    Calculates synthetic magnitudes for and existing isochrone grid that
    contains physical parameters.
    """

    ISOGRID_CONFIG = {
        'dartmouth': {
            'type': IsoGrid
        }
    }

    SPECGRID_CONFIG = {
        'bosz-rbf': {
            'config': Bosz(pca=False, normalized=False),
            'grid': RbfGrid
        },
        'phoenix-rbf': {
            'config': Phoenix(pca=False, normalized=False),
            'grid': RbfGrid
        }
    }

    def __init__(self):
        super(SynthMag, self).__init__()

        self._top = None
        self._outdir = None
        self._isogrid_type = None
        self._isogrid_path = None
        self._specgrid_type = None
        self._specgrid_path = None
        self._filter_names = None
        self._filter_files = None

    def add_args(self, parser):
        super(SynthMag, self).add_args(parser)

        parser.add_argument('--top', type=int, help='\nProcess this many models only.')
        parser.add_argument('--out', type=str, required=True, help='\nOutput file.')
        parser.add_argument('--isogrid-type', type=str, required=True, choices=[k for k in SynthMag.ISOGRID_CONFIG], help='\nType of isochrone grid.')
        parser.add_argument('--isogrid-path', type=str, required=True, help='\nIsochrone grid file')
        parser.add_argument('--specgrid-type', type=str, required=True, choices=[k for k in SynthMag.SPECGRID_CONFIG], help='\nType of spectrum grid.')
        parser.add_argument('--specgrid-path', type=str, required=True, help='\nSpectrum grid file')
        parser.add_argument('--filter-names', type=str, nargs='+', help='\nFilter names as should appear in results grid.')
        parser.add_argument('--filter-files', type=str, nargs='+', help='\nList of filter files.')

    def parse_args(self):
        super(SynthMag, self).parse_args()
        
        self._top = self.get_arg('top', self._top)
        self._outdir = self.get_arg('out', self._outdir)
        self._isogrid_type = self.get_arg('isogrid_type', self._isogrid_type)
        self._isogrid_path = self.get_arg('isogrid_path', self._isogrid_path)
        self._specgrid_type = self.get_arg('specgrid_type', self._specgrid_type)
        self._specgrid_path = self.get_arg('specgrid_path', self._specgrid_path)
        self._filter_names = self.get_arg('filter_names', self._filter_names)
        self._filter_files = self.get_arg('filter_files', self._filter_files)

    def load_isogrid(self):
        self.isogrid = SynthMag.ISOGRID_CONFIG[self._isogrid_type]['type']()
        self.isogrid.load(os.path.join(self._isogrid_path, 'isochrones.h5'))

        print('Loaded isochrone grid.')
        print('[Fe/H] = ', self.isogrid.Fe_H)
        print('log_t = ', self.isogrid.log_t)

    def load_specgrid(self):
        c = SynthMag.SPECGRID_CONFIG[self._specgrid_type]
        self.specconfig = c['config']
        self.specgrid = ModelGrid(c['config'], c['grid'])
        self.specgrid.preload_arrays = False
        self.specgrid.load(os.path.join(self._specgrid_path, 'spectra.h5'), format='h5')

        print('Loaded spectrum grid.')
        axes = { k: ax for _, k, ax in self.specgrid.enumerate_axes() }
        for k in axes:
            print(k, axes[k].values)

    def load_filters(self):        
        self.filters = {}
        for fn, ff in zip(self._filter_names, self._filter_files):
            f = Filter()
            f.read(ff)
            self.filters[fn] = f

        print('Loaded {} filter curves.'.format(len(self.filters)))
        for fn in self.filters:
            print(fn, self.filters[fn].wave.min(), self.filters[fn].wave.max())

    def calculate_mags(self):
        # TODO: extend here if the isochrone grid has more than two fundamental
        #       parameters, i.e. [a/M] or similar.

        fe_h_count = self.isogrid.Fe_H.shape[0]
        log_t_count = self.isogrid.log_t.shape[0]
        eep_count = self.isogrid.EEP.shape[0]
        total_count = fe_h_count * log_t_count * eep_count
        t = tqdm(total=total_count, file=sys.__stdout__)

        mags = {}
        for f in self.filters:
            mags[f] = np.empty((fe_h_count, log_t_count, eep_count), dtype=np.float64)

        for i, fe_h in enumerate(self.isogrid.Fe_H.numpy()):
            print(f'\nProcessing [Fe/H] = {fe_h} ({i + 1} out of {fe_h_count})')
            for j, log_t in enumerate(self.isogrid.log_t.numpy()):
                for k, eep in enumerate(self.isogrid.EEP.numpy()):

                    t.update()

                    t_eff = 10 ** self.isogrid.values['Log_T_eff'][i, j, k].numpy()
                    log_l = self.isogrid.values['log_L'][i, j, k].numpy()
                    log_g = self.isogrid.values['log_g'][i, j, k].numpy()

                    if np.isinf(log_g):
                        for f in self.filters:
                            mags[f][i, j, k] = log_g
                    else:
                        # TODO: how to make this work for any grid type when we don't know
                        #       the parameter list beforehand?
                        spec = self.specgrid.get_model(Fe_H=fe_h, M_H=fe_h, T_eff=t_eff, log_g=log_g, a_M=0, C_M=0)
                        for f in self.filters:
                            if spec is None:
                                m = np.nan
                            else:
                                m = spec.synthmag_carrie(self.filters[f], log_l)
                            mags[f][i, j, k] = m

                    if t.n == self._top:
                        return mags

        return mags

    def prepare(self):
        super(SynthMag, self).prepare()

        self.create_output_dir(self._outdir)
        self.init_logging(self._outdir)
        self.init_tensorlib()

        self.load_isogrid()
        self.load_specgrid()
        self.load_filters()

    def run(self):

        mags = self.calculate_mags()
        for f in mags:
            self.isogrid.values[f] = mags[f]

        self.isogrid.save(os.path.join(self._outdir, 'isochrones.h5'))

    def execute_notebooks(self):
        super(SynthMag, self).execute_notebooks()

        # TODO: add notebooks here        
                
def main():
    script = SynthMag()
    script.execute()

if __name__ == "__main__":
    main()