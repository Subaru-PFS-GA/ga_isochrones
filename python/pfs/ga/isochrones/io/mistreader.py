import os
import logging
import numpy as np
import h5py
import pandas as pd
from shutil import copyfile
from itertools import permutations 
import re
import logging

from .. import MIST
from ..util.astro import *
from ..util.data import *
from .isogridreader import IsoGridReader

# TODO: Implement similarly to Dartmouth, use notebook implementation as base

class MISTReader(IsoGridReader):
    PHOTOMETRY = {
        'SDSSugriz': {
            'source': [ 'sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z' ],
            'dest': [ 'sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z' ],
        },
        'CFHTugriz': {
            'source': [ 'u', 'g', 'r', 'i_new', 'i_old'],
            'dest': [ 'cfht_u', 'cfht_g', 'cfht_r', 'cfht_i', None ]
        },
        'HSC': {
            'source': [ 'hsc_g', 'hsc_r', 'hsc_i', 'hsc_z', 'hsc_y', 'hsc_nb816', 'hsc_nb921' ],
            'dest': [ 'hsc_g', 'hsc_r', 'hsc_i', 'hsc_z', 'hsc_y', 'hsc_nb816', 'hsc_nb921' ]
        },
        'PanSTARRS': {
            'source': [ 'open', 'gp1', 'rp1', 'ip1', 'zp1', 'yp1', 'wp1' ],
            'dest': [ 'ps_o', 'ps_g', 'ps_r', 'ps_i', 'ps_z', 'ps_y', 'ps_w' ]
        }
    }

    Fe_H_map = {
        'm0.25': -0.25,
        'm0.50': -0.5,
        'm0.75': -0.75,
        'm1.00': -1.00,
        'm1.25': -1.25,
        'm1.50': -1.5,
        'm1.75': -1.75,
        'm2.00': -2.0,
        'm2.50': -2.5,
        'm3.00': -3.0,
        'm3.50': -3.5,
        'm4.00': -4.0,
        'p0.00':  0,
        'p0.25':  0.25,
        'p0.50':  0.5,
        }

    V_V_crit_map = {
        '0.4': 0.4,
        '0.0': 0.0
    }

    A_Fe_map = {
        'p0':  0.0,
    }

    def __init__(self):
        self._file_pattern = 'MIST_{version}_vvcrit{v_v_crit}_{photometry}/MIST_{version}_feh_{Fe_H}_afe_{a_Fe}_vvcrit{v_v_crit}_{photometry}.iso.cmd'

        self._version = "v1.2"
        self._a_Fe = 'p0'
        self._v_v_crit = '0.4'

    def add_args(self, parser):
        super().add_args(parser)

        parser.add_argument('--version', type=str, help='MIST version.\n')
        parser.add_argument('--a_Fe', type=str, help='Alpha abundance.\n')
        parser.add_argument('--v_v_crit', type=str, help='v_v_crit.\n')

    def parse_args(self, args):
        super().parse_args(args)

        if 'version' in args and args['version'] is not None:
            self._version = args['version']
        if 'a_Fe' in args and args['a_Fe'] is not None:
            self._a_Fe = args['a_Fe']
        if 'v_v_crit' in args and args['v_v_crit'] is not None:
            self._v_v_crit = args['v_v_crit']

    def _get_filename(self, Fe_H, a_Fe, v_v_crit, photometry, version='v1.2'):
        return os.path.join(self._in, self._file_pattern.format(
            version=version,
            v_v_crit=v_v_crit,
            photometry=photometry,
            Fe_H=Fe_H,
            a_Fe=a_Fe
        ))
        
    def _read_file(self, Fe_H, a_Fe, v_v_crit, photometry, version='v1.2'):
        filename = self._get_filename(Fe_H, a_Fe, v_v_crit, photometry, version=version)
        if not os.path.isfile(filename):
            logging.warning('Isochrone file not found: `{}`.'.format(filename))
            return None

        names = [
            'EEP',
            'log_t',
            'M_ini',
            'M',
            'log_T_eff',
            'log_g',
            'log_L',
            'Fe_H_ini',
            'Fe_H',] + \
            MISTReader.PHOTOMETRY[photometry]['source'] + \
            ['phase']
    
        df = pd.read_csv(filename, sep='\s+', comment='#', header=None, names=names)
        df['Fe_H'] = MISTReader.Fe_H_map[Fe_H]
        return df

    def _read_all(self, photometry, a_Fe=None, v_v_crit=None, version=None):
        a_Fe = a_Fe if a_Fe is not None else self._a_Fe
        v_v_crit = v_v_crit if v_v_crit is not None else self._v_v_crit
        version = version if version is not None else self._version

        data = {}
        for fe_h in MISTReader.Fe_H_map:
            df = self._read_file(fe_h, a_Fe, v_v_crit, photometry, version=version)
            if df is not None:
                data[fe_h] = df
        return data

    def _create_grid(self):
        return MIST()

    def _build_grid(self, data, photometry):
        param_Fe_H = np.array([MISTReader.Fe_H_map[k] for k in MISTReader.Fe_H_map if k in data], dtype=np.float32)
        param_Fe_H.sort()
        index_Fe_H = {v: i for i, v in enumerate(param_Fe_H)}
        logging.info('param_Fe_H, param_Fe_H.shape = {}, {}'.format(param_Fe_H, param_Fe_H.shape))
        logging.info('index_Fe_H = {}'.format(index_Fe_H))
        
        # param_Alpha_Fe = np.array([A_Fe_map[k] for k in A_Fe_map], dtype=np.float32)
        # param_Alpha_Fe.sort()
        # param_Alpha_Fe, param_Alpha_Fe.shape
        # index_Alpha_Fe = {v: i for i, v in enumerate(param_Alpha_Fe)}
        
        # Find unique ages and EEPs:
        param_log_t = []
        for feh in data:
            param_log_t.append(np.array(np.around(data[feh]['log_t'], decimals=3).unique(), dtype=np.float32))
        param_log_t = np.unique(np.concatenate(param_log_t).flatten())
        param_log_t.sort()
        index_log_t = {v: i for i, v in enumerate(param_log_t)}
        logging.info('index_log_t = {}'.format(index_log_t))
        
        param_EEP = []
        for feh in data:
            param_EEP.append(np.array(data[feh]['EEP'].unique(), dtype=np.int32))
        param_EEP = np.unique(np.concatenate(param_EEP).flatten())
        param_EEP.sort()
        index_EEP = {v: i for i, v in enumerate(param_EEP)}
        logging.info('param_EEP, param_EEP.shape = {}, {}'.format(param_EEP, param_EEP.shape))
        
        values = ['M_ini', 'log_T_eff', 'log_g', 'log_L']
        
        grid = {}
        for k in values + MISTReader.PHOTOMETRY[photometry]['dest']:
            if k is not None:
                grid[k] = np.full((param_Fe_H.shape[0], param_log_t.shape[0], param_EEP.shape[0]), np.nan, dtype=np.float32)

        logging.info('grid[M_ini].shape = {}'.format(grid['M_ini'].shape))
        
        for feh in data:
            logging.info('Processing {} [Fe/H] = {}'.format(photometry, feh))

            def set_value(source, dest):
                grid[dest][
                    data[feh]['Fe_H'].map(lambda x: index_Fe_H[np.float32(x)]),
                    data[feh]['log_t'].map(lambda x: index_log_t[np.float32(np.around(x, decimals=3))]), 
                    data[feh]['EEP'].map(lambda x: index_EEP[np.int32(x)])] = \
                        data[feh][source]

            for v in values:
                set_value(v, v)
                    
            for i, v in enumerate(MISTReader.PHOTOMETRY[photometry]['dest']):
                if v is not None:
                    set_value(MISTReader.PHOTOMETRY[photometry]['source'][i], v)
        
        # If it is the first photometric system processed, set the axes,
        # otherwise verify if they match
        if self._grid._axes is None:
            self._grid._axes = {}
            self._grid._axes['Fe_H'] = param_Fe_H
            self._grid._axes['log_t'] = param_log_t
            self._grid._axes['EEP'] = param_EEP
            self._grid._values = {}
        else:
            np.testing.assert_array_equal(self._grid._axes['Fe_H'], param_Fe_H)
            np.testing.assert_array_equal(self._grid._axes['log_t'], param_log_t)
            np.testing.assert_array_equal(self._grid._axes['EEP'], param_EEP)
            
        self._grid._values.update(grid)