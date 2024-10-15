import os
import logging
import numpy as np
import pandas as pd
import re
import logging

from ..dartmouth import Dartmouth
from ..util.astro import *
from ..util.data import *
from .isogridreader import IsoGridReader

def _convert_to_hsc(grid):
    """
    Calculates HSC photometry from SDSS for the entire isochrone grid.

    Args:
        grid (IsoGrid): An isochrone grid with SDSS photometry
    """

    # Temporarily ignore floating point errors to deal with nans in source arrays
    with np.errstate(all="ignore"):
        hsc = sdss_to_hsc(grid._values['sdss_g'], grid._values['sdss_r'],
                                grid._values['sdss_i'], grid._values['sdss_z'])

    grid._values.update(hsc)

class DartmouthReader(IsoGridReader):
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
            'conversion': _convert_to_hsc,
            'dest': [ 'hsc_g', 'hsc_i' ]
        },
        'PanSTARRS': {
            'source': [ 'open', 'gp1', 'rp1', 'ip1', 'zp1', 'yp1', 'wp1' ],
            'dest': [ 'ps_o', 'ps_g', 'ps_r', 'ps_i', 'ps_z', 'ps_y', 'ps_w' ]
        }
    }

    Fe_H_map = {
        'm05': -0.5,
        'm10': -1.0,
        'm15': -1.5,
        'm20': -2.0,
        'm25': -2.5,
        #'m30': -3.0,
        #'m35': -3.5, not full grid points here
        #'m40': -4.0,
        'p00':  0.0,
        'p02':  0.2,
        'p03':  0.3,
        'p05':  0.5,
    }

    A_Fe_map = {
        #'m2': -0.2,
        'p0':  0.0,
        'p2':  0.2,
        'p4':  0.4,
        'p6':  0.6,
        'p8':  0.8,
    }

    # Not putting in Y values yet
    Y_map = {
        '': 0.2452,
        'y40':0.4,
        'y33':0.33,
    }

    def __init__(self):
        super().__init__()

        self._file_pattern = 'feh{}afe{}{}.{}{}'     # Fe_H, A, Y, photometry, postfix

        self._alpha = 'p0'
        self._helium = ''
        self._read_young = True

        self._grid = None

    def add_args(self, parser):
        super().add_args(parser)

        parser.add_argument('--alpha', type=str, help='Alpha abundance.\n')
        parser.add_argument('--helium', type=str, help='Helium abundance.\n')

    def parse_args(self, args):
        super().parse_args(args)

        if 'alpha' in args and args['alpha'] is not None:
            self._alpha = args['alpha']
        if 'helium' in args and args['helium'] is not None:
            self._helium = args['helium']

    def _get_filename(self, Fe_H, A, Y, photometry, postfix=''):
        return os.path.join(self._in, photometry, self._file_pattern.format(Fe_H, A, Y, photometry, postfix))

    def _get_ages_and_breaks(self, filename):
        # Read file and file lines starting with #AGE=, this will determine
        # where we need to break the file to get isochrones with different ages
        ages = []
        breaks = []
        with open(filename, 'r') as f:
            q = 0
            while True: 
                line = f.readline()
                if not line:
                    break
                
                m = re.search(r'#AGE=\s*([0-9.]+)\s+EEPS=\s*([0-9]+)', line)
                if m is not None:
                    age = line[m.regs[1][0]:m.regs[1][1]]
                    eeps = line[m.regs[2][0]:m.regs[2][0]]
                    ages.append(age)
                    breaks.append(q)
                q += 1
            breaks.append(q)

        return np.array(ages, dtype=float), breaks

    def _read_file(self, Fe_H, A, Y, photometry, postfix=''):
        filename = self._get_filename(Fe_H, A, Y, photometry, postfix=postfix)
        names = ['EEP', 'M_ini', 'log_T_eff', 'log_g', 'log_L'] + \
                DartmouthReader.PHOTOMETRY[photometry]['source']
        if not os.path.isfile(filename):
            logging.warning('Isochrone file not found: `{}`.'.format(filename))
            return None
        
        logging.info('Loading file `{}`'.format(filename))

        ages, breaks = self._get_ages_and_breaks(filename)
        
        all = []
        for i in range(len(ages)):
            df = pd.read_csv(filename, sep='\s+', header=None, names=names, encoding='utf-8',
                             comment='#',
                             skiprows=lambda j: (j < breaks[i]) | (breaks[i + 1] <= j))
            df['age'] = ages[i] # Gyr
            df['log_t'] = np.log10(ages[i]) + 9.0
            df['Fe_H'] = DartmouthReader.Fe_H_map[Fe_H]
            df['alpha_Fe'] = DartmouthReader.A_Fe_map[A]
            
            all.append(df)

        df = pd.concat(all)
        return df

    def _read_all(self, photometry, alpha=None, helium=None, read_young=None):
        alpha = alpha if alpha is not None else self._alpha
        helium = helium if helium is not None else self._helium
        read_young = read_young if read_young is not None else self._read_young

        data = {}
        for Fe_H in DartmouthReader.Fe_H_map:
            df1 = self._read_file(Fe_H, alpha, helium, photometry)           # old
            if not read_young:
                data[Fe_H] = df1
            else:
                df2 = self._read_file(Fe_H, alpha, helium, photometry, '_2')     # young
                if df2 is not None:
                    # Get rid of repeating isochrones
                    df2 = df2[df2['age'] != 1.0]
                    data[Fe_H] = pd.concat([df2, df1])
        return data

    def _create_grid(self):
        return Dartmouth()

    def _build_grid(self, data, photometry):
        param_Fe_H = np.array([DartmouthReader.Fe_H_map[k] for k in DartmouthReader.Fe_H_map if k in data], dtype=np.float32)
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
        for k in values + DartmouthReader.PHOTOMETRY[photometry]['dest']:
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
                    
            for i, v in enumerate(DartmouthReader.PHOTOMETRY[photometry]['dest']):
                if v is not None:
                    set_value(DartmouthReader.PHOTOMETRY[photometry]['source'][i], v)
        
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
