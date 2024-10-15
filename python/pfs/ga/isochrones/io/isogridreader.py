import os
import logging
import numpy as np

class IsoGridReader():
    def __init__(self):
        self._in = None
        self._out = None
        self._photometry = None
        self._grid = None

    def add_args(self, parser):
        parser.add_argument('--photometry', type=str, nargs='+', required=True, help='List of photometric systems.\n')

    def parse_args(self, args):
        self._in = args['in'][0]
        self._out = args['out']
        if 'photometry' in args and args['photometry'] is not None:
            self._photometry = args['photometry']

    def _create_grid(self):
        raise NotImplementedError()

    def run(self):
        self._grid = self._create_grid()

        for p in self._photometry:
            logging.info('Processing isochrones for photometric system {}.'.format(p))

            if 'conversion' not in self.PHOTOMETRY[p]:
                data = self._read_all(photometry=p)

                if len(data) == 0:
                    raise Exception('No data files for photometric system `{}` found.'.format(p))

                self._build_grid(data, p)
            else:
                # Do photometric system conversions
                self.PHOTOMETRY[p]['conversion'](self._grid)

        self._find_grid_limits(self._grid)

        fn = os.path.join(self._out, 'isochrones.h5')
        self._grid.save(fn)

    def _find_grid_limits(self, grid):
        # Replace NaNs of non-existing EEPs with -inf (below minimum M_ini) and +inf (above maximum M_ini)
        # this give the position of cuts
        # Substitute NaNs
        for k in grid._values.keys():
            for i, x in enumerate(grid._values[k]):
                for j, y in enumerate(x):
                    try:
                        split = np.nanargmin(grid._values[k][i, j, :])
                        #print(i, j, y.shape)
                        m = np.full(y.shape, False)
                        m[:split] = True
                        m[m] = np.isnan(y[m])
                        y[m] = -np.inf

                        m = np.full(y.shape, False)
                        m[split:] = True
                        m[m] = np.isnan(y[m])
                        y[m] = np.inf

                        #print(m)
                        #print(y)
                    except ValueError:
                        print('All values are Nan in `{}` at `{}`, `{}`'.format(k, grid._axes['Fe_H'][i], grid._axes['log_t'][j]))

    def execute_notebooks(self, script):
        pass