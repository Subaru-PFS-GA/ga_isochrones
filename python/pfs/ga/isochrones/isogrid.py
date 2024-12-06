import h5py as h5

import numpy as np

from . import tensorlib as tt
from .constants import Constants
from .interpnd import InterpNd

class Grid(object):
    def __init__(self, dtype=Constants.TT_PRECISION, device=None):
        self._dtype = dtype
        self._device = device
        self._axes = None
        self._values = None

    @property
    def axes(self):
        return self._axes

    @property
    def values(self):
        return self._values

class IsoGrid(Grid):
    def __init__(self, dtype=Constants.TT_PRECISION, device=None):
        super(IsoGrid, self).__init__(dtype=dtype, device=device)
        self._ip = None

    @property
    def Fe_H(self):
        return self._axes['Fe_H']

    @property
    def log_t(self):
        return self._axes['log_t']

    @property
    def EEP(self):
        return self._axes['EEP']

    @property
    def M_ini(self):
        return self._values['M_ini']

    def load(self, filename):
        with h5.File(filename, 'r') as f:
            # Loading order of axes does matter
            axes = ['Fe_H', 'log_t', 'EEP']
            grp = f['isochrones']['axes']
            self._axes = {k: grp[k][:] for k in axes}

            grp = f['isochrones']['values']
            self._values = {k: grp[k][:] for k in grp}

        # Force allocation on the device
        self._axes = {k: tt.tensor(self._axes[k], dtype=self._dtype, device=self._device) for k in self._axes}
        self._values = {k: tt.tensor(self._values[k], dtype=self._dtype, device=self._device) for k in self._values}

        # Use interpnd's implementation to bracket all parameters
        self._ip = InterpNd([self.Fe_H, self.log_t, self.EEP])
        self._find_limits()

    def save(self, filename):
        with h5.File(filename, 'w') as f:
            grp_iso = f.create_group('isochrones')

            # Save parameters, here we assume numpy arrays everywhere
            grp_params = grp_iso.create_group('axes')
            for k in self._axes:
                grp_params.create_dataset(k, data=self._axes[k])

            # Save grids
            grp_values = grp_iso.create_group('values')
            for k in self._values:
                grp_values.create_dataset(k, data=self._values[k])

    def _find_limits(self):
        # Some isochrones don't start at EEP=0 and don't end at EEP=1. These
        # are marked with -inf and inf, respectively, in the M_ini field.
        # Count the number of infs in the M_ini field to find the minimum and maximum
        # EEP values along each isochrone.

        # Find absolute minimum and maximum EEP and M_ini along every isochrone
        mask = tt.cast(~(self.M_ini > -tt.inf), tt.int32)
        self._lo_idx = tt.sum(mask, axis=-1)
        self._lo_EEP = tt.gather(self.EEP, self._lo_idx)

        mask = tt.cast((self.M_ini < tt.inf), tt.int32)
        self._hi_idx = tt.sum(mask, axis=-1) - 1
        self._hi_EEP = tt.gather(self.EEP, self._hi_idx)

    def _interp3d_EEP(self, Fe_H, log_t, EEP, values, update_index=True):
        # Interpolate in 3D, parallel to grid lines only.
        # This function optionally updates the index in the EEP direction to
        # allow caching of Fe_H and log_t directions.
        x = tt.stack([Fe_H, log_t, EEP], axis=-1)
        if self._ip._idx is None:
            self._ip._create_index(x)
        elif update_index:
            self._ip._update_index(2, x)
        y = self._ip(x, values)
        return y

    def _bracket_EEP(self):
        # Find EEP limits given the current index built for Fe_H and log_t.
        # This function cannot be called without finding the 4 neighboring grid lines first.
        # Use create_index to find neighboring isochrones.
        # Cannot interpolate outside the covered EEP range if any of the neighboring grid lines
        # are outside, so take max of lower limits and min of upper limits.
        # Here we assume 3 dimensions (Fe_H, log_t, EEP) and the constant range 0:2 refer to
        # the indexes along dimensions Fe_H and log_t - we ignore the bracketing in the EEP direction
        lo_EEP = tt.max(tt.gather_nd(self._lo_EEP, self._ip._idx[..., 0:4, 0:2]), axis=-1)
        hi_EEP = tt.min(tt.gather_nd(self._hi_EEP, self._ip._idx[..., 0:4, 0:2]), axis=-1)
        return lo_EEP, hi_EEP

    def _find_EEP(self, Fe_H, log_t, M_ini, tolerance=1e-3):
        # Find the EEP corresponding to M_ini along every isochrones
        # interpolated from the grid to constant ([Fe/H], log_t) lines.
        # This function assumes that the index is created over the grid and
        # will update it in the EEP direction.
        
        # TODO: During the first few iterations, the midpoint moves a lot and the index along
        #       the EEP axis needs constantly be updated. When EEP is bracketed into a single
        #       grid cell, this update is no longer necessary. Figure out how to make this speed up.

        # Find the minimum and maximum M_ini along each isochrone where we have
        # a change to interpolate to based on the neighboring gridpoints to the
        # specified Fe_H, log_t combinations
        # Bracket EEP finds the maximum of minimal EEPs and the minimum of maximal
        # EEPs of the neigboring grid cells which means that we lose those EEP cells
        # where any of the neighboring isochrones are missing near the end of the
        # initial mass range.
        lo_EEP, hi_EEP = self._bracket_EEP()

        # Look up the mass limits corresponding to the EEP limits along each
        # isochrone quadruplet surronding each star's real isochrone. 
        # Note, that the EEP intervals are treated as (,], so a small offset is added
        # to the minimum EEPs to make sure we are inside the valid range.
        
        # After each step, we can store the index corresponding to the EEP axis
        # for the 8 surrounding points
        
        # TODO: use eep_eps or something, instead of constants
        
        [lo_M_ini] = self._interp3d_EEP(Fe_H, log_t, lo_EEP + 0.001, [self.M_ini])
        # lo_EEP_idx = self._ip._idx[..., 2]
        
        [hi_M_ini] = self._interp3d_EEP(Fe_H, log_t, hi_EEP - 0.001, [self.M_ini])
        # hi_EEP_idx = self._ip._idx[..., 2]

        # Mask stars with M_ini outside the range of the isochrones
        mask = (M_ini <= lo_M_ini) | (hi_M_ini < M_ini)
        
        bad_count = 2 * M_ini.ndim
        q = 0
        while bad_count > 0 and q < 20:
            mi_EEP = (lo_EEP + hi_EEP) / 2

            # TODO: this is the expensive call here which finds the indices
            #       along the EEP axis for each mi_EEP and updates the index
            #       of the interpolator. Could we do any better here without
            #       making assumptions on the binning of the EEP axis?
            # TODO: insted of taking the the middle EEP we could take the
            #       middle index
            # TODO: we could also simplify the interpolation once inside a single
            #       grid cell and avoid another reverse index lookup
            #       but this doesn't seem to happen for at least some of the stars
            [mi_M_ini] = self._interp3d_EEP(Fe_H, log_t, mi_EEP, [self.M_ini])

            c = (mi_M_ini < M_ini)
            lo_EEP = tt.where(c, mi_EEP, lo_EEP)
            hi_EEP = tt.where(c, hi_EEP, mi_EEP)

            # Verify the convergence
            delta = tt.abs(M_ini - mi_M_ini)
            bad = ((delta > tolerance) & ~mask)
            bad_count = tt.cpu(tt.sum(tt.cast(bad, tt.int32)))

            q += 1

        # Also mask values that are nan
        mask = mask | tt.isnan(mi_M_ini)

        return mi_EEP, mi_M_ini, mask
        
    def interp3d(self, Fe_H, log_t, M_ini, values, update_index=True):
        """Interpolate the isochrones give the initial mass."""

        # This is an implicit interpolation
        
        x = tt.stack([Fe_H, log_t, tt.zeros_like(M_ini)], axis=-1)
        self._ip._create_index(x)
        mi_EEP, mi_M_ini, mask = self._find_EEP(Fe_H, log_t, M_ini)
        res = self._interp3d_EEP(Fe_H, log_t, mi_EEP, values, update_index=update_index)
        
        # Set masked results to nan, this means that we are trying to extrapolate out from the grid
        res = [tt.where(mask, tt.nan, v) for v in res]
        return mi_EEP, mi_M_ini, res, mask

    def interp3d_EEP(self, Fe_H, log_t, EEP, values):
        """Interpolate the isochrones give the EEPs."""

        # This a single forward interpolation

        x = tt.stack([Fe_H, log_t, EEP], axis=-1)
        self._ip._create_index(x)
        
        # Append M_ini to the list of interpolated values
        values = values.copy()       # shallow copy
        values.append(self.values['M_ini'])

        res = self._interp3d_EEP(Fe_H, log_t, EEP, values, update_index=False)
        
        # Remove M_ini from the results
        M_ini = res[-1]
        del res[-1]
        
        mask = tt.is_inf(M_ini)
        res = [tt.where(mask, tt.nan, v) for v in res]
        return EEP, M_ini, res, mask