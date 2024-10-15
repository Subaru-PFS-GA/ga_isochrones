# from .tensorlib import tensorlib as tl

class InterpNd(object):
    """
    N-dimensional linear interpolation over a rectangular grid with irregular
    binning. _idx_ax and _idx are used as index lookups that keep track of the
    grid cells where interpolation should happen. This index can be kept between
    calls to speed up iterative operation.
    """
    
    def __init__(self, axes):
        self._axes = axes
        self._dim = len(axes)
        self._idx_ax = None
        self._mask_ax = None
        self._idx = None

    def __call__(self, x, values, reindex=False):
        """
        Interpolates the grids of `values` to positions `x`.

        Parameters
        ----------
        x : array of float
            Array representing the new coordinates
        values : list of arrays of float
            List of grid value arrays to interpolate from
        reindex : bool, optional
            If True, the grid is reindexed, see _create_index
        """
        return self._interpNd(x, values, reindex=reindex)

    def _interpNd(self, x, values, reindex=False):
        if self._idx is None or reindex:
            self._create_index(x)

        # x.shape (..., d) where the last index must be equal to the dimensions of the grid
        # self._idx.shape (..., 2**d, d) which indexes the surrounding grid points
            
        batch_shape = x.shape[:-1]

        i0 = 0
        i1 = 1
        s1 = 2**(self._dim - 1)
        s2 = 2**self._dim

        # Combine the mask for each axis
        mask = None
        for i in range(len(self._axes)):
            if mask is None:
                mask = self._mask_ax[i]
            else:
                mask |= self._mask_ax[i]

        # Find the values at the surrounding grid points
        # Note, that gather_nd returns 0 for indexes outside the valid range and
        # does not support wrap-around or negative indexes. On the CPU, gather_nd
        # throws and error when indexing outside the array
        # shape: (..., 8)
        idx = self._idx[..., i0:s2, :]

        # If running on the CPU, we need to reset the indices which would raise an exception
        s = (Ellipsis,) + (idx.ndim - len(batch_shape)) * (tl.newaxis,)
        idx = tl.where(mask[s], tl.zeros_like(idx), idx)

        y = []
        for v in values:
            v = tf.gather_nd(v, idx)
            s = (Ellipsis,) + (v.ndim - len(batch_shape)) * (tl.newaxis,)
            v = tl.where(mask[s], tl.nan, v)
            y.append(v)

        # Perform the linear interpolation along the grid lines for each axis
        for i in range(len(self._axes) - 1, -1, -1):
            # Note, that gather returns 0 for indexes outside the valid range and
            # does not support wrap-around or negative indexes.
            idx = self._idx_ax[i]
            idx = tl.where(mask[..., tl.newaxis], tl.zeros_like(idx), idx)

            xx = tl.gather(self._axes[i], idx)
            x0 = xx[..., i0]
            x1 = xx[..., i1]
            xi = x[..., i]
            y = [self._interp1d(x0, x1, v[..., i0:s1], v[..., s1:s2], xi) for v in y]

            s2 = s1
            s1 = s1 // 2

        vv = []
        for v in y:
            v = v[..., 0]
            v = tl.where(mask, tl.nan, v)
            vv.append(v)

        return vv

    def _interp1d(self, x1, x2, y1, y2, x):
        """Interplate value linearly from `(x1, y1)` and `(x2, y2)` to `x`."""
        # x1, x2 and x should have the shape of (N,)
        X = (x - x1) / (x2 - x1)
        Y = y1 + (y2 - y1) * X[..., tl.newaxis]
        return Y

    def _digitize(self, x, ax):
        xx = tl.reshape(x, [-1])
        l = tl.searchsorted(ax, xx, side='right')
        l = tl.reshape(l, x.shape)
        mask = (x < ax[0]) | (x >= ax[-1])
        return l, mask

    def _find_nearby(self, ax, x):
        # Find bracketing indices for every item in x along axis ax
        # Indices are bracket by assuming (,] parameter intervals
        idx, mask = self._digitize(x, ax)
        idx = tl.stack([idx - 1, idx], axis=-1)
        return idx, mask

    def _create_index(self, x):
        """
        Create an inverse look-up index along each dimension of the data cube.
        Results will contain the indices bracketing the value of x in each dimension.
        x has the shape of (batch_shape, dim)
        idx_ax will contain indices along each dimension with a shape of (batch_shape)
        idx will contain indices along all dimensions for the 2**d neighboring grid points
        with a shape of (batch_dim, 2**d, d)
        """

        # TODO: we could save a little bit memory here by only computing the lo index of the
        #       surrounding grid points. But then it would require dealing with intervals in
        #       _interpNd on smaller tensors which might be the same amount of work anyway

        batch_shape = x.shape[:-1]

        # Create single axis indices by looping over the dimensions of the grid
        # The value of mask is True of an item of the input is outside the grid bounds
        self._mask_ax = []
        self._idx_ax = []
        for i, ax in enumerate(self._axes):
            # Find bracketing indices for each value in x along the axis
            idx, mask = self._find_nearby(ax, x[..., i])
            self._mask_ax.append(mask)
            self._idx_ax.append(idx)

        # Combine single axis indices together to get a list of coordinates that mark the d**3
        # corners of the nearby grid points

        # This is sort of magic here that works with grids of any dimension by iteratively
        # sizing up the index tensor by powers of two to end up with an index of shape
        # (batch_shape, 2^dim, dim)
        # shape[-2]: the 2^3 surrounding grid points
        # shape[-1]: the indices of the surrounding grid points

        X = []
        for i, ax in enumerate(self._idx_ax):
            A = tl.reshape(tf.repeat(ax, repeats=2**i, axis=-1), batch_shape + (2**(i + 1), 1))
            X.append(A)
        
        Q = X[0]
        for i in range(1, len(X)):
            Q = tl.reshape(tf.repeat(Q, repeats=2, axis=-3), batch_shape + (2**(i + 1), i))
            Q = tl.concat([Q, X[i]], axis=-1)
            
        self._idx = Q

    def _update_index(self, i, x):
        # Update the index along a single direction by finding the new limits then
        # slicing and recombining the index array

        batch_shape = x.shape[:-1]

        # Update the single axis index along the selected dimension
        self._idx_ax[i], self._mask_ax[i] = self._find_nearby(self._axes[i], x[..., i])

        A = tl.reshape(tf.repeat(self._idx_ax[i], repeats=2**(self._dim - 1), axis=-1), batch_shape + (2**self._dim, 1))
        if i == 0:
            self._idx = tl.concat([A, self._idx[..., 1:]], axis=-1)
        elif i == self._dim - 1:
            self._idx = tl.concat([self._idx[..., :-1], A], axis=-1)
        else:
            self._idx = tl.concat([self._idx[..., 0:i], A, self._idx[..., i + 1:]], axis=-1)