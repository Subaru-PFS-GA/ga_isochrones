import numpy as np
import tensorflow.compat.v2 as tf

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

        i0 = tf.identity(0)
        i1 = tf.identity(1)
        s1 = tf.identity(2**(self._dim - 1))
        s2 = tf.identity(2**self._dim)

        # Find the values at the surrounding grid points
        # Note, that gather_nd returns 0 for indexes outside the valid range and
        # does not support wrap-around or negative indexes.
        # shape: (..., 8)
        idx = self._idx[..., i0:s2, :]
        y = [tf.gather_nd(v, idx) for v in values]

        # Perform the linear interpolation along the grid lines for each axis
        for i in range(len(self._axes) - 1, -1, -1):
            # Note, that gather returns 0 for indexes outside the valid range and
            # does not support wrap-around or negative indexes.
            xx = tf.gather(self._axes[i], self._idx_ax[i])
            x0 = xx[..., i0]
            x1 = xx[..., i1]
            xi = x[..., i]
            y = [self._interp1d(x0, x1, v[..., i0:s1], v[..., s1:s2], xi) for v in y]

            s2 = s1
            s1 = s1 // 2

        return [v[..., 0] for v in y]

    #@tf.function
    def _interp1d(self, x1, x2, y1, y2, x):
        """Interplate value linearly from `(x1, y1)` and `(x2, y2)` to `x`."""
        # x1, x2 and x should have the shape of (N,)
        X = (x - x1) / (x2 - x1)
        Y = y1 + (y2 - y1) * X[..., tf.newaxis]
        return Y

    #@tf.function
    def _digitize(self, x, ax):
        # Binary search
        # l = tf.fill(x.shape, tf.constant(0, tf.int32))
        # r = tf.fill(x.shape, tf.constant(ax.shape[0], tf.int32))
        # for i in range(int(np.floor(np.log2(ax.shape[0]) + 1))):
        #     m = (l + r) // 2
        #     v = tf.gather(ax, m)
        #     c = v < x
        #     l = tf.where(c, m + 1, l)
        #     r = tf.where(c, r, m)

        # return l

        xx = tf.reshape(x, [-1])
        l = tf.searchsorted(ax, xx, side='right')
        l = tf.reshape(l, x.shape)
        return l

    #@tf.function
    def _find_nearby(self, ax, x):
        # Find bracketing indices for every item in x along axis ax
        # Indices are bracket by assuming (,] parameter intervals
        idx = self._digitize(x, ax)
        idx = tf.stack([idx - tf.constant(1, dtype=tf.int32), idx], axis=-1)
        return idx

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
        self._idx_ax = []
        for i, ax in enumerate(self._axes):
            # Find bracketing indices for each value in x along the axis
            idx = self._find_nearby(ax, x[..., i])
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
            A = tf.reshape(tf.repeat(ax, repeats=2**i, axis=-1), batch_shape + (2**(i + 1), 1))
            X.append(A)
        
        Q = X[0]
        for i in range(1, len(X)):
            Q = tf.reshape(tf.repeat(Q, repeats=2, axis=-3), batch_shape + (2**(i + 1), i))
            Q = tf.concat([Q, X[i]], axis=-1)
            
        self._idx = Q

    def _update_index(self, i, x):
        # Update the index along a single direction by finding the new limits then
        # slicing and recombining the index array

        batch_shape = x.shape[:-1]

        # Update the single axis index along the selected dimension
        self._idx_ax[i] = self._find_nearby(self._axes[i], x[..., i])

        A = tf.reshape(tf.repeat(self._idx_ax[i], repeats=2**(self._dim - 1), axis=-1), batch_shape + (2**self._dim, 1))
        if i == 0:
            self._idx = tf.concat([A, self._idx[..., 1:]], axis=-1)
        elif i == self._dim - 1:
            self._idx = tf.concat([self._idx[..., :-1], A], axis=-1)
        else:
            self._idx = tf.concat([self._idx[..., 0:i], A, self._idx[..., i + 1:]], axis=-1)