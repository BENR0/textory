#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import dask.array as da

from .util import num_neighbours, neighbour_diff_squared, _dask_neighbour_diff_squared

#TODO
# - add stats for rodogram, madogram, cross variogram


def variogram(x, lag=1):
    """
    Calculate variogram with specified lag for array.

    Parameters
    ----------
    x : array like
        Input array
    lag : int
        Lag distance for variogram, defaults to 1.

    Returns
    -------
    float
        Variogram
    """
    if isinstance(x, da.core.Array):
        diff = _dask_neighbour_diff_squared(x, lag=lag, func="nd_variogram")
    else:
        diff = neighbour_diff_squared(x, lag=lag, func="nd_variogram")

    res = np.nansum(diff)

    #calculate 1/2N part of variogram
    neighbours = num_neighbours(lag)

    cols, rows = x.shape
    num_pix = cols * rows

    factor = 2 * num_pix * neighbours

    return res / factor


def pseudo_cross_variogram(x, y, lag=1):
    """
    Calculate pseudo-variogram with specified lag for
    the two arrays.

    Parameters
    ----------
    x, y : array like
        Input arrays
    lag : int
        Lag distance for variogram, defaults to 1.

    Returns
    -------
    float
        Pseudo-variogram between the two arrays
    """
    if isinstance(x, da.core.Array):
        diff = _dask_neighbour_diff_squared(x, lag=lag, func="nd_variogram")
    else:
        diff = neighbour_diff_squared(x, lag=lag, func="nd_variogram")

    res = np.nansum(diff)

    #calculate 1/2N part of variogram
    neighbours = num_neighbours(lag)

    cols, rows = x.shape
    num_pix = cols * rows

    factor = 2 * num_pix * neighbours

    return res / factor


#def variogram_diff_old(band1, band2, lag=None, window=None):
    #band2 = np.pad(band2, ((1,1),(1,1)), mode="edge")

    #out = np.zeros(band1.shape, dtype=band1.dtype.name)

    ##left and right neighbour
    #out = (band1 - band2[1:-1,2::])**2
    #out += (band1 - band2[1:-1,0:-2:])**2
    ##above and below neighbours
    #out += (band1 - band2[2::,1:-1])**2
    #out += (band1 - band2[0:-2,1:-1])**2
    ##left diagonal neighbours
    #out += (band1 - band2[0:-2,0:-2])**2
    #out += (band1 - band2[2::,0:-2])**2
    ##right diagonal neigbours
    #out += (band1 -band2[0:-2,2::])**2
    #out += (band1 - band2[2::,2::])**2

    #return out

#def variogram_diff_loop(band1, band2, lag=1, window=None):
    #band2 = np.pad(band2, ((lag,lag),(lag,lag)), mode="edge")

    #out = np.zeros(band1.shape, dtype=band1.dtype.name)

    #win = 2*lag + 1
    #radius = int(win/2)

    #r = list(range(win))
    #for x in r:
        #x_off = x - radius

        #if x == min(r) or x == max(r):
            #y_r = r
        #else:
            #y_r = [max(r), min(r)]

        #for y in y_r:
            #y_off = y - radius

            #out += (band1 - band2[y_off:-y_off, x_off:-x_off])**2

    #return out
