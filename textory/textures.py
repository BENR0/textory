#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import functools
import dask.array as da
from scipy.ndimage.filters import convolve

from .util import view, create_kernel, num_neighbours, _dask_neighbour_diff_squared, _win_view_stat


def variogram(x, lag=1, win_size=5, win_geom="square", **kwargs):
    """
    Calculate moveing window variogram with specified
    lag for array.
    
    Parameters
    ----------
    x : array like
        Input array
    lag : int
        Lag distance for variogram, defaults to 1.
    win_size : int, optional
        Length of one side of window. Window will be of size window*window.
    geom : {"square", "round"}
        Geometry of the kernel. Defaults to square.
    
    Returns
    -------
    array like
        Array where each element is the variogram of the window around the element
    """
    diff = _dask_neighbour_diff_squared(x, lag=lag)

    k = create_kernel(n=win_size, geom=win_geom)

    #create convolve function with reduced parameters for mapping
    pcon = functools.partial(convolve, weights=k)
    
    conv_padding = int(win_size//2)
    res = diff.map_overlap(pcon, depth={0: conv_padding, 1: conv_padding})
    
    #calculate 1/2N part of variogram
    neighbours = num_neighbours(lag)
    
    num_pix = np.sum(k)
    
    factor = 2 * num_pix * neighbours
    
    return res / factor


def pseudo_cross_variogram(x, y, lag=1, win_size=5, win_geom="square", **kwargs):
    """
    Calculate moveing window pseudo-variogram with specified
    lag for the two arrays.
    
    Parameters
    ----------
    x : array like
        Input array
    lag : int
        Lag distance for variogram, defaults to 1.
    win_size : int, optional
        Length of one side of window. Window will be of size window*window.
    geom : {"square", "round"}
        Geometry of the kernel. Defaults to square.
    
    Returns
    -------
    array like
        Array where each element is the pseudo-variogram
        between the two arrays of the window around the element.
    """
    diff = _dask_neighbour_diff_squared(x, y, lag)
    
    k = create_kernel(n=win_size, geom=win_geom)

    #create convolve function with reduced parameters for mapping
    pcon = functools.partial(convolve, weights=k)
    
    conv_padding = int(win_size//2)
    res = diff.map_overlap(pcon, depth={0: conv_padding, 1: conv_padding})
    
    #calculate 1/2N part of variogram
    neighbours = num_neighbours(lag)
    
    num_pix = np.sum(k)
    
    factor = 2 * num_pix * neighbours
    
    return res / factor


def window_statistic(x, stat="nanmean", win_size=5, win_geom="square", **kwargs):
    """
    Calculate moveing window pseudo-variogram with specified
    lag for the two arrays.
    
    Parameters
    ----------
    x : array like
        Input array
    stat : {"nanmean", "nanmax", "nanmin", "nanmedian", "nanstd"}
        Statistical measure to calculate.
    win_size : int, optional
        Length of one side of window. Window will be of size window*window.
    geom : {"square", "round"}
        Geometry of the kernel. Defaults to square.
    
    Returns
    -------
    array like

    Todo
    ----
    - checking if array dimensions are multiple of win_size. pad if not
    - make sure that each chunk is multiple of win_size in map_overlap
    """
    if win_size%2 == 0:
        raise("Window size must be odd.")

    #create view_as_windows function with reduced parameters for mapping
    pcon = functools.partial(_win_view_stat, win_size=win_size, stat=stat)
    
    conv_padding = int(win_size//2)
    res = x.map_overlap(pcon, depth={0: conv_padding, 1: conv_padding}, boundary={0: np.nan, 1: np.nan})#, trim=False)

    return res


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


