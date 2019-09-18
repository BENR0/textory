#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import functools
import dask.array as da
from scipy.ndimage.filters import convolve


def variogram(band1, band2, lag=None, window=None):
    band2 = np.pad(band2, ((1,1),(1,1)), mode="edge")
    
    out = np.zeros(band1.shape, dtype=band1.dtype.name)
    
    #left and right neighbour
    out = (band1 - band2[1:-1,2::])**2
    out += (band1 - band2[1:-1,0:-2:])**2
    #above and below neighbours
    out += (band1 - band2[2::,1:-1])**2
    out += (band1 - band2[0:-2,1:-1])**2
    #left diagonal neighbours
    out += (band1 - band2[0:-2,0:-2])**2
    out += (band1 - band2[2::,0:-2])**2
    #right diagonal neigbours
    out += (band1 -band2[0:-2,2::])**2
    out += (band1 - band2[2::,2::])**2
    
    return out


def view(offset_y, offset_x, size_y, size_x, step=1):
    """
    Calculates views for windowes operations on arrays.
    
    For windowed operations on arrays instead of looping through
    every numpy element and then do a second loop for the window,
    the element loop can be substituted by shifting the whole array
    against itself while looping through the window.
    
    In order to do this without padding the array first this implementation
    swaps views when the shift is "outside" the array dimensions.
    
   
    Parameters
    ----------
    offset_y : integer
        Row offset of current window row index from center point.
    offset_x : integer
        Column offset of current window column index from center point.
    size_y : integer
        Number of rows of array
    size_x : integer
        Number of columns of array
    step : integer, optional
    
    Returns
    -------
    tuple of 2D numpy slices
    
    
    Example
    -------
    window_size = 3

    radius = int(window/2)
 
    rows, columns = data.shape
    temp_sum = np.zeros((rows, columns))
 
    # Our window loop  
    for y in range(window):

    #we need offsets from centre !
        y_off = y - radius

        for x in range(window):
            x_off = x - radius
 
            view_in, view_out = view(y_off, x_off, rows, columns)
 
            temp_sum[view_out] += data[view_in]
    
    Notes
    -----
    Source: https://landscapearchaeology.org/2018/numpy-loops/
    """
 
    x = abs(offset_x)
    y = abs(offset_y)
 
    x_in = slice(x , size_x, step) 
    x_out = slice(0, size_x - x, step)
 
    y_in = slice(y, size_y, step)
    y_out = slice(0, size_y - y, step)
 
    # the swapping trick    
    if offset_x < 0: x_in, x_out = x_out, x_in                                 
    if offset_y < 0: y_in, y_out = y_out, y_in
 
    # return window view (in) and main view (out)
    return np.s_[y_in, x_in], np.s_[y_out, x_out]


def vario(arr1, arr2=None, lag=1):
    """
    Calculates the (pseudo-) variogram between two arrays.
    
    If only one array is supplied variogram is calculated
    for itself (same array is used as the second array).
    
    Parameters
    ----------
    arr1 : np.array
    arr2 : np.array, optional
    lag : int, optional
        The lag distance for the variogram, defaults to 1.
    
    Returns
    -------
    np.array
        Variogram
    
    """
    win = 2*lag + 1
    radius = int(win/2)
    rows, cols = arr1.shape
    
    arr1 = np.asarray(arr1)
    
    if arr2 is None:
        arr2 = arr1.copy()
    
    out_arr = np.zeros(arr1.shape, dtype=arr1.dtype.name)

    r = list(range(win))
    for x in r:
        x_off = x - radius

        if x == min(r) or x == max(r):
            y_r = r
        else:
            y_r = [max(r), min(r)]
        
        for y in y_r:
            y_off = y - radius
            
            view_in, view_out = view(y_off, x_off, rows, cols)
            out_arr[view_out] += (arr1[view_out] - arr2[view_in])**2
            
    return out_arr

def vario1(arr1, arr2=None, lag=1):
    """
    Calculates the (pseudo-) variogram between two arrays.
    
    If only one array is supplied variogram is calculated
    for itself (same array is used as the second array).
    
    Parameters
    ----------
    arr1 : np.array
    arr2 : np.array, optional
    lag : int, optional
        The lag distance for the variogram, defaults to 1.
    
    Returns
    -------
    np.array
        Variogram
    
    """
    twoband = False
    win = 2*lag + 1
    radius = int(win/2)
    
    #if arr2 is None:
    #    arr2 = arr1.copy()
    inshape0 = arr1.shape[0]
    if len(arr1.shape) == 3:# and inshape0 == 2:
        input1 = arr1[0,:,:]
        input2 = arr1[1,:,:]
        twoband = True
    #elif len(arr1.shape) == 2:
    #    print(arr1.shape)
    #    input1 = arr1
    #    input2 = arr1.copy()
    elif arr2 is not None:
        #Raise error only two bands are allowed
        #pass
        input1 = arr1
        input2 = arr2
    
    
    input1 = np.asarray(input1)
    rows, cols = input1.shape
    
    out_arr = np.zeros(input1.shape, dtype=input1.dtype.name)
    
    r = list(range(win))
    for x in r:
        x_off = x - radius

        if x == min(r) or x == max(r):
            y_r = r
        else:
            y_r = [max(r), min(r)]
        
        for y in y_r:
            y_off = y - radius
            
            #view_in, view_out = view(y_off, x_off, rows, cols)
             
            x_in = slice(abs(x_off) , cols, 1) 
            x_out = slice(0, cols - abs(x_off), 1)

            y_in = slice(abs(y_off), rows, 1)
            y_out = slice(0, rows - abs(y_off), 1)

            # the swapping trick    
            if x_off < 0: x_in, x_out = x_out, x_in                                 
            if y_off < 0: y_in, y_out = y_out, y_in

            # return window view (in) and main view (out)
            #return np.s_[y_in, x_in], np.s_[y_out, x_out]
            out_arr[y_out, x_out] += (input1[y_out, x_out] - input2[y_in, x_in])**2
   
    if twoband:
        arr1[0,:,:] = out_arr
        return arr1
    else:
        return out_arr


def num_neighbours(lag=1):
    """
    Calculate number of neigbour pixels for a given lag.
    
    Parameters
    ----------
    lag : int
        Lag distance, defaults to 1.
        
    Returns
    -------
    int
        Number of neighbours
    """
    win_size = 2*lag + 1
    neighbours = win_size**2 - (2*(lag-1) + 1)**2
    
    return neighbours

def con(x, n=5):
    """
    Convolve array x with square kernel of size n.
    
    Parameters
    ----------
    x : np.array
    n : int, optional
        Kernel size, defaults to 5.
    
    Returns
    -------
    np.array
    """
    k = np.ones((n,n))
    
    return convolve(x, k)


def global_vario(x, lag=1):
    """
    Calculate variogram with specified lag for array.
    
    Parameters
    ----------
    x : np.array
    lag : int, optional
    
    Returns
    -------
    variogram : float
    """
    #new = ddata.map_overlap(pvario, depth = 1)
    v = vario(x, lag=lag)
    
    res = np.sum(v)
    
    #calculate 1/2N part of variogram
    neighbours = num_neighbours(lag)
    
    cols, rows = x.shape
    num_pix = cols * rows
    
    factor = 1.0/(2 * num_pix * neighbours)
    
    return factor * res

def _vario_difference(x, y=None, lag=1):
    """
    Calculate windowed variogram with specified lag for array.
    
    Parameters
    ----------
    x : np.array
    y : np.array, optional
        Defaults to None
    lag : int, optional
    
    Returns
    -------
    np.array
        Difference part of variogram calculations
    """
    #v = vario1(x, lag=lag)
    
    #pvario = functools.partial(vario1, arr2=None, lag=lag)
    pvario = functools.partial(vario1, lag=lag)
    
    #olap = da.overlap.overlap(x, depth={0: 0, 1: lag, 2: lag}, boundary={1: "reflect", 2: "reflect"})
    #v = olap[0,:,:]
    #w = olap[1,:,:]
    if y is None:
        x = da.overlap.overlap(x, depth={0: lag, 1: lag}, boundary={0: "reflect", 1: "reflect"})
        y = x
    else:
        x = da.overlap.overlap(x, depth={0: lag, 1: lag}, boundary={0: "reflect", 1: "reflect"})
        y = da.overlap.overlap(y, depth={0: lag, 1: lag}, boundary={0: "reflect", 1: "reflect"})
    
    res = da.map_blocks(pvario, x, y)
    res = da.overlap.trim_internal(res, {0: lag, 1: lag})
    
    return res


def variogram_texture(x, lag=1, window=5):
    """
    Calculate moveing window variogram with specified
    lag for array.
    
    Parameters
    ----------
    x : array like
        Input array
    lag : int
        Lag distance for variogram, defaults to 1.
    window : int, optional
        Length of one side of window. Window will be of size window*window.
    
    Returns
    -------
    array like
        Array where each element is the variogram of the window around the element
    """
    diff = _vario_difference(x, lag=lag)
    
    #create convolve function with reduced parameters for mapping
    pcon = functools.partial(con, n=window)
    
    conv_padding = int(window//2)
    res = diff.map_overlap(pcon, depth={0: conv_padding, 1: conv_padding})
    
    #calculate 1/2N part of variogram
    neighbours = num_neighbours(lag)
    
    num_pix = window**2
    
    factor = 2 * num_pix * neighbours
    
    return res / factor

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
    diff = _vario_difference(x, lag=lag)
    
    res = np.sum(diff)
    
    #calculate 1/2N part of variogram
    neighbours = num_neighbours(lag)
    
    cols, rows = x.shape
    num_pix = cols * rows
    
    factor = 2 * num_pix * neighbours
    
    return res / factor


def pseudo_variogram_texture(x, y, lag=1, window=5):
    """
    Calculate moveing window pseudo-variogram with specified
    lag for the two arrays.
    
    Parameters
    ----------
    x : array like
        Input array
    lag : int
        Lag distance for variogram, defaults to 1.
    window : int, optional
        Length of one side of window. Window will be of size window*window.
    
    Returns
    -------
    array like
        Array where each element is the pseudo-variogram
        between the two arrays of the window around the element.
    """
    diff = _vario_difference(x, y, lag, window)
    
    #create convolve function with reduced parameters for mapping
    pcon = functools.partial(con, n=window)
    
    conv_padding = int(window//2)
    res = diff.map_overlap(pcon, depth={0: conv_padding, 1: conv_padding})
    
    #calculate 1/2N part of variogram
    neighbours = num_neighbours(lag)
    
    num_pix = window**2
    
    factor = 2 * num_pix * neighbours
    
    return res / factor


def pseudo_variogram(x, y, lag=1):
    """
    Calculate pseudo-variogram with specified lag for 
    the two arrays.
    
    Parameters
    ----------
    x : array like
        Input array
    lag : int
        Lag distance for variogram, defaults to 1.
    
    Returns
    -------
    float
        Pseudo-variogram between the two arrays
    """
    diff = _vario_difference(x, y, lag)
    
    res = np.sum(diff)
    
    #calculate 1/2N part of variogram
    neighbours = num_neighbours(lag)
    
    cols, rows = x.shape
    num_pix = cols * rows
    
    factor = 2 * num_pix * neighbours
    
    return res / factor
