#! /usr/bin/python
# -*- coding: utf-8 -*-
import functools
import decorator
import numpy as np
import dask.array as da
import xarray as xr
import skimage as ski
from scipy.ndimage.filters import convolve
#import bottlenack as bn


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
    # for y in range(window):

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

    x_in = slice(x, size_x, step)
    x_out = slice(0, size_x - x, step)

    y_in = slice(y, size_y, step)
    y_out = slice(0, size_y - y, step)

    # the swapping trick
    if offset_x < 0:
        x_in, x_out = x_out, x_in
    if offset_y < 0:
        y_in, y_out = y_out, y_in

    # return window view (in) and main view (out)
    return np.s_[y_in, x_in], np.s_[y_out, x_out]


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
    win_size = 2 * lag + 1
    neighbours = win_size**2 - (2 * (lag - 1) + 1)**2

    return neighbours


def neighbour_count(shape, kernel):
    """
    Count the number of contributing pixels based on a kernel for
    an array which gets convolved with that kernel.

    This function gives precise count on the edges too.

    Parameters
    ----------
    shape : tuple
        Shape of the array for which counts should be given
    kernel : np.array
        Each element in the array equal to 1 increases count by 1.

    Returns
    -------
    np.array
        Array with counts
    """
    win_size = kernel.shape[0]
    t = np.ones((win_size, win_size), dtype=np.int)
    corner_top_left = np.zeros_like(t)
    k = kernel
    center = win_size // 2
    #k[center, center] = 0

    convolve(t, k, output=corner_top_left, mode="constant", cval=0)
    corner_top_left = corner_top_left[0:center + 1, 0:center + 1]

    #shape / 2 in each dimension - (center+1) needs to be padded
    pad_size = np.array(shape) / 2 - (center + 1)
    y_pad, x_pad = pad_size.astype(np.int)

    one = np.pad(corner_top_left, ((0, y_pad), (0, x_pad)), mode="edge")
    #three = np.pad(corner_top_left[::-1,:], ((y_pad,0),(0,x_pad)), mode="edge")
    #two = np.pad(corner_top_left[:,::-1], ((0,y_pad),(x_pad,0)), mode="edge")
    #four = np.pad(corner_top_left[::-1,::-1], ((y_pad,0),(x_pad,0)), mode="edge")
    three = one[::-1, :]
    two = one[:, ::-1]
    four = one[::-1, ::-1]

    counts = np.block([[one, two], [three, four]])

    return counts


def create_kernel(n=5, geom="square", kernel=None):
    """
    Create a kernel of size n.

    Parameters
    ----------
    n : int, optional
        Kernel size, defaults to 5.
    geom : {"square", "round"}
        Geometry of the kernel. Defaults to square.
    kernel : np.array, optional
        Custom kernel to convolve with. If kernel argument is given
        parameters n and geom are ignored.

    Returns
    -------
    np.array
    """
    if kernel is None:
        if geom == "square":
            k = np.ones((n, n))
        elif geom == "round":
            xind, yind = np.indices((n, n))
            c = n // 2
            center = (c, c)
            radius = n / 2

            circle = (xind - center[0])**2 + (yind - center[1])**2 < radius**2
            k = circle.astype(np.int)
    else:
        k = kernel

    return k


def nd_variogram(x, y):
    """
    Inner most calculation step of variogram and pseudo-cross-variogram

    This function is used in the inner most loop of `neighbour_diff_squared`.

    Parameters
    ----------
    x, y : np.array

    Returns
    -------
    np.array

    """
    res = np.square(x - y)

    return res


def nd_madogram(x, y, *args):
    """
    Inner most calculation step of madogram

    This function is used in the inner most loop of `neighbour_diff_squared`.

    Parameters
    ----------
    x, y : np.array

    Returns
    -------
    np.array

    """
    res = np.abs(x - y)

    return res


def nd_rodogram(x, y, *args):
    """
    Inner most calculation step of rodogram

    This function is used in the inner most loop of `neighbour_diff_squared`.

    Parameters
    ----------
    x, y : np.array

    Returns
    -------
    np.array

    """
    res = np.sqrt(np.abs(x - y))

    return res


def nd_cross_variogram(x1, y2, x2, y1):
    """
    Inner most calculation step of cross-variogram

    This function is used in the inner most loop of `neighbour_diff_squared`.

    Parameters
    ----------
    x, y : np.array

    Returns
    -------
    np.array

    """
    res = (x1 - x2) * (y1 - y2)

    return res


def neighbour_diff_squared(arr1, arr2=None, lag=1, func="nd_variogram"):
    """
    Calculates the squared difference between a pixel and its neighbours
    at the specified lag.

    If only one array is supplied variogram is calculated
    for itself (same array is used as the second array).

    Parameters
    ----------
    arr1 : np.array
    arr2 : np.array, optional
    lag : int, optional
        The lag distance for the variogram, defaults to 1.
    func : {nd_variogram, nd_pseudo_cross_variogram, nd_madogram, nd_rodogram, nd_cross_variogram}
        Calculation method of innermost step of the different variogram methods.

    Returns
    -------
    np.array
        Variogram

    """
    method = globals()[func]

    win = 2 * lag + 1
    radius = win // 2
    rows, cols = arr1.shape

    if arr2 is None:
        arr2 = arr1.copy()

    out_arr = np.zeros_like(arr1)

    r = list(range(win))
    for y in r:
        y_off = y - radius

        if y == min(r) or y == max(r):
            x_r = r
        else:
            x_r = [max(r), min(r)]

        for x in x_r:
            x_off = x - radius
            view_in, view_out = view(y_off, x_off, rows, cols)
            if func == "nd_cross_variogram":
                out_arr[view_out] += method(arr1[view_out], arr2[view_in], arr1[view_in], arr2[view_out])
            else:
                out_arr[view_out] += method(arr1[view_out], arr2[view_in])

            #out_arr[view_out] += method(arr1[view_out], arr2[view_in])
            #a1 = arr1[view_out]
            #a2 = arr2[view_in]
            #out_arr[view_out] += (a1 - a2)**2

    return out_arr


def _dask_neighbour_diff_squared(x, y=None, lag=1, func="nd_variogram"):
    """
    Calculate quared difference between pixel and its
    neighbours at specified lag for dask arrays

    Parameters
    ----------
    x : np.array
    y : np.array, optional
        Defaults to None
    lag : int, optional
    func : {nd_variogram, nd_madogram, nd_rodogram, nd_cross_variogram}
        Calculation method of innermost step of different variogram methods.

    Returns
    -------
    np.array
        Difference part of variogram calculations
    """
    pvario = functools.partial(neighbour_diff_squared, lag=lag, func=func)

    if y is None:
        x = da.overlap.overlap(x, depth={0: lag, 1: lag}, boundary={0: "reflect", 1: "reflect"})
        y = x
    else:
        x = da.overlap.overlap(x, depth={0: lag, 1: lag}, boundary={0: "reflect", 1: "reflect"})
        y = da.overlap.overlap(y, depth={0: lag, 1: lag}, boundary={0: "reflect", 1: "reflect"})

    res = da.map_blocks(pvario, x, y)
    res = da.overlap.trim_internal(res, {0: lag, 1: lag})

    return res


def convolution(x, win_size=5, win_geom="square", kernel=None, **kwargs):
    """
    Convolute array with kernel and normalize by count of kernel
    elements > 0.

    Parameters
    ----------
    x : array like
        Input array
    win_size : int, optional
        Length of one side of window. Window will be of size window*window.
        Defaults to 5.
    geom : {"square", "round"}
        Geometry of the kernel. Defaults to square.
    kernel : np.array, optional
        Custom kernel to use for convolution. If specified `geom` and `win_size`
        parameter will be ignored.

    Returns
    -------
    array like
        Array where each element is the variogram of the window around the element

    """
    if kernel is not None:
        k = kernel
    else:
        k = create_kernel(n=win_size, geom=win_geom)

    #create convolve function with reduced parameters for map_overlap
    pcon = functools.partial(convolve, weights=k, mode="constant", cval=0.0)

    if isinstance(x, da.core.Array):
        conv_padding = int(win_size // 2)
        res = x.map_overlap(pcon, depth={0: conv_padding, 1: conv_padding}, boundary={0: 0.0, 1: 0.0})
    else:
        res = pcon(x)

    kernel_significant_elements = np.where(k > 0, 1, 0)
    num_pix = np.sum(kernel_significant_elements)

    return res / num_pix


def window_sum(x, lag=1, win_size=5, win_geom="square", kernel=None):
    """
    Calculate the window sum for the various textures

    Parameters
    ----------
    x : array like
        Input array
    lag : int
        Lag distance for variogram, defaults to 1.
    win_size : int, optional
        Length of one side of window. Window will be of size window*window.
        Defaults to 5.
    geom : {"square", "round"}
        Geometry of the kernel. Defaults to square.
    kernel : np.array, optional
        Custom kernel to use for convolution. If specified `geom` and `win_size`
        parameter will be ignored.

    Returns
    -------
    array like
        Array where each element is the variogram of the window around the element

    """
    res = convolution(x, win_size=win_size, win_geom=win_geom, kernel=kernel)

    #calculate 1/2N part of variogram
    neighbours = num_neighbours(lag)

    factor = 2 * neighbours

    return res / factor


def _win_view_stat(x, win_size=5, stat="nanmean", **kwargs):
    """
    Calculates specified basic statistical measure for a moveing window
    over an array.

    Parameters
    ----------
    x : np.array
    win_size : int, optional
        Window size, defaults to 5.
    stat : {"nanmean", "nanmax", "nanmin", "nanmedian", "nanstd"}
        Statistical measure to calculate.
    kwargs : optional
        Additional keyword arguments some stat may need.

    Returns
    -------
    np.array

    """
    #if x.shape == (1, 1):
        #return x

    np_measure = getattr(np, stat)

    measure = functools.partial(np_measure, **kwargs) 

    pad = int(win_size // 2)
    data = np.pad(x, (pad, pad), mode="constant", constant_values=(np.nan))

    #sh = np.asarray(x).shape
    #mask = np.zeros_like(x)
    #mask[pad:sh[0]-pad, pad:sh[1]-pad] = 1

    #data = np.where(mask==1, x, np.nan)

    #get windowed view of array
    windowed = ski.util.view_as_windows(data, (win_size, win_size))

    #calculate measure over last to axis
    res = measure(windowed, axis=(2, 3))

    return res


#def xr_wrapper(fun):
    ##functools wraps keeps docstrings
    #@functools.wraps(fun)
    #def wrapped_fun(*args, **kwargs):
        #if isinstance(args[0], xr.core.dataarray.DataArray):
            #out = args[0].copy()
            #if len(args) == 2:
                #out.data = fun(args[0].data, args[1].data, **kwargs)
                #out.attrs["name"] = fun.__name__ + "_{}_{}".format(args[0].attrs["name"], args[1].attrs["name"])
            #else:
                #out.data = fun(args[0].data, **kwargs)
                #out.attrs["name"] = fun.__name__ + "_{}".format(args[0].attrs["name"])

            #if fun.__name__ == "window_statistic":
                #out.attrs["statistic"] = kwargs.get("stat")
                #out.name = out.attrs["name"] + "_{stat}_{win_size}".format(**kwargs)
            #else:
                #out.attrs["lag_distance"] = kwargs.get("lag")
                #out.attrs["window_geometry"] = kwargs.get("win_geom")
                #out.name = out.attrs["name"] + "_{lag}_{win_size}_{win_geom}".format(**kwargs)

            #out.attrs["window_size"] = kwargs.get("win_size")
        #else:
            #if len(args) == 2:
                #out = fun(args[0], args[1], **kwargs)
            #else:
                #out = fun(args[0], **kwargs)

        #return out
    #return wrapped_fun

@decorator.decorator
def xr_wrapper(fun, *args, **kwargs):
    import inspect

    run_sig = inspect.getfullargspec(fun)
    params = dict(zip(run_sig.args, args))
    params.pop("x")

    if isinstance(args[0], xr.core.dataarray.DataArray):
        out = args[0].copy()
        x_input = args[0]
        if "name" not in x_input.attrs.keys():
            x_input.attrs["name"] = "Input array"
        if "y" in params.keys():
            y_input = args[1]
            if "name" not in y_input.attrs.keys():
                y_input.attrs["name"] = "Input array"

            params.pop("y")
            out.data = fun(x_input.data, y_input.data, **params)
            out.attrs["name"] = fun.__name__ + "_{}_{}".format(x_input.attrs["name"], y_input.attrs["name"])
        else:
            out.data = fun(x_input.data, **params)
            out.attrs["name"] = fun.__name__ + "_{}".format(x_input.attrs["name"])

        if fun.__name__ == "window_statistic":
            out.attrs["statistic"] = params.get("stat")
            out.name = out.attrs["name"] + "_{stat}_{win_size}".format(**params)
        elif fun.__name__ == "tpi":
            out.name = out.attrs["name"] + "_{win_size}".format(**params)
        else:
            out.attrs["lag_distance"] = params.get("lag")
            out.attrs["window_geometry"] = params.get("win_geom")
            out.name = out.attrs["name"] + "_{lag}_{win_size}_{win_geom}".format(**params)

        out.attrs["window_size"] = params.get("win_size")
    else:
        if "y" in params.keys():
            out = fun(args[0], **params, **kwargs)
        else:
            out = fun(args[0], **params, **kwargs)

    return out


##########################
#alternative version test
##########################
#def neighbour_diff_squared1(arr1, arr2=None, lag=1):
    #"""
    #Calculates the (pseudo-) variogram between two arrays.

    #If only one array is supplied variogram is calculated
    #for itself (same array is used as the second array).

    #Parameters
    #----------
    #arr1 : np.array
    #arr2 : np.array, optional
    #lag : int, optional
        #The lag distance for the variogram, defaults to 1.

    #Returns
    #-------
    #np.array
        #Variogram

    #"""
    #twoband = False
    #win = 2*lag + 1
    #radius = int(win/2)

    ##if arr2 is None:
    ##    arr2 = arr1.copy()
    #inshape0 = arr1.shape[0]
    #if len(arr1.shape) == 3:# and inshape0 == 2:
        #input1 = arr1[0,:,:]
        #input2 = arr1[1,:,:]
        #twoband = True
    ##elif len(arr1.shape) == 2:
    ##    print(arr1.shape)
    ##    input1 = arr1
    ##    input2 = arr1.copy()
    #elif arr2 is not None:
        ##Raise error only two bands are allowed
        ##pass
        #input1 = arr1
        #input2 = arr2

    #input1 = np.asarray(input1)
    #rows, cols = input1.shape

    #out_arr = np.zeros(input1.shape, dtype=input1.dtype.name)

    #r = list(range(win))
    #for x in r:
        #x_off = x - radius

        #if x == min(r) or x == max(r):
            #y_r = r
        #else:
            #y_r = [max(r), min(r)]

        #for y in y_r:
            #y_off = y - radius

            ##view_in, view_out = view(y_off, x_off, rows, cols)

            #x_in = slice(abs(x_off) , cols, 1)
            #x_out = slice(0, cols - abs(x_off), 1)

            #y_in = slice(abs(y_off), rows, 1)
            #y_out = slice(0, rows - abs(y_off), 1)

            ## the swapping trick
            #if x_off < 0: x_in, x_out = x_out, x_in
            #if y_off < 0: y_in, y_out = y_out, y_in

            ## return window view (in) and main view (out)
            ##return np.s_[y_in, x_in], np.s_[y_out, x_out]
            #out_arr[y_out, x_out] += (input1[y_out, x_out] - input2[y_in, x_in])**2

    #if twoband:
        #arr1[0,:,:] = out_arr
        #return arr1
    #else:
        #return out_arr
