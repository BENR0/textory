#! /usr/bin/python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import xarray as xr
from textory.util import neighbour_diff_squared, num_neighbours, neighbour_count, create_kernel,\
convolution, xr_wrapper

@pytest.fixture
def init_np_arrays():
    """Inits two random np arrays"""
    np.random.seed(42)

    n = 50

    a1 = np.random.random((n,n)) * 157
    a2 = np.random.random((n,n)) * 237

    return a1.astype(np.float32), a2.astype(np.float32)


def calc_for_window(a, func="sum", win_size=5):
    """Calculate function for window.

    Parameters
    ----------
    a : np.array
    func : function
        Function to calculate for window.
    win_size : int
        Window size

    Returns
    -------
    np.array

    """
    measure = getattr(np, func)
    res = np.zeros_like(a)
    rows, cols = a.shape
    win_offset = int(win_size // 2)
    for i in range(0, cols):
        for j in range(0, rows):
            ipos_start = i - win_offset
            ipos_end = i + win_offset + 1
            if (ipos_start < 0):
                ipos_start = 0
            if (ipos_end >= cols):
                ipos_end = cols

            jpos_start = j - win_offset
            jpos_end = j + win_offset + 1
            if (jpos_start < 0):
                jpos_start = 0
            elif (jpos_end >= rows):
                jpos_end = rows

            i_slice = slice(ipos_start, ipos_end)
            j_slice = slice(jpos_start, jpos_end)

            res[i, j] = measure(a[i_slice, j_slice])

    return res


def test_num_neighbours():
    assert num_neighbours(lag=1) == 8
    assert num_neighbours(lag=2) == 16


def test_neighbour_count():
    """
    Currently only test one window size with each elemten in the
    window contributing.

    Todo
    ----
    - test different window sizes
    - test kernels where not each element is 1 (i.e. with non contributing elements)
    """
    shape = (50, 50)
    kernel = np.ones((5, 5))
    edge_count_top_left = np.array([[9, 12, 15],
                                    [12, 16, 20],
                                    [15, 20, 25]])

    assert np.array_equal(neighbour_count(shape=shape, kernel=kernel)[0:3, 0:3],  edge_count_top_left)


def test_create_kernel_squared():
    assert np.allclose(create_kernel(n=5, geom="square"), np.ones((5, 5)))
    assert np.allclose(create_kernel(n=13, geom="square"), np.ones((13, 13)))


def test_create_kernel_round():
    round_5 = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]])
    assert np.allclose(create_kernel(n=5, geom="round"), round_5)

    round_13 = np.array([[0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
  					     [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
					     [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
					     [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
					     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
					     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
					     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
					     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
					     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
					     [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
					     [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
					     [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
					     [0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0.]])
    assert np.allclose(create_kernel(n=13, geom="round"), round_13)


def test_create_kernel_custom():
    custom = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]])
    assert np.allclose(create_kernel(kernel=custom), custom)


def test_create_kernel_even_size():
    custom = np.array([[1, 0, 0, 0, 1, 0],
                       [0, 1, 0, 1, 0, 1],
                       [0, 0, 1, 0, 0, 1],
                       [0, 1, 0, 1, 0, 0],
                       [0, 1, 0, 0, 0, 1],
                       [1, 0, 0, 0, 1, 0]])

    with pytest.raises(ValueError):
        create_kernel(kernel=custom)

    with pytest.raises(ValueError):
        create_kernel(kernel=custom[:, 0:-1])

    with pytest.raises(ValueError):
        create_kernel(n=6)


def test_neighbour_diff_squared(init_np_arrays):
    """
    Todo:
        - add case for NaN's in array
    """
    a, _ = init_np_arrays

    tmp = np.zeros_like(a)
    lag = 1
    lags =  range(-lag, lag + 1)

    rows, cols = a.shape

    #calculate variogram difference
    for i in range(0, cols):
        for j in range(0, rows):
            for l in lags:
                for k in lags:
                    if (i+l < 0) | (i+l >= cols) | (j+k < 0) | (j+k >= rows) | ((l == 0) & (k == 0)):
                        continue
                    else:
                        tmp[i,j] += np.square((a[i, j] - a[i+l, j+k]))

    assert np.allclose(neighbour_diff_squared(a, arr2=None, lag=1, func="nd_variogram"), tmp)
    #assert neighbour_diff_squared(a, arr2=None, lag=1, func="nd_variogram")[24, 24] == tmp[24,24]


def test_convolution(init_np_arrays):
    a, _ = init_np_arrays

    custom_kernel = np.ones((3,3))
    res = convolution(a, kernel=custom_kernel)
    assert res[25,25] == np.sum(a[24:27, 24:27] / 9)


def test_xr_wrapper(init_np_arrays):
    
    a, b = init_np_arrays

    a = xr.DataArray(a, dims=["x", "y"])
    a.attrs["name"] = "a"
    a.attrs["first_attr"] = "test_1"
    a.attrs["second_attr"] = "test_2"
    b = xr.DataArray(b, dims=["x", "y"])
    b.attrs["name"] = "b"

    @xr_wrapper
    def fun(x, y, win_size, win_geom, lag):
        return x

    res = fun(a, b, win_size=5, win_geom="square", lag=1)

    assert res.name == "fun_a_b_1_5_square"
    assert res.attrs["name"] == "fun_a_b"
    assert res.attrs["window_size"] == 5
    assert res.attrs["window_geometry"] == "square"
    assert res.attrs["lag_distance"] == 1
    assert res.attrs["first_attr"] == "test_1"
    assert res.attrs["second_attr"] == "test_2"


    @xr_wrapper
    def tpi(x, win_size=5):
        return x

    res = tpi(a)

    assert res.name == "tpi_a_5"
    assert res.attrs["name"] == "tpi_a"
    assert res.attrs["window_size"] == 5
    assert res.attrs["first_attr"] == "test_1"
    assert res.attrs["second_attr"] == "test_2"


    @xr_wrapper
    def window_statistic(x, stat="test", win_size=5):
        return x

    res = window_statistic(a)

    assert res.name == "window_statistic_a_test_5"
    assert res.attrs["name"] == "window_statistic_a"
    assert res.attrs["window_size"] == 5
    assert res.attrs["first_attr"] == "test_1"
    assert res.attrs["second_attr"] == "test_2"
