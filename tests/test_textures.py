#! /usr/bin/python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from textory.textures import variogram, rodogram, madogram, pseudo_cross_variogram, window_statistic, tpi

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



def test_variogram_default_values_center(init_np_arrays):
    """Tests the center value of variogram with default parameters."""
    a, _ = init_np_arrays

    res = np.zeros_like(a)
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

    win_size = 5
    res = calc_for_window(tmp, win_size=win_size)
                        
    #normalize by number of neighbours
    num_neighbours = 8 #lag=1 case
    res = res / (2 * num_neighbours * win_size**2)

    check_pixel_index = 25
    assert variogram(a, lag=1, win_size=win_size, win_geom="square")[check_pixel_index, check_pixel_index] == res[check_pixel_index, check_pixel_index]


def test_pseudo_cross_variogram_default_values_center(init_np_arrays):
    """Tests the center value of variogram with default parameters."""
    a, b = init_np_arrays

    res = np.zeros_like(a)
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
                        tmp[i,j] += np.square((a[i, j] - b[i+l, j+k]))

    win_size = 5
    res = calc_for_window(tmp, win_size=win_size)
                        
    #normalize by number of neighbours
    num_neighbours = 8 #lag=1 case
    res = res / (2 * num_neighbours * win_size**2)

    check_pixel_index = 25
    assert pseudo_cross_variogram(a, b, lag=1, win_size=win_size, win_geom="square")[check_pixel_index, check_pixel_index] == res[check_pixel_index, check_pixel_index]


def test_madogram_default_values_center(init_np_arrays):
    """Tests the center value of rodogram with default parameters."""
    a, _ = init_np_arrays

    res = np.zeros_like(a)
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
                        tmp[i,j] += np.abs((a[i, j] - a[i+l, j+k]))

    win_size = 5
    res = calc_for_window(tmp, win_size=win_size)
                        
    #normalize by number of neighbours
    num_neighbours = 8 #lag=1 case
    res = res / (2 * num_neighbours * win_size**2)

    check_pixel_index = 25
    assert np.allclose(madogram(a, lag=1, win_size=win_size, win_geom="square")[check_pixel_index, check_pixel_index], res[check_pixel_index, check_pixel_index])


def test_rodogram_default_values_center(init_np_arrays):
    """Tests the center value of rodogram with default parameters."""
    a, _ = init_np_arrays

    res = np.zeros_like(a)
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
                        tmp[i,j] += np.sqrt(np.abs((a[i, j] - a[i+l, j+k])))

    win_size = 5
    res = calc_for_window(tmp, win_size=win_size)
                        
    #normalize by number of neighbours
    num_neighbours = 8 #lag=1 case
    res = res / (2 * num_neighbours * win_size**2)

    check_pixel_index = 25
    assert rodogram(a, lag=1, win_size=win_size, win_geom="square")[check_pixel_index, check_pixel_index] == res[check_pixel_index, check_pixel_index]


def test_window_statistic_std(init_np_arrays):
    """Tests the window statistic for standard deviation."""
    a, _ = init_np_arrays

    rows, cols = a.shape

    win_size = 5
    res = calc_for_window(a, func="nanstd", win_size=win_size)
                        
    check_pixel_index = 25
    assert np.allclose(window_statistic(a, stat="nanstd", win_size=win_size)[check_pixel_index, check_pixel_index], res[check_pixel_index, check_pixel_index])
