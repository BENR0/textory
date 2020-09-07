#! /usr/bin/python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from textory.util import neighbour_diff_squared, num_neighbours, neighbour_count, create_kernel
from textory.statistics import variogram, pseudo_cross_variogram

@pytest.fixture
def init_np_arrays():
    """Inits two random np arrays"""
    np.random.seed(42)

    n = 50

    a1 = np.random.random((n,n)) * 157
    a2 = np.random.random((n,n)) * 237

    return a1.astype(np.float32), a2.astype(np.float32)


def test_variogram(init_np_arrays):
    """THIS TEST ONLY COVERS THE VERSION WITH INEXACT NEIGHBOUR COUNT ON THE EDGES
    This test needs improvement in calculation and what is tested.
    Much code is shared with the "neighbour_diff_squared" test in test_util.
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

    tmp = np.nansum(tmp)

    res = tmp / 40000

    assert variogram(a, lag=1) == res


def test_pseudo_cross_variogram(init_np_arrays):
    """THIS TEST ONLY COVERS THE VERSION WITH INEXACT NEIGHBOUR COUNT ON THE EDGES
    This test needs improvement in calculation and what is tested.
    Much code is shared with the "neighbour_diff_squared" test in test_util.
    """

    a, b = init_np_arrays

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

    tmp = np.nansum(tmp)

    res = tmp / 40000

    assert pseudo_cross_variogram(a, b, lag=1) == res
