=========================
Basic usage
=========================

Calculating textures
======================

All available textures are located in the ``textures`` submodule and except for the windowed basic
statistics :func:`~textory.textures.window_statistic`, have the same parameters.

First we import necessary packages and create some dummy data:

.. code-block:: python

   import textory as tx
   import numpy as np

   n = 50
   data1 = np.random.rand(n*n).reshape(n,n)
   data2 = np.random.rand(n*n).reshape(n,n)


Then we can calculate, for example the variogram, with:

.. code-block:: python

   tx.textures.variogram(x=data1, lag=2, win_size=7)

Here the parameter `win_geom` is ommitted and therefore defaults to "square". The ``lag`` and ``win_size``
parameters can also be omitted in which case they default to `1` and `5` respectively.


Textures based on two images like :func:`~textory.textures.pseudo_cross_variogram` have an additional
parameter ``y``:

.. code-block:: python

   tx.textures.pseudo_cross_variogram(x=data1, y=data2)


With the :func:`~textory.textures.window_statistic` function basic statistical measures like min, max
median, etc. can be calculated for a moving window. To get accurate results the nan version of the numpy
functions should be used. Currently this function only supports square windows.

.. code-block:: python

   tx.textures.window_statistic(x=data1, stat="nanmax")


