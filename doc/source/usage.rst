=========================
Basic usage
=========================

Calculating textures
======================

- all similar parameters
- basic description influence of parameters
- input: dask arrays?! -> make wrapper to use normal np.arrays?!


.. code-block:: python

   import textory as tx
   import numpy as np
   import dask.array as da

   n = 50
   data = np.random.rand(n*n).reshape(n,n)
   data = da.from_array(data)

   tx.textures.variogram(data, lag=2, win_size=7)


Calculating non windowed statistics
===================================