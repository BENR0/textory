=========================
Usage
=========================

Calculating textures
======================


.. code-block:: python

   import textory as tx
   import numpy as np
   import dask.array as da

   n = 50
   data = np.random.rand(n*n).reshape(n,n)
   data = da.from_array(data)

   tx.textures.variogram(data, lag=2, win_size=7)
