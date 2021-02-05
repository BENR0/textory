=========================
Performance
=========================

In general performance of Textory shold be good. For very small arrays, even
though performance should be quite similiar, it is slightly faster if the textures
are used with numpy arrays instead of dask arrays since they have a small overhead
(see Dask documentation for details about that).

One thing to note for is; for very large windows, even though textory can use
dask, memory might be a problem. Of course the exact window size limit depends on
the actuall amount of memory available in the system used. For example for a system
with 16Gb of memory window sizes below ~190 should be possible.

The reason for this problem is the usage of :func:`~scipy.ndimage.filters.convolve`
which is know to have inefficient memory management for large window sizes.

Some hints for calculating with large windows:

- choose smaller chunk sizes than normaly would be efficent
- reduce the number of chunks processed at the same time. For a local Dask cluster
  limit the number of threads for example

  .. code-block:: python

     import dask
     from multiprocessing.pool import ThreadPool

     dask.config.set(pool=ThreadPool(1))

Of course limiting the number of chunks processed at the same time mitigates
the performance increase normally seen while using Dask.
