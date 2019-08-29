Textory
=======
With Textory local (windowed) textures can be calculated for images (array like structures).
It has conveniant wrappers to be used with xarray or Satpy scenes.

Currently implemented are:
- variogram
- pseudo-variogram

Todo
----
Implement:
- Rodogram
- Madogram
- Crossvariogram
- Local Binary patterns (also between two images)
  - https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
- other textures?
- Add parameters for round windows (possible with current code but hardcoded to rectangular windows)


Installation
------------

Textory can be installed with pip after cloning this repository and cd'ing into the 
textory directory:

.. code-block:: bash

    pip install .

