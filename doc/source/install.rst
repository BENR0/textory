=========================
Installation Instructions
=========================

Pip-based Installation
======================

Textory can be installed from PyPi with pip:

.. code-block:: bash

    $ pip install textory 

Conda-based Installation
========================

A pip based install from github can also be included in a conda environment.yaml file
for example:

.. code-block:: yaml

    name: your_env_name
    dependencies:
      - python=3.7
      - numpy
      - scipy
      - xarray
      - dask
      - distributed
      - pip:
        - textory

Latest version
==============

Currently releases on PyPi might be infrequent. If you want to install the latest version
from the github repository use:

.. code-block:: bash

    $ pip install git+https://github.com/BENR0/textory.git@master#egg=textory 
