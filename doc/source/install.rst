=========================
Installation Instructions
=========================

Pip-based Installation
======================

Textory is currently only available via github but can be installed
with pip:

.. code-block:: bash

    $ pip install git+https://github.com/BENR0/textory.git@master#egg=textory

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
        - "git+https://github.com/BENR0/textory.git@master#egg=textory
