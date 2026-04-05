Installation
============

Recommended: conda environment
-------------------------------

The easiest way to get a fully working environment is to clone the repo and create
the bundled conda environment:

.. code-block:: bash

    git clone https://github.com/ruffgroup/bauer
    cd bauer
    conda env create -f environment.yml   # creates the 'bauer' env
    conda activate bauer

This installs all runtime and documentation dependencies automatically.

pip install (PyPI)
------------------

Once released on PyPI you can install with:

.. code-block:: bash

    pip install bauer

And then add the optional extras you need:

.. code-block:: bash

    pip install "bauer[docs]"    # sphinx / nbsphinx for documentation
    pip install "bauer[dev]"     # flake8 / pytest for development

Install from source (development)
----------------------------------

.. code-block:: bash

    git clone https://github.com/ruffgroup/bauer
    cd bauer
    pip install -e ".[docs,dev]"

Runtime dependencies
--------------------

The core package requires:

* ``pymc >= 5``
* ``pytensor``
* ``pandas``
* ``numpy``
* ``patsy``
* ``arviz``
* ``scipy``
* ``seaborn``
* ``matplotlib``

Building the documentation additionally requires ``sphinx``, ``nbsphinx``,
``nbconvert``, and ``nbformat``.
