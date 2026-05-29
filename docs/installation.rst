Installation
============

bauer has three install "profiles" depending on what you need. Pick one.

1. Choice-only models (CPU)
---------------------------

For the static (cumulative-normal) choice models — psychophysical, magnitude
comparison, risky choice — the base install is enough:


.. code-block:: bash

    git clone https://github.com/ruffgroup/bauer
    cd bauer
    pip install -e .          # (or: pip install bauer, once on PyPI)

2. DDM / RDM models + the fast sampler (CPU laptop)
---------------------------------------------------

The reaction-time models (drift-diffusion, race-diffusion) need the WFPT
likelihood from **HSSM**, and you'll want the **numpyro (JAX) backend**, which
is far faster than the default ``pymc`` backend (see
:ref:`backend-speed` below). The ``[ddm]`` extra pulls all of this:

.. code-block:: bash

    pip install -e ".[ddm]"   # hssm + jax (CPU) + numpyro + blackjax

3. GPU (workstation or cluster)
-------------------------------

A single GPU gives a large additional speedup (a hierarchical n=64 DDM is
~45 min on an L4 GPU vs many hours on CPU). **GPU use is decided entirely by
which JAX you install** — there is no code flag. Install the CUDA build of
JAX on top of the ``[ddm]`` extra:

.. code-block:: bash

    pip install -e ".[ddm]"
    pip install "jax[cuda12]"     # CUDA 12 build of JAX

Then just fit with ``backend='numpyro'`` (below) — bauer uses the GPU
automatically; you do **not** pass a device argument, and ``chain_method`` is
auto-set to ``'vectorized'`` so the chains run together on the one GPU.

.. _backend-speed:

Choosing a sampler backend (this is the speed knob)
---------------------------------------------------

``model.sample()`` takes a ``backend``:

* ``backend='pymc'`` (the **default**) — PyMC's NUTS. Reliable but **slow**.
* ``backend='numpyro'`` — JAX-backed NUTS. **Much faster** (~3–10× on CPU,
  ~5–30× on GPU), and it parallelises the chains automatically.

So for any non-trivial fit you want ``backend='numpyro'``; on a GPU machine it
then uses the GPU with no further changes. (``blackjax`` is an alternative
JAX backend; not generally needed.)

Conda environments
-------------------

Equivalent conda envs are bundled: ``environment.yml`` (CPU) and
``environment_cuda.yml`` (GPU, with ``jax[cuda12]``). Each ends by installing
bauer editable:

.. code-block:: bash

    conda env create -f environment.yml        # CPU, 'bauer' env
    # or, on a GPU box:
    conda env create -f environment_cuda.yml   # GPU, 'bauer_cuda' env
    conda activate bauer

Development / docs
------------------

.. code-block:: bash

    pip install -e ".[dev]"      # flake8 / pytest / pre-commit
    pre-commit install           # run flake8 on every commit (matches CI)
    pip install -e ".[docs]"     # sphinx / nbsphinx, to build these docs

Runtime dependencies
--------------------

Core (always): ``pymc >= 5``, ``pytensor``, ``pandas``, ``numpy``, ``patsy``,
``arviz < 1.0``, ``scipy``, ``seaborn``, ``matplotlib``. The ``[ddm]`` extra
adds ``hssm``, ``jax``, ``numpyro``, ``blackjax``.
