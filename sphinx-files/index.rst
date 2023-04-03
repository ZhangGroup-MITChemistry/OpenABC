.. OpenABC documentation master file, created by
   sphinx-quickstart on Sun Apr  2 22:16:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OpenABC's documentation!
===================================

OpenABC stands for **Open**\ MM GPU-\ **A**\ ccelerated simulations of **B**\ iomolecular **C**\ ondensates. It is flexible and implements multiple popular coarse-grained force fields for simulations, including the hydropathy scale (HPS) model, MOFF :math:`C_{\alpha}`  model, and the molecular renormalization group (MRG)-CG DNA model. The package dramatically simplifies the simulation setup: users only need a few lines of python code to carry out condensate simulations starting from initial configurations of a single protein or DNA. The package is integrated with OpenMM, a GPU-accelerated MD simulation engine, enabling efficient simulations with advanced sampling techniques. We include tools for converting coarse-grained configurations to atomistic structures for further simulations with all-atom force fields. We provide tutorials in Jupyter notebooks to demonstrate the various capabilities. We anticipate OpenABC to significantly facilitate the application of existing computer models for simulating biomolecular condensates and the continued development of new force fields.


Environment
===========

We recommend using openmm 7.5.1 for using OpenABC, as OpenABC is built based on openmm 7.5.1.

Install openmm 7.5.1 with the following command: ``conda install -c conda-forge openmm=7.5.1``

Other required packages: numpy, pandas, mdanalysis, mdtraj.

If running replica exchange with ``openabc.utils.replica_exchange``, then torch is also required.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   manual


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Citations
=========

We will add reference after the paper is formally online.

