Conformer Multi-Instance Machine Learning
==========================================================
This repository contains the Python source code from the `paper <https://pubs.acs.org/doi/10.1021/acs.jcim.1c00692>`_.

Overview
------------
In Multi-Instance Learning, each training object is represented by several feature
vectors (bag) and a label. In our implementation, an example (i.e., a molecule) is presented
by a bag of instances (i.e., a set of conformers), and a label (a bioactivity value) is available
only for a bag (a molecule), but not for individual instances (conformations).

Installation
------------
.. code-block:: bash

    pip install qsarmil

Supplementary packages
------------
The modelling pipeline is based on two supplementary packages: 

- `QSARmil <https://github.com/KagakuAI/QSARmil>`_ – Molecular multi-instance machine learning
- `milearn <https://github.com/KagakuAI/milearn>`_ - Multi-instance machine learning in Python

Refer to these packages for more examples and application cases.

How To Use
------------
Original datasets can be found at `datasets`. The folder contains 200 datasets on ligand bioactivity extracted from ChEMBL.
Follow the `Notebook <https://github.com/cimm-kzn/3D-MIL-QSAR/blob/main/notebooks/Notebook_1_MIL_for_conformers.ipynb>`_ for usage example.
