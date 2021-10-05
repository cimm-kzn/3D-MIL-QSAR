# QSAR modeling based on conformation ensembles using a multi-instance learning approach
This repository containes the Python source code from paper ["QSAR modeling based on conformation ensembles using a
multi-instance learning approach"](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00692)

## Overview
Our research focuses on the application of Multi-Instance Learning (MIL) in QSAR modeling.
In Multi-Instance Learning, each training object is represented by several feature
vectors (bag) and a label. In our implementation, an example (i.e., a molecule) is presented
by a bag of instances (i.e., a set of conformations), and a label (a bioactivity value) is available
only for a bag (a molecule), but not for individual instances (conformations).
Both traditional MI algorithms and MI deep neural networks were used for model building.

## Installation
This code requires the installation of the following packages:
+ joblib
+ numpy
+ pandas
+ scikit-learn
+ pytorch
+ torch-optimizer
+ rdkit
+ networkx

All packages can be installed using *conda*. Neural networks can be trained with CPU or GPU.

## How To Use
The `datasets` folder contains 175 datasets on ligand bioactivity extracted from ChEMBL.
These datasets were used to build and compare 2D and 3D models.

The `miqsar` contains scripts for conformer generation, calculation of 2D and 3D descriptors,
and implementation of Multi-Instance machine learning algorithms. This folder also includes the
file `utils.py`, which contains supporting scripts for demonstrtation of model building process in `example.ipynb`.

The `example.ipynb` is a jupyter notebook with some details and code to perform modeling.

## Citation
If you use this code, please cite our source paper:

```
@article{Zankov2021,
author = {Zankov, Dmitry V. and Matveieva, Mariia and Nikonenko, Aleksandra V. and Nugmanov, Ramil I. and Baskin, Igor I. and Varnek, Alexandre and Polishchuk, Pavel and Madzhidov, Timur I.},
doi = {10.1021/acs.jcim.1c00692},
issn = {1549-9596},
journal = {Journal of Chemical Information and Modeling},
month = {sep},
pages = {acs.jcim.1c00692},
title = {{QSAR Modeling Based on Conformation Ensembles Using a Multi-Instance Learning Approach}},
url = {https://pubs.acs.org/doi/10.1021/acs.jcim.1c00692},
year = {2021}
}
```
