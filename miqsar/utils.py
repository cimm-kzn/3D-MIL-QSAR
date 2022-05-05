import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import groupby
from pmapper.customize import load_smarts, load_factory
from .conformer_generation.gen_conformers import gen_confs
from .descriptor_calculation.pmapper_3d import calc_pmapper_descriptors
from .descriptor_calculation.rdkit_morgan import calc_morgan_descriptors


def read_pkl(fname):
    with open(fname, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def calc_3d_pmapper(input_fname=None, nconfs_list=[1, 50], energy=10, descr_num=[4], ncpu=5, path='.'):
    
    conf_files = gen_confs(input_fname, ncpu=ncpu, nconfs_list=nconfs_list, stereo=False, energy=energy, path=path)

    smarts_features = load_smarts('./miqsar/smarts_features.txt')
    factory = load_factory('./miqsar/smarts_features.fdef')
    
    for conf_file in conf_files:
    
        out_fname = os.path.join(path, 'PhFprPmapper_{}.txt'.format(os.path.basename(conf_file).split('.')[0]))

        calc_pmapper_descriptors(inp_fname=conf_file, out_fname=out_fname, 
                                 smarts_features=smarts_features, factory=factory,
                                 descr_num=descr_num, remove=0.05, keep_temp=False, ncpu=ncpu, verbose=False)
        
        #
        data = pd.read_csv(input_fname, header=None, index_col=1)
        rownames = pd.read_csv(out_fname.replace('.txt', '.rownames'), header=None)
        idx = [i.split('_')[0] for i in rownames[0]]
        labels = [ f'{i}:{data.loc[i, 2]}\n' for i in idx]

        with open(out_fname.replace('.txt', '.rownames'), 'w') as f:
            f.write(''.join(labels))

    return out_fname