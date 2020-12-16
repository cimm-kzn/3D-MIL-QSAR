import pickle
import joblib
import numpy as np
import pandas as pd
from miqsar.conformer_generation.gen_conformers import gen_confs
from miqsar.descriptor_calculation.rdkit_3d import calc_3d_descriptors
from miqsar.descriptor_calculation.pmapper_3d import calc_pmapper_descriptors

from sklearn.preprocessing import MinMaxScaler


def read_pkl(fname):
    with open(fname, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

                
def calc_3d_pmapper(dataset_file, nconfs=1, stereo=False, path='.', ncpu=10):
    
    conf_files = gen_confs(dataset_file, nconfs_list=[nconfs], stereo=stereo, path=path, ncpu=ncpu)

    for conf in conf_files:
        dsc_file = calc_pmapper_descriptors(conf, path=path, ncpu=ncpu, col_clean=None, del_undef=True)

        with open(dsc_file, 'rb') as inp:
            data = joblib.load(inp)

        if 'mol_title' not in data.columns:
            data = data.reset_index()
        data['mol_id'] = data['mol_id'].str.lower()
        fname = dsc_file.split
        dsc_file = dsc_file.replace('_proc.pkl', '.csv')
        data.to_csv(dsc_file, index=False)
        
    bags, labels, idx = read_data(dsc_file)

    return bags, labels, idx


def read_data(fname):
    data = pd.read_csv(fname, index_col='mol_id')
    data.index = [i.upper() for i in data.index]
    data = data.sort_index()

    idx = []
    bags = []
    labels = []
    for i in data.index.unique():
        bag = data.loc[i:i].drop(['mol_title', 'act'], axis=1).values
        label = float(data.loc[i:i]['act'].unique()[0])

        bags.append(bag)
        labels.append(label)
        idx.append(i)

    bags = np.array(bags)
    labels = np.array(labels)

    return bags, labels, idx


def scale_descriptors(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(np.vstack(X_train))
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    for i, bag in enumerate(X_train):
        X_train_scaled[i] = scaler.transform(bag)
    for i, bag in enumerate(X_test):
        X_test_scaled[i] = scaler.transform(bag)
    return X_train_scaled, X_test_scaled