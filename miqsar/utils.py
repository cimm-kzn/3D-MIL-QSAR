import os
import pickle
import numpy as np
import pandas as pd
from collections import  Counter
from pmapper.customize import load_smarts, load_factory
from .conformer_generation.gen_conformers import gen_confs
from .descriptor_calculation.pmapper_3d import calc_pmapper_descriptors
import joblib
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler

mol_frag_sep = "###"

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

class SirmsFile():
    """
    Version of class from spci. The difference from original is: this class reads and prepares data for miqsar, i.e.
    it creates 3 d array of bags.

    Parameters
    ----------
    fname:str
    Path to .svm file. xxx.rownmes and xxx.colnames must be in the same dir too. xxx.rownames must be in the following form:
    MolID  or MolID###atom_id. Conformer id MUST NOT be used in these names.(That is, all conformers of same mol must have same name)
    WARNING: input must be sorted by molecule title, or at least order should be such
    that ALL conformers of the same molecule (or all conformers of all its atom-depleted versions) reside in adjacent lines.
    Iff these all conditions hold,  correct bag formation is ensured.
    """

    def __init__(self, fname, frag_file=False, chunks=float("Inf")):

        self.__varnames = [v.strip() for v in open(os.path.splitext(fname)[0] + '.colnames').readlines()]
        self.__mol_full_names = [v.strip() for v in open(os.path.splitext(fname)[0] + '.rownames').readlines()]
        self.__frag_full_names = [v for v in self.__mol_full_names if v.find(mol_frag_sep) > -1]
        self.__is_frag_full_names_read = True

        self.__file = open(fname)

        self.__nlines = chunks               # file is read by chunks to avoid memory problems,
                                             # to read the whole file at once set to float('Inf')
        self.__is_frag_file = frag_file      # set which file to read ordinary (False) or with fragments (True)
                                             # in the latter case each chunk will contain all fragments for each

        self.__cur_mol_read = 0              # number of mols (lines) already read (or index of line to read)


    def reset_read(self):
        self.__cur_mol_read = 0
        self.__file.seek(0)

    def __read_svm_next(self, train_names_file=None):
        """
        Firstly: determine start as  self.__cur_mol_read; end as self.__nlines.
        If self.__mol_full_names is already less than end, reset end to  this;
        else: expand end to ensure covearge of all conformers of last molecule.
        Secondly: read file from start to end (right-open interval: range()).

        Parameters (self attributes current values)
        --------------------------------------------
        self.__cur_mol_read: previous  end or zero
        self.__nlines: chunk size
        self.__mol_full_names: all rownames

        Modifies
        --------
        If input __cur_mol_read < len(self.__mol_full_names):
        self.__cur_mol_read is MODIFIED - set to end. Else: nothing

        Returns
        -------
        Tuple[list,np.array]

        If input __cur_mol_read == len(self.__mol_full_names): return is empty.Else: populated with:
        chunk mol_ids (incl. atom_id) and x of shape (Nmols_chunk;max(Nconf);Ndescr). NOTE: single molecule in this context is
        single-atom-depleted version of molecule (if so supplied), or  single normal molecule (if so supplied).
        Bag is either all confs of single-atom-depleted version of molecule (if so supplied)
        or all confs of  single normal molecule (if so supplied). NOTE: order of columns will be given by new_names if supplied
        to guarantee that test inputs will have same output order of columns as train.

        """

        if self.__cur_mol_read == len(self.__mol_full_names):
            return [], np.asarray([])
        else:
            start = self.__cur_mol_read
            end = start + self.__nlines
            if end > len(self.__mol_full_names):  # number of mols left to read is less than all mols - read them all.
                end = len(self.__mol_full_names)
            else:  # number of mols left to read is less/= than all mols - read whole chunk + any remaining conformers of last mol in chunk.
                cur_mol = self.__mol_full_names[end - 1].split(mol_frag_sep)[0]  # assign cur_mol = last in chunck
                # increase end until NEW mol id appears
                while end < len(self.__mol_full_names) and self.__mol_full_names[end].split(mol_frag_sep)[0] == cur_mol:
                    end += 1
            lines = [self.__file.readline().strip() for _ in range(start, end)]
            x = np.zeros((len(lines),len(self.__varnames)))
            for n,line in enumerate(lines):
                for entry in line.split():
                    index, value = entry.split(':')
                    x[n][int(index)] = value
                # print("nlines",n)
            # prepare 3d array of bags

            x = pd.DataFrame(x, index=self.__mol_full_names[start:end], columns=self.__varnames)
            # x.index = [i.upper() for i in x.index]

            if train_names_file is not None:
                new_varnames = [v.strip() for v in open(train_names_file).readlines()]
                print(set(x.columns) == set(new_varnames))# TRUE
                x = x.reindex(new_varnames, axis="columns", fill_value=0) # ensure  order of columns of test is same astrain
                print(x.columns == new_varnames)

            x = x.sort_index() # sort is memory pain!

            bags = []
            idx = []
            bag_size = max(Counter(x.index).values()) # molecule with max number of confs

            for i in x.index.unique():
                bag = x.loc[i:i].values # 2 d array (Nconfs,Ndescr)
                bag = np.pad(bag, pad_width=((0,bag_size-bag.shape[0]),(0,0))) #default constant_value=0;see docs pad for explain of notation
                bags.append(bag)
                idx.append(i)
                # print(bag.shape)
            bags = np.array(bags)
            # print(bags.shape)
            # print(x.index)
            # set next start
            self.__cur_mol_read = end

            return idx,  bags


    def read_next(self, train_names_file=None):
        return self.__read_svm_next(train_names_file)

def scale_descriptors(X_train, X_test, save_fname=None):
    scaler = MinMaxScaler()
    scaler.fit(np.vstack(X_train))

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    for i, bag in enumerate(X_train):
        X_train_scaled[i] = scaler.transform(bag)
    for i, bag in enumerate(X_test):
         X_test_scaled[i] = scaler.transform(bag)

    if save_fname is not None:
        joblib.dump(scaler, save_fname)

    return X_train_scaled, X_test_scaled

def read_sdf_labels(fname, act_field):
    """
    Parameters
    ----------
    fname:str
    Sdf path
    act_field:str
    filed name where activity is contained

    Returns
    -------
    pd.Series, index - mol name from sdf; values - activity labels
    """
    suppl = Chem.SDMolSupplier(fname)
    lbl={}
    for mol in suppl:
        if mol is not None:
            lbl[mol.GetProp("_Name")]=float(mol.GetProp(act_field))
    return pd.Series(lbl)
