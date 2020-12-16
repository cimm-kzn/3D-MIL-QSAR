import os
import argparse
import collections
import pkg_resources
import numpy as np
import pandas as pd
from time import time
from multiprocessing import Pool
#from sklearn.externals import joblib
import joblib
from .pmapper import pharmacophore as P
from .read_input import read_input

smarts = P.read_smarts_feature_file(pkg_resources.resource_filename(__name__, 'pmapper/smarts_features.txt'))


def get_phf(mol):
    p = P.Pharmacophore()
    try:
        p.load_from_smarts(mol, smarts)
        phf_descript = pd.Series(p.get_descriptors(), dtype=np.uint8)
    except IndexError:
        # with open(log,'a') as out_log:
        #     out_log.write(name)
        # print('phf error')
        return None

    # phf_descript['mol_id'] = mol_id
    # phf_descript['act'] = act
    # phf_descript['mol_title'] = name

    return phf_descript

def get_phf_for_mol(mol_tup):
    mol, mol_title, act, mol_id = mol_tup
    phf_descr = get_phf(mol)

    if phf_descr is None:
        #print('phf error', mol_title)
        return None

    return phf_descr, mol_title, act, mol_id



def get_mols(fname):
    mols = None

    if fname.endswith('.sdf'):
        sdfmol_ids = read_input(fname)
        mols = [[m, mol_title, m.GetProp('Act'), m.GetProp('Mol')] for m, mol_title in sdfmol_ids if m is not None]
    if fname.endswith('.pkl'):
        mols = read_input(fname)

    return mols


def get_phf_clean(fname, ncpu, part=0.05):
    p = Pool(ncpu, maxtasksperchild=50)

    mols = get_mols(fname=fname)
    if mols is None:
        print('Wrong format or file')
        return None

    phf_dict = collections.Counter()
    mol_count = 0

    for res in p.imap_unordered(get_phf_for_mol, mols, chunksize=10):
        if not res is None:
            mol_count += 1
            phf = res[0]
            for phf_ind in phf.index:
                phf_dict[phf_ind] += 1

    p.close()
    p.join()

    phf_clean_ind = (np.array(list(phf_dict.values())) >= 2) & (np.array(list(phf_dict.values())) >= mol_count*part)
    phf_clean = np.array(list(phf_dict.keys()))[phf_clean_ind]

    #print('Preprocess columns:', len(phf_clean))
    return phf_clean

def add_uniq_col(fname, ncpu, col_clean):
    p = Pool(ncpu, maxtasksperchild=50)
    mols = get_mols(fname=fname)

    if mols is None:
        print('Wrong format or file')
        return None

    cols_add = set()

    for res in p.imap_unordered(get_phf_for_mol, mols, chunksize=10):
        phf_descr = res[0]
        if all(phf_descr.reindex(col_clean, fill_value=0).astype(np.uint8).values == 0):
            cols_add.update(phf_descr.index)

    #print('Additional columns:', len(cols_add))

    result_cols = np.array(list(set(col_clean) | cols_add))
    # print(result_cols)

    return result_cols


def main(fname, ncpu, path=None, col_clean=None, del_undef=True):
# col_clean can use for test set
    p = Pool(ncpu, maxtasksperchild=50)
    start = time()

    if path is None:
        path = os.path.dirname(os.path.abspath(fname))

    mols = get_mols(fname=fname)
    if mols is None:
        print('Wrong format or file')
        return None

    out_fname = os.path.join(path, 'PhFprPmapper_{f_name}_proc.pkl'.format(f_name=os.path.basename(fname).split('.')[0]))
    phf_res = [[],[]]

    #select common phf_descr - quantity of mols dataset >= 5%
    if col_clean is None:
        col_clean = get_phf_clean(fname=fname, ncpu=ncpu, part=0.05)
        if len(col_clean) == 0:
            print('Descriptors selection error. Clean col = 0. Threshold was lowered. To get descriptors if quantity > 2 ')
            col_clean = get_phf_clean(fname=fname, ncpu=ncpu, part=0)
        # col_clean = add_uniq_col(fname, ncpu, col_clean)

    #print('clean', time()-start, 'sec')

    for res in p.imap_unordered(get_phf_for_mol, mols, chunksize=10):
        if not res is None:
            phf_descr, mol_title, act, mol_id = res
            phf_res[0].append([mol_title, act, mol_id])
            phf_res[1].append(phf_descr.reindex(col_clean, fill_value=0).astype(np.uint8).values)

    p.close()
    p.join()

    phf = pd.DataFrame(phf_res[1], columns=col_clean).merge(pd.DataFrame(phf_res[0], columns=['mol_title', 'act', 'mol_id']), left_index=True, right_index=True,
                                                            copy=False)
    # print(sys.getsizeof(phf))
    #print('phf columns', len(col_clean))

    if del_undef:
        mol_ish = phf.loc(axis='columns')['mol_id'].unique()
        mask_def_mol = phf.drop(axis='columns', labels=['mol_title', 'act', 'mol_id']).apply(axis='columns',
                                                                                            func=lambda x: False if all(x == 0) else True)

        mols_del = np.setdiff1d(mol_ish, phf.loc(axis='index')[mask_def_mol]['mol_id'].unique()).tolist()
        #print('Mols for del', mols_del)

        out_mol_clean_fname = os.path.join(path,
                                 'PhFprPmapper_{f_name}_proc_molclean.pkl'.format(f_name=os.path.basename(fname).split('.')[0]))
        log_mol_clean_fname = os.path.join(path,
                                           'PhFprPmapper_{f_name}_proc_moldel.log'.format(
                                               f_name=os.path.basename(fname).split('_')[0]))
        with open(out_mol_clean_fname, 'wb') as out:
            # joblib.dump(fp_data, out)
            joblib.dump(phf.loc(axis='index')[mask_def_mol], out, compress='zlib')

        if os.path.exists(log_mol_clean_fname):
            with open(log_mol_clean_fname) as mol_log:
                fdata = mol_log.read()
                fdata = fdata.split('\t')
        else:
            fdata=[]

        mol_del_add = [str(i) for i in mols_del if i not in fdata]
        if mol_del_add:
            with open(log_mol_clean_fname, 'a') as log:
                log.write('\t'.join([*mol_del_add, '']))

    with open(out_fname, 'wb') as out:
        # joblib.dump(fp_data, out)
        joblib.dump(phf, out, compress='zlib')

    #print(out_fname, time() - start, 'sec')

    return out_fname


def calc_pmapper_descriptors(*args, **kwargs):
    return main(*args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', metavar='input.sdf', required=True,
                        help='SDF [3D coordinates. Field: <Act>, <Mol>] or '
                             'pkl [(RDKit.mol, mol_title: str, act:int/float, mol_title:str)].'
                             'Input for calculation of the 3D Pharmacophore Fingerprints Pmapper')
    parser.add_argument('-nc', '--ncpu', metavar='num', required=False, default=2, type=int,
                        help='num of core for calculation')

    args = parser.parse_args()
    _in_fname = args.input
    _nc = args.ncpu

    main(fname=_in_fname, ncpu=_nc)
