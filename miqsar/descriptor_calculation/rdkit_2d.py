import os
import argparse
import pandas as pd
from multiprocessing import Pool
from rdkit.Chem import Descriptors
from .read_input import read_input
from .read_input import calc_max_tau

def _rdkit_2d(mol_id_input):
    mol, name, act, _ = mol_id_input
    tmp = pd.DataFrame()
    for k, f in Descriptors._descList:
        tmp.loc[name, k] = f(mol)
    tmp = tmp[tmp <= 10 ** 35]
    tmp.loc[name, 'mol_id'] = name
    tmp.loc[name, 'act'] = act
    return tmp


def main(fname, ncpu, tautomers_smi=False, path=None):
    if path is None:
        path = os.path.dirname(os.path.abspath(fname))
    p = Pool(ncpu)
    mols = read_input(fname)
    if tautomers_smi:
        max_tau = calc_max_tau(fname)
    else:
        max_tau = 0
    pdres = pd.DataFrame()
    for res in p.imap_unordered(_rdkit_2d, mols, chunksize=100):
        pdres = pdres.append(res)
    p.close()
    p.join()

    pdres.index.name = 'mol_title'
    # clean
    if pdres.isna().sum().sum() != 0:
        print('Warning. Nan 2D descr columns', fname)
        nan_col = list(pdres.columns[pdres.isna().sum().to_numpy().nonzero()])
        print('2D descr Nan columns = {}'.format(nan_col))
        pdres = pdres.dropna( axis='columns')

    out_path = os.path.join(path, '2DDescrRDKit_{f_name}_{nconf}.csv'.format(f_name=os.path.basename(fname).split('.')[0],
                                                                       nconf=max_tau))
    pdres.to_csv(out_path)

    return out_path


def calc_2d_descriptors(*args, **kwargs):
    return main(*args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', metavar='input.smi', required=True,
                        help='smi for calculation of 2D RDKit descriptors. sep: tab. Columns: <smi>, <name>, [<act>].'
                             ' Without colname')
    parser.add_argument('-p', '--path', metavar='path', default=None,
                        help='out path')
    parser.add_argument('-n', '--nc', metavar='num', required=False, default=2, type=int,
                        help='Num of cores for calculation')

    args = parser.parse_args()

    _in_fname = args.input
    _path = args.path
    _nc = args.nc

    main(fname=_in_fname, ncpu=_nc, path=_path)
