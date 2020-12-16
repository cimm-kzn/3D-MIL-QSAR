import os
import argparse
import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import AllChem
from .read_input import read_input
from .read_input import calc_max_tau

def rdkit_numpy_convert(fp):
    output = []
    for f in fp:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        output.append(arr)
    return np.asarray(output)


def main(fname, tautomers_smi=False, path=None):
    if path is None:
        path = os.path.dirname(fname)
    if tautomers_smi:
        max_tau = calc_max_tau(fname)
    else:
        max_tau = 0
    mol_id_gener = read_input(fname)
    mol_id_list = [m for m in mol_id_gener]

    fp = [AllChem.GetMorganFingerprintAsBitVect(m[0], 2) for m in mol_id_list]
    x = rdkit_numpy_convert(fp)
    name = [m[1] for m in mol_id_list]
    act = [m[2] for m in mol_id_list]

    pdres = pd.DataFrame(x)
    pdres.loc[:, 'act'] = act
    pdres.loc[:, 'mol_id'] = name
    pdres.loc[:, 'mol_title'] = name

    out_path = os.path.join(path,
                            'MorganFprRDKit_{f_name}_{nconf}.csv'.format(f_name=os.path.basename(fname).split('.')[0],
                                                                         nconf=max_tau))

    pdres.to_csv(out_path, index=False)

    return out_path


def calc_morgan_descriptors(*args, **kwargs):
    return main(*args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', metavar='input.smi', required=True,
                        help='smi for calculation of the 2D Morgan Fingerprints RDKit. sep: tab. Columns: <smi>, <name>, [<act>]. '
                             'Without colname')
    parser.add_argument('-p', '--path', metavar='path', default=None, help='out path')

    args = parser.parse_args()
    _in_fname = args.input
    _path = args.path

    main(fname=_in_fname, path=_path)
