import os
import argparse
import pandas as pd
from multiprocessing import Pool
from rdkit.Chem import Descriptors3D
from .read_input import read_input


def _rdkit_3d(mol_tup):
    mol, name, act, mol_id = mol_tup

    tmp = pd.DataFrame()

    desc_list = ['CalcAsphericity', 'CalcEccentricity', 'CalcInertialShapeFactor', 'CalcNPR1', 'CalcNPR2', 'CalcPMI1',
                 'CalcPMI2', 'CalcPMI3', 'CalcRadiusOfGyration', 'CalcSpherocityIndex', 'CalcPBF']

    for d in desc_list:
        try:
            tmp.loc[name, d] = getattr(Descriptors3D.rdMolDescriptors, d)(mol)
        except RuntimeError:
            print('Error 3D. Mol {0}. Descript {1}'.format(name, d))

    vector_descr = ['CalcAUTOCORR3D', 'CalcRDF', 'CalcMORSE', 'CalcWHIM', 'CalcGETAWAY']

    for v_d in vector_descr:
        for n, num in enumerate(getattr(Descriptors3D.rdMolDescriptors, v_d)(mol)):
            tmp.loc[name, '{descr}_{k}'.format(descr=v_d, k=n)] = num

    tmp = tmp[tmp <= 10 ** 35]

    tmp.loc[name, 'mol_id'] = mol_id
    tmp.loc[name, 'act'] = act

    return tmp


def main(fname, ncpu, path=None, del_log=True):
    if path is None:
        path = os.path.dirname(os.path.abspath(fname))

    mols = None

    p = Pool(ncpu)

    if fname.endswith('.sdf'):
        sdfmols = read_input(fname)
        mols = [[m, mol_title, m.GetProp('Act'), m.GetProp('Mol')] for m, mol_title in sdfmols if m is not None]
    if fname.endswith('.pkl'):
        mols = read_input(fname)

    if mols is None:
        print('Wrong format')
        return None

    d3_data = pd.DataFrame()

    for res in p.imap_unordered(_rdkit_3d, mols, chunksize=100):
        d3_data = d3_data.append(res)

    p.close()
    p.join()

    # descript columns contain Nan. 3D descriptors sometimes have error calculation for some mols
    if d3_data.isna().sum().sum() != 0:
        print('Warning. Nan 3D descr columns', fname)
        nan_col = list(d3_data.columns[d3_data.isna().sum().to_numpy().nonzero()])
        print('3D descr Nan columns = {}'.format(nan_col))
        # when predict col <act> = nan
        if nan_col != ['act']:
            try:
                nan_col.remove('act')
            except ValueError:
                pass

            if del_log:
                fname_log = os.path.join(path, '3d_del.log')
                if os.path.exists(fname_log):
                    with open(fname_log) as col_log:
                        fdata = col_log.read()
                        fdata = fdata.split('\t')
                else:
                    fdata = []

                col_del_add = [i for i in nan_col if i not in fdata]
                if col_del_add:
                    with open(fname_log, 'a') as log:
                        log.write('\t'.join([*col_del_add, '']))

            d3_data = d3_data.dropna(axis='columns')

    d3_data.index.name = 'mol_title'

    out_fname = os.path.join(path, '3DDescrRDKit_{f_name}.csv'.format(f_name=os.path.basename(fname).split('.')[0]))

    d3_data.to_csv(out_fname)

    return out_fname

    
def calc_rdkit_descriptors(*args, **kwargs):
    return main(*args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', metavar='input.sdf', required=True,
                        help='SDF [3D coordinates. Field: <Act>, <Mol>] or '
                             'pkl [(RDKit.mol, mol_title: str, act:int/float, mol_id:str)].'
                             'Input for calculation of the 3D RDKit descriptors')
    parser.add_argument('-p', '--path', metavar='path', default=None, help='out path')
    parser.add_argument('-n', '--nc', metavar='num', required=False, default=2, type=int,
                        help='Num of cores for calculation')

    args = parser.parse_args()
    _in_fname = args.input
    _path = args.path
    _nc = args.nc

    main(fname=_in_fname, ncpu=_nc, path=_path)
