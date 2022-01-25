import os
import argparse
import pkg_resources
import pandas as pd
from multiprocessing import Pool
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from .read_input import read_input
from .read_input import calc_max_tau

fdef_fname = pkg_resources.resource_filename(__name__, 'pmapper/smarts_features.fdef')
featFactory = ChemicalFeatures.BuildFeatureFactory(fdef_fname)
sigFactory = SigFactory(featFactory, minPointCount=2, maxPointCount=3, trianglePruneBins=False)
sigFactory.SetBins([(0, 2), (2, 5), (5, 8)])
sigFactory.Init()


def _ph_rdkit(mols_tup):
    mol, name, act, _ = mols_tup
    ph = Generate.Gen2DFingerprint(mol, sigFactory)
    tmp = pd.DataFrame(columns=range(ph.GetNumBits()))
    ph_bits = list(ph.GetOnBits())
    for n_bit in ph_bits:
        tmp.loc[name, n_bit] = 1
    tmp.loc[name, 'mol_id'] = name
    tmp.loc[name, 'act'] = act
    tmp = tmp.fillna(0)
    return tmp


def main(fname, ncpu, tautomers_smi=False, path=None):

    p = Pool(ncpu)
    if path is None:
        path = os.path.dirname(os.path.abspath(fname))
    mols = read_input(fname)
    pdres = pd.DataFrame()
    if tautomers_smi:
        max_tau = calc_max_tau(fname)
    else:
        max_tau = 0
    for res in p.imap_unordered(_ph_rdkit, mols, chunksize=100):
        pdres = pdres.append(res)
    p.close()
    p.join()
    pdres.index.name = 'mol_title'
    #  mol_id, act columns
    lb = len(pdres.columns) - 2
    out_path = os.path.join(path,
                            'PhFprRDKit_{f_name}_{nconf}.csv'.format(f_name=os.path.basename(fname).split('.')[0],
                                                                     nconf=max_tau))
    pdres.to_csv(out_path)
    return out_path


def calc_ph_descriptors(*args, **kwargs):
    return main(*args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', metavar='input.smi', required=True,
                        help='smi for calculation of the 2D Pharmacophore Fingerprints RDKit. sep: tab. Columns: <smi>, <name>, [<act>].'
                             ' Without colname')
    parser.add_argument('-p', '--path', metavar='path', default=None, help='out path')
    parser.add_argument('-n', '--nc', metavar='num', required=False, default=2, type=int,
                        help='Num of cores for calculation')

    args = parser.parse_args()
    _in_fname = args.input
    _path = args.path
    _nc = args.nc

    main(fname=_in_fname, ncpu=_nc, path=_path)
