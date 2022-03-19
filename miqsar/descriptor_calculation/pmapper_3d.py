import argparse
import os
import sys
import string
import random
import pickle
import itertools
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import defaultdict, Counter
from rdkit.Chem import AllChem

from pmapper.pharmacophore import Pharmacophore
from pmapper.customize import load_smarts


__smarts_patterns = load_smarts()


def __read_pkl(fname):
    with open(fname, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def read_input(fname, input_format=None, id_field_name=None, sanitize=True, sdf_confs=False):
    """
    fname - is a file name, None if STDIN
    input_format - is a format of input data, cannot be None for STDIN
    id_field_name - name of the field containing molecule name, if None molecule title will be taken
    sdf_confs - return consecutive molecules with the same name as a single Mol object with multiple conformers
    """
    if input_format is None:
        tmp = os.path.basename(fname).split('.')
        if tmp == 'gz':
            input_format = '.'.join(tmp[-2:])
        else:
            input_format = tmp[-1]
    input_format = input_format.lower()
    if fname is None:  # handle STDIN
        if input_format == 'sdf':
            suppl = __read_stdin_sdf(sanitize=sanitize)
        elif input_format == 'smi':
            suppl = __read_stdin_smiles(sanitize=sanitize)
        else:
            raise Exception("Input STDIN format '%s' is not supported. It can be only sdf, smi." % input_format)
    elif input_format in ("sdf", "sdf.gz"):
        suppl = __read_sdf_confs(os.path.abspath(fname), input_format, id_field_name, sanitize, sdf_confs)
    elif input_format in ('smi'):
        suppl = __read_smiles(os.path.abspath(fname), sanitize)
    elif input_format == 'pkl':
        suppl = __read_pkl(os.path.abspath(fname))
    else:
        raise Exception("Input file format '%s' is not supported. It can be only sdf, sdf.gz, smi, pkl." % input_format)
    for mol, mol_id, act, mol_name in suppl:
        yield mol, mol_id


class SvmSaver:

    def __init__(self, file_name):
        self.__fname = file_name
        self.__varnames_fname = os.path.splitext(file_name)[0] + '.colnames'
        self.__molnames_fname = os.path.splitext(file_name)[0] + '.rownames'
        self.__varnames = dict()
        if os.path.isfile(self.__fname):
            os.remove(self.__fname)
        if os.path.isfile(self.__molnames_fname):
            os.remove(self.__molnames_fname)
        if os.path.isfile(self.__varnames_fname):
            os.remove(self.__varnames_fname)

    def save_mol_descriptors(self, mol_name, mol_descr_dict):

        new_varnames = list(mol_descr_dict.keys() - self.__varnames.keys())
        for v in new_varnames:
            self.__varnames[v] = len(self.__varnames)

        values = {self.__varnames[k]: v for k, v in mol_descr_dict.items()}

        if values:  # values can be empty if all descriptors are zero

            with open(self.__molnames_fname, 'at') as f:
                f.write(mol_name + '\n')

            if new_varnames:
                with open(self.__varnames_fname, 'at') as f:
                    f.write('\n'.join(new_varnames) + '\n')

            with open(self.__fname, 'at') as f:
                values = sorted(values.items())
                values_str = ('%i:%i' % (i, v) for i, v in values)
                f.write(' '.join(values_str) + '\n')

            return tuple(i for i, v in values)

        return tuple()


def process_mol(mol, mol_title, descr_num, smarts_features):
    # descr_num - list of int
    """
    Returns
    -------
    mol_title:str
    res:dict
    Keys: signatures sep by "|"; values - counts ;  size of dict may vary

    """
    ph = Pharmacophore(bin_step=1, cached=False)
    ph.load_from_smarts(mol, smarts_features=smarts_features)

    res = dict()
    for n in descr_num:
        res.update(ph.get_descriptors(ncomb=n))
    return mol_title, res


def process_mol_map(items, descr_num, smarts_features):
    return process_mol(*items, descr_num=descr_num, smarts_features=smarts_features)


def main(inp_fname=None, out_fname=None, smarts_features=None, factory=None,
         descr_num=[4], remove=0.05, keep_temp=False, ncpu=1, verbose=False):
    if remove < 0 or remove > 1:
        raise ValueError('Value of the "remove" argument is out of range [0, 1]')

    for v in descr_num:
        if v < 1 or v > 4:
            raise ValueError('The number of features in a single descriptor should be within 1-4 range.')

    pool = Pool(max(min(ncpu, cpu_count()), 1))

    tmp_fname = os.path.splitext(out_fname)[0] + '.' + ''.join(random.sample(string.ascii_lowercase, 6)) + '.txt'
    svm = SvmSaver(tmp_fname)

    stat = defaultdict(set)

    # create temp file with all descriptors
    for i, (mol_title, desc) in enumerate(
            pool.imap(partial(process_mol_map, descr_num=descr_num, smarts_features=smarts_features),
                      read_input(inp_fname), chunksize=1), 1):

        # print(mol_title)

        if desc:
            ids = svm.save_mol_descriptors(mol_title, desc) # ids= signatures
            stat[mol_title].update(ids)
        if verbose and i % 10 == 0:
            sys.stderr.write(f'\r{i} molecule records were processed')
    sys.stderr.write('\n')

    if remove == 0:  # if no remove - rename temp files to output files
        os.rename(tmp_fname, out_fname)
        os.rename(os.path.splitext(tmp_fname)[0] + '.colnames', os.path.splitext(out_fname)[0] + '.colnames')
        os.rename(os.path.splitext(tmp_fname)[0] + '.rownames', os.path.splitext(out_fname)[0] + '.rownames')

    else:
        # determine frequency of descriptors occurrence and select frequently occurring
        c = Counter(itertools.chain(stat.values()))
        threshold = len(stat) * remove
        print(len(stat))
        print(threshold)

        desc_ids = {k for k, v in c.items() if v >= threshold}

        # create output files with removed descriptors

        replace_dict = dict()  # old_id, new_id
        with open(os.path.splitext(out_fname)[0] + '.colnames', 'wt') as fout:
            with open(os.path.splitext(tmp_fname)[0] + '.colnames') as fin:
                for i, line in enumerate(fin):
                    if i in desc_ids:
                        replace_dict[i] = len(replace_dict)
                        fout.write(line)

        with open(os.path.splitext(out_fname)[0] + '.rownames', 'wt') as fmol, open(out_fname, 'wt') as ftxt:
            with open(os.path.splitext(tmp_fname)[0] + '.rownames') as fmol_tmp, open(tmp_fname) as ftxt_tmp:
                for line1, line2 in zip(fmol_tmp, ftxt_tmp):
                    desc_str = []
                    for item in line2.strip().split(' '):
                        i, v = item.split(':')
                        i = int(i)
                        if i in replace_dict:
                            desc_str.append(f'{replace_dict[i]}:{v}')
                    if desc_str:
                        fmol.write(line1)
                        ftxt.write(' '.join(desc_str) + '\n')

        if not keep_temp:
            os.remove(tmp_fname)
            os.remove(os.path.splitext(tmp_fname)[0] + '.colnames')
            os.remove(os.path.splitext(tmp_fname)[0] + '.rownames')


def calc_pmapper_descriptors(*args, **kwargs):
    return main(*args, **kwargs)