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
from miqsar.descriptor_calculation.read_input import __read_pkl, __read_sdf

__smarts_patterns = load_smarts()


def __read_pkl(fname):
    with open(fname, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break
                


def read_input(fname,  input_format=None, id_field_name=None, sanitize=True):
    """
    fname - is a file name, None if STDIN
    input_format - is a format of input data, cannot be None for STDIN
    id_field_name - name of the field containing molecule name, if None molecule title will be taken
    Yields
    ------
     List[rdkit.mol, str]
     molecule(conformer) & its mol_id 
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
        suppl = __read_sdf(os.path.abspath(fname), input_format, id_field_name, sanitize)
    elif input_format in ('smi'):
        suppl = __read_smiles(os.path.abspath(fname), sanitize)
    elif input_format == 'pkl':
        suppl = __read_pkl(os.path.abspath(fname))
    else:
        raise Exception("Input file format '%s' is not supported. It can be only sdf, sdf.gz, smi, pkl." % input_format)
    for i in suppl:
        yield i[0:2]   # yield List[mol, mol_id] from arbitrary length iterable


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

    def save_mol_descriptors(self, mol_name, mol_descr_dict, cols=None):
        """
        Parameters
        ----------
        mol_name:str
        mol_descr_dict:dict
        cols:Sequence
        List /tuple with names of columns to save; the rest will be dismissed. This can be useful when we create descriptors for test set,
        and we already have train-set descriptors .svm file. Passing them will save memory, because
        we wont  ever need any other descriptors but train set ones. Note: the order of columns saved will be arbitrary, not same as in cols.

        """

        if cols is not None:
            mol_descr_dict = {i: v for i, v in mol_descr_dict.items() if i in cols}

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
    Creates descriptors for molecules

    Returns
    -------
    mol_title:str
    res:dict
    Keys: signatures sep by "|"; values - counts ;  size of dict may vary

    """
    ph = Pharmacophore(bin_step=1, cached=False)
    ph.load_from_smarts(mol, smarts=smarts_features)

    res = dict()
    for n in descr_num:
        res.update(ph.get_descriptors(ncomb=n))
    return mol_title, res

def process_mol_atom_excl(mol, mol_title, descr_num, smarts_features):
    # descr_num - list of int
    """
    Creates descriptors for atom-depleted molecules for interpretation of models.

    Parameters
    ----------
    mol - rdkit mol
    mol_title:str
    descr_num - list of int
    How many points should pharmacophore contain: e.g. triplets, quadruplets. Recommended is [4]

    Returns
    -------
    mol_title:str
    res_atom:Dict[dict]
    outer keys: 0-based atom ids; inner keys: signatures sep by "|"; inner values - counts

    Note
    ----
    Atoms whose descr dict is same as molecules' desc dict are filtered out (for interpretation it's useless, contrib will be 0)

    """
    ph = Pharmacophore(bin_step=1, cached=True) # cached must be to save and not re-compute
    ph.load_from_smarts(mol, smarts=smarts_features)

    res_mol = dict() # reference whole molecule
    res_atom = defaultdict(dict)

    for n in descr_num:
        res_mol.update(ph.get_descriptors(ncomb=n))
    for n in descr_num:
        for i, res in enumerate(ph.get_descriptors_atom_exclusion(natoms=mol.GetNumHeavyAtoms(),ncomb=n)):
                res_atom[i].update(res)

    res_atom = {k: v for k, v in res_atom.items() if v != res_mol}

    return mol_title, res_atom

def add_prefix(mol_name, nested_dict):
    """
    utility to unfold nested dict with atom-depleted molecules' descriptors.
    Yields one atom-depleted molecule's descriptors dict at a time.
    Param
    -----
    mol_name:str
    nested_dict: collections.defaultdict[dict]
    outer keys: atom_ids; inner keys: signatures; inner values: counts
    yields
    ------
    str
    "mol_name###atomid"
    dict
    keys: signatures sep by "|"; values - counts
    """
    # print(nested_dict)
    for key, subdict in nested_dict.items():
        yield (mol_name + "###" + str(key)), subdict


def process_mol_map(items, descr_num, smarts_features):
    return process_mol(*items, descr_num=descr_num, smarts_features=smarts_features)

def process_mol_atom_excl_map(items, descr_num, smarts_features):
    return process_mol_atom_excl(*items, descr_num=descr_num, smarts_features=smarts_features)


def main(inp_fname=None, out_fname=None, atom_exclusion=False, smarts_features=None,
         descr_num=[4], remove=0.05, colnames=None, keep_temp=False, ncpu=1, verbose=False):
    """
    WARNING:order of saving conformers/atom-depleted versions will be  same as order of input;
    Only  ordered input (i.e. all conformers of same mol go one by one) gives correct filtering by using parameter remove
    (if input was not ordered, output can be (and must be for correct bag formation)  ordered afterwards, e.g. BASH:
    paste file.rownames file.txt | sort | cut -f2 > file_sorted.txt - to sort rows and X and write new X.
    Also dont forget write new rownames).

    Parameters
    ----------
    #  todo
    """
    if colnames is not None:
        cols = []
        with open(colnames, "rt") as tmp:
            for line in tmp:
                cols.append(line.strip())
    else:
        cols = None

    if remove < 0 or remove > 1:
        raise ValueError('Value of the "remove" argument is out of range [0, 1]')

    for v in descr_num:
        if v < 1 or v > 4:
            raise ValueError('The number of features in a single descriptor should be within 1-4 range.')

    if atom_exclusion:
        out_fname = (os.path.splitext(out_fname)[0] + "_atoms" + os.path.splitext(out_fname)[1])

    ncores = max(min(ncpu, cpu_count()), 1)
    pool = Pool(ncores)
    chunksize = 1000

    tmp_fname = os.path.splitext(out_fname)[0] + '.' + ''.join(random.sample(string.ascii_lowercase, 6)) + '.txt'
    svm = SvmSaver(tmp_fname)

    # init some vars concerned with further filtering of decriptors:
    tmp_titles = set()  # mol titles to further compute len()
    ids_per_mol_ttl = set()  # with this set we ll update counter each time new  mol appears
    c = Counter()  # all descriptors counts for all mols
    length = 1  # init len of tmp_titles as 1 for the first pass of for loop. Later it will be updated each pass

    if not atom_exclusion:

        # create temp file with all descriptors
        for i, (mol_title, desc) in enumerate(
                pool.imap(partial(process_mol_map, descr_num=descr_num, smarts_features=smarts_features),
                          read_input(inp_fname), chunksize=chunksize), 1):
            if desc:
                tmp_titles.update(mol_title)
                ids = svm.save_mol_descriptors(mol_title, desc, cols)  # ids= signatures
                if len(tmp_titles) > length:  # new mol_title appeared
                    c.update(ids_per_mol_ttl)  # update counter with set for previous mol, as new one appeared
                    ids_per_mol_ttl = set()
                else:
                    ids_per_mol_ttl.update(ids)  # continue adding signatures of same mol_ttl
                length = len(tmp_titles)

            if verbose and i % 10 == 0:
                sys.stderr.write(f'\r{i} molecule records were processed')
        sys.stderr.write('\n')

    else:  # atom exclusion
        # create temp file with all descriptors
        for i, (mol_title, desc) in enumerate(
                pool.imap(partial(process_mol_atom_excl_map, descr_num=descr_num, smarts_features=smarts_features),
                          read_input(inp_fname), chunksize=chunksize), 1):
            tmp_titles.update(mol_title)
            # print(desc)
            for (mol_at_name, subd) in add_prefix(mol_name=mol_title, nested_dict=desc):
                if subd:  # todo: do we need this check?
                    ids = svm.save_mol_descriptors(mol_at_name, subd, cols)  # ids= signatures
                    if len(tmp_titles) == length:
                        # continue adding signatures of same mol_ttl;
                        # we have to check it each loop pass, thou it can happen only in first
                        ids_per_mol_ttl.update(ids)

                if verbose and i % 10 == 0:
                    sys.stderr.write(f'\r{i} molecule records were processed')

            if len(tmp_titles) > length:  # new ttl appeared
                c.update(ids_per_mol_ttl)
                ids_per_mol_ttl = set()
            # no need in else, this case will be covered in inner loop (if len(..)==lngth)
            length = len(tmp_titles)
            sys.stderr.write("c=" + str(sys.getsizeof(c)))
            sys.stderr.write("tmp_titles=" + str(sys.getsizeof(tmp_titles)))
            sys.stderr.write("ids_per_mol_ttl=" + str(sys.getsizeof(ids_per_mol_ttl)))
            sys.stderr.flush()
            sys.stderr.write('\n')

    if remove == 0:  # if no remove - rename temp files to output files
        os.rename(tmp_fname, out_fname)
        os.rename(os.path.splitext(tmp_fname)[0] + '.colnames', os.path.splitext(out_fname)[0] + '.colnames')
        os.rename(os.path.splitext(tmp_fname)[0] + '.rownames', os.path.splitext(out_fname)[0] + '.rownames')

    else:
        # determine frequency of descriptors occurrence and select frequently occurring
        threshold = len(tmp_titles) * remove
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-i', '--input', metavar='input.pkl', required=True,
                        help='(pkl or sdf) Input  molecules (conformers)')
    parser.add_argument('-o', '--output', metavar='output.txt', required=True,
                        help='output file name')
    parser.add_argument('-a',  '--atom_exclusion',  action='store_true',default=False,
                        help='')
    parser.add_argument('-d',  '--descr_num',   nargs='+', required=False,default=4,
                        help='')
    parser.add_argument('-r',  '--remove',   required=False,default=0,
                        help='Remove rare descr. Note: result will depend on mol_title of input file'
                             ' (all confs have same name or different)')
    parser.add_argument( '--colnames',   required=False,default=None,
                        help='colnames file to use as filter. Dont use this arg with --remove>0, they serve same aim.'
                             'Use this (supply train clonames) when working with test-set if train had --remove>0'
                             'to guarantee same set of colnames as train')
    parser.add_argument('-k',  '--keep_temp',   action='store_true', default=False,
                        help='')
    parser.add_argument('-c',  '--ncpu',   required=False, default=1,
                        help='')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='')

    args = parser.parse_args()
    _inp_fname = args.input
    _out_fname = args.output
    _atom_exclusion = args.atom_exclusion
    _descr_num = [args.descr_num]
    _remove = float(args.remove)
    _colnames = args.colnames
    _keep_temp = args.keep_temp
    _ncpu = int(args.ncpu)
    _verbose = args.verbose

    main(inp_fname=_inp_fname,
         out_fname=_out_fname,
         smarts_features = __smarts_patterns,
         atom_exclusion=_atom_exclusion,
         descr_num=_descr_num,
         remove=_remove,
         colnames=_colnames,
         keep_temp=_keep_temp,
         ncpu=_ncpu,
         verbose=_verbose)