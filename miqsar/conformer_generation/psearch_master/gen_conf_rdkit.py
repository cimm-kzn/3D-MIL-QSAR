#!/usr/bin/env python3
# author          : Pavel Polishchuk
# date            : 13.07.16
# license         : BSD-3
# ==============================================================================

__author__ = 'Pavel Polishchuk'

import os, time
import sys
import gzip
import argparse
import pickle
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool, cpu_count
from openbabel import pybel

# from .read_input import read_input
if __name__ == '__main__':
    from read_input import read_input
else:
    from .read_input import read_input

ob = pybel.ob

def gen_confs_obabel(rdkit_mol, nconf=50):
    mol = pybel.readstring('mol', Chem.MolToMolBlock(rdkit_mol)).OBMol  # convert mol from RDKit to OB
    mol.AddHydrogens()

    pff = ob.OBForceField_FindType('mmff94')
    if not pff.Setup(mol):  # if OB FF setup fails use RDKit conformer generation (slower)
        print('err')

    pff.DiverseConfGen(0.5, 1000, 50, False)  # rmsd, nconf_tries, energy, verbose
    pff.GetConformers(mol)

    obconversion = ob.OBConversion()
    obconversion.SetOutFormat('mol')

    output_strings = []
    for conf_num in range(max(0, mol.NumConformers() - nconf),
                          mol.NumConformers()):
        mol.SetConformer(conf_num)
        output_strings.append(obconversion.WriteString(mol))

    pff.ConjugateGradients(100, 1.0e-3)

    rdkit_confs = []
    for i in output_strings:
        mol = Chem.MolFromMolBlock(i, removeHs=False)
        #AllChem.MMFFOptimizeMolecule(mol, confId=0)
        #AllChem.UFFOptimizeMolecule(mol, confId=0)
        rdkit_confs.append(mol)

    return rdkit_confs


def prep_input(fname, id_field_name, nconf, energy, rms, seed):
    input_format = 'smi' if fname is None else None
    for mol, mol_name, act, mol_id in read_input(fname, input_format=input_format, id_field_name=id_field_name):
        yield mol, mol_name, nconf, energy, rms, seed, act, mol_id


def map_gen_conf(args):
    return gen_confs(*args)


def sorted_confids(mol):
    sorted_list = []

    for conf in mol.GetConformers():
        #ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf.GetId())
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())
        if ff is None:
            print(Chem.MolToSmiles(mol))
        else:
            sorted_list.append((conf.GetId(), ff.CalcEnergy()))

    if sorted_list:
        sorted_list = sorted(sorted_list, key=lambda x: x[1])

    return sorted_list


def remove_confs(mol, energy, rms):
    e = []
    for conf in mol.GetConformers():
        #ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf.GetId())
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())

        if ff is None:
            print(Chem.MolToSmiles(mol))
            return
        e.append((conf.GetId(), ff.CalcEnergy()))
    e = sorted(e, key=lambda x: x[1])

    if not e:
        return

    kept_ids = [e[0][0]]
    remove_ids = []

    for item in e[1:]:
        if item[1] - e[0][1] <= energy:
            kept_ids.append(item[0])
        else:
            remove_ids.append(item[0])

    if rms is not None:
        rms_list = [(i1, i2, AllChem.GetConformerRMS(mol, i1, i2)) for i1, i2 in combinations(kept_ids, 2)]
        while any(item[2] < rms for item in rms_list):
            for item in rms_list:
                if item[2] < rms:
                    i = item[1]
                    remove_ids.append(i)
                    break
            rms_list = [item for item in rms_list if item[0] != i and item[1] != i]

    for cid in set(remove_ids):
        mol.RemoveConformer(cid)


def gen_confs(mol, mol_name, nconf, energy, rms, seed, act, mol_id):
    mol = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=nconf, maxAttempts=700, randomSeed=seed)

    if len(cids) == 0:
        confs = gen_confs_obabel(mol, nconf=nconf)
        for conf in confs:
            mol.AddConformer(conf.GetConformer())
        cids = list(range(0, len(confs)))

    for cid in cids:
        try:
            #AllChem.MMFFOptimizeMolecule(mol, confId=cid)
            AllChem.UFFOptimizeMolecule(mol, confId=cid)
        except:
            continue
    remove_confs(mol, energy, rms)
    return mol_name, mol, act, mol_id


def main_params(in_fname, out_fname, id_field_name, nconf, energy, rms, ncpu, seed, verbose, log=False):
    start_time = time.time()

    output_file_type = None
    if out_fname is not None:

        if os.path.isfile(out_fname):
            os.remove(out_fname)

        if out_fname.lower().endswith('.sdf.gz'):
            writer = gzip.open(out_fname, 'a')
            output_file_type = 'sdf.gz'
        elif out_fname.lower().endswith('.sdf'):
            # writer = open(out_fname, 'at')
            writer = Chem.SDWriter(out_fname)
            output_file_type = 'sdf'
        elif out_fname.lower().endswith('.pkl'):
            writer = open(out_fname, 'wb')
            output_file_type = 'pkl'
        else:
            raise Exception("Wrong output file format. Can be only SDF, SDF.GZ or PKL.")

    nprocess = min(cpu_count(), max(ncpu, 1))
    p = Pool(nprocess)

    try:
        for i, (mol_name, mol, act, mol_id) in enumerate(
                p.imap_unordered(map_gen_conf, prep_input(in_fname, id_field_name, nconf, energy, rms, seed),
                                 chunksize=10), 1):

            # get sorted by energy list of conformerIDs [(confIds, energy)]
            ids_sorted = sorted_confids(mol)

            if not ids_sorted:
                pass
                print(Chem.MolToSmiles(mol), mol_name)

            if output_file_type == 'pkl':
                mol_conf_list = []

                for confId, energ in ids_sorted:
                    name = '{name}_{confId}'.format(name=mol_name, confId=confId)
                    mol_conf = Chem.Mol(mol, False, confId)
                    if not log:
                        pickle.dump((mol_conf, name, act, mol_id), writer, -1)
                    else:
                        mol_conf_list.append((mol_conf, name, act, mol_id, energ))

                if log and mol_conf_list:
                    pickle.dump(mol_conf_list, writer, -1)

            else:
                mol.SetProp("_Name", mol_name)
                mol.SetProp("Act", str(act))
                mol.SetProp("Mol", mol_id)

                if output_file_type == 'sdf':
                    for confId, energ in ids_sorted:
                        name = '{name}_{confId}'.format(name=mol_name, confId=confId)
                        mol.SetProp("_Name", name)
                        writer.write(mol, confId=confId)
                else:
                    string = "$$$$\n".join(Chem.MolToMolBlock(mol, confId=c.GetId()) for c in mol.GetConformers())
                    if string:  # wrong molecules (no valid conformers) will result in empty string
                        string += "$$$$\n"
                        if out_fname is None:
                            sys.stdout.write(string)
                            #sys.stdout.flush()
                        else:
                            writer.write(string.encode("ascii") if output_file_type == 'sdf.gz' else string)

            if verbose and i % 10 == 0:
                sys.stderr.write('\r%i molecules passed/conformers (%is)' % (i, time.time() - start_time))
                sys.stderr.flush()

    finally:
        p.close()
        p.join()

    if out_fname is not None:
        writer.close()

    if verbose:
        sys.stderr.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate specified number of conformers using RDKit.')
    parser.add_argument('-i', '--in', metavar='input.sdf', required=True,
                        help='input file with structures to generate conformers. Allowed formats SDF or SMILES. '
                             'if omitted STDIN will be used. STDIN takes only SMILES input (one or two columns).')
    parser.add_argument('-o', '--out', metavar='output.sdf', required=True,
                        help='output SDF file where conformers are stored. If extension will be SDF.GZ the output file '
                             'will be automatically gzipped. Alternatively for faster storage output can be stored '
                             'in a file with extension PKL. That is pickled storage of tuples (mol, mol_name). '
                             'If the output option will be omitted the output will be done to STDOUT in SDF format.')
    parser.add_argument('-d', '--id_field_name', metavar='field_name', default=None,
                        help='field name of compound ID in input SDF file. If omitted for sdf molecule titles '
                             'will be used or SMILES strings as names.')
    parser.add_argument('-n', '--nconf', metavar='conf_number', default=50,
                        help='number of generated conformers. Default: 50.')
    parser.add_argument('-e', '--energy_cutoff', metavar='10', default=10,
                        help='conformers with energy difference from the lowest found one higher than the specified '
                             'value will be discarded. Default: 10.')
    parser.add_argument('-r', '--rms', metavar='rms_threshold', default=.5,
                        help='only conformers with RMS higher then threshold will be kept. '
                             'Default: None (keep all conformers).')
    parser.add_argument('-s', '--seed', metavar='random_seed', default=-1,
                        help='integer to init random number generator. Default: -1 (means no seed).')
    parser.add_argument('-c', '--ncpu', metavar='cpu_number', default=1,
                        help='number of cpu to use for calculation. Default: 1.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='print progress to STDERR.')

    args = vars(parser.parse_args())
    for o, v in args.items():
        if o == "in": in_fname = v
        if o == "out": out_fname = v
        if o == "id_field_name": id_field_name = v
        if o == "nconf": nconf = int(v)
        if o == "ncpu": ncpu = int(v)
        if o == "energy_cutoff": energy = float(v)
        if o == "seed": seed = int(v)
        if o == "rms": rms = float(v) if v is not None else None
        if o == "verbose": verbose = v

    main_params(in_fname=in_fname,
                out_fname=out_fname,
                id_field_name=id_field_name,
                nconf=nconf,
                energy=energy,
                rms=rms,
                ncpu=ncpu,
                seed=seed,
                verbose=verbose)
