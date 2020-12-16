import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pickle
import numpy as np
from .psearch_master import gen_stereo_rdkit, gen_conf_rdkit, read_input


def get_part_conf(all_confs_list, nconf, out_partfname):
    out_fname = out_partfname.format(nconf)
    writer = open(out_fname, 'wb')

    # confs for mols
    for conf_list in all_confs_list:
        n_part = np.linspace(0, len(conf_list) - 1, nconf, dtype=int)
        for n in set(n_part):
            # del log energy
            conf = conf_list[n][:-1]
            pickle.dump(conf, writer, -1)

    return out_fname


def get_n_confs(conf_log, nconf_list, out_partfname):
    # confs for all mols
    all_confs_list = list(read_input.read_input(conf_log))
    if not all_confs_list:
        #print('Error. Conformer log file is empty', conf_log)
        return None

    out_fnames = []

    for nconf in nconf_list:
        out_fnames.append(get_part_conf(all_confs_list=all_confs_list, nconf=nconf, out_partfname=out_partfname))

    return out_fnames


def gen_confs(fname, nconfs_list, stereo=True, path=None, ncpu=4):
    '''

    :param fname: smi file. Mol_name, smiles, act
    :param nconfs_list: list[int]
    :param stereo: bool. False if only compute 3D coordinates.
    :param path: out path. If None uses dirname of fname
    :param ncpu: int
    :return:
    '''

    if path is None:
        path = os.path.dirname(fname)
    if not os.path.exists(path):
        os.makedirs(path)

    if stereo:
        #print('Stereo generation')
        in_fname = os.path.join(path, 'stereo-{}'.format(os.path.basename(fname)))
        gen_stereo_rdkit.main_params(in_fname=fname,
                                     out_fname=in_fname,
                                     tetrahedral=True,
                                     double_bond=True,
                                     max_undef=-1,
                                     id_field_name=None,
                                     ncpu=ncpu,
                                     verbose=False)

    else:
        in_fname = fname

    #print('Conformers generation')

    max_conf = max(nconfs_list)

    conf_log = os.path.join(path, 'conf-{0}_{1}_log.pkl'.format(max_conf, os.path.basename(in_fname).split('.')[0]))
    # conf_tupl[-1]=energy
    gen_conf_rdkit.main_params(in_fname=in_fname,
                               out_fname=conf_log,
                               id_field_name=None,
                               nconf=max_conf,
                               energy=100,
                               rms=.5,
                               ncpu=ncpu,
                               seed=42,
                               verbose=False,
                               log=True)

    out_partfname = os.path.join(path, 'conf-{0}_{1}.pkl'.format(os.path.basename(in_fname).split('.')[0], '{}'))
    # take n-confromer from conformer-log file
    out_fnames = get_n_confs(conf_log=conf_log, nconf_list=nconfs_list, out_partfname=out_partfname)

    return out_fnames

def get_from_exist_log(conf_log, nconfs_list):
    ex_path = os.path.dirname(conf_log)
    out_partfname = os.path.join(ex_path, 'conf-{0}_{1}.pkl'.format(os.path.basename(conf_log).split('.')[0], '{}'))

    out_fnames = get_n_confs(conf_log=conf_log, nconf_list=nconfs_list, out_partfname=out_partfname)
    #print(out_fnames)

    return out_fnames

if __name__ == '__main__':
    parser = ArgumentParser(description='', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', metavar='file.smi', required=False, default=None,
                        help='input smi file. Sep: tab. Columns: <smiles>, <mol_id>, [<act>]. Without column names.')
    parser.add_argument('-p', '--path', metavar='out dir', required=False, default=None,
                        help='Output dir')
    parser.add_argument('--nconf', metavar='10 25 50', required=False, nargs='*', type=int,
                        default=[1, 10, 25], help='List num of conformers')
    parser.add_argument('--stereo', action='store_true', default=False,
                        help='Default - compute 3D coordinates only and use input stereo information')
    parser.add_argument('-nc', '--ncpu', metavar='cpu_number', default=1, type=int,
                        help='number of cpus to use for generating conformers')
    parser.add_argument('--ex_log', metavar='use exist Logfile', default=None,
                        help='Generate conformers from existed conformer file')

    args = parser.parse_args()

    in_fname_ = args.input
    path_ = args.path
    ncpu_ = args.ncpu
    stereo_ = args.stereo
    nconf_list_ = args.nconf
    ex_log_ = args.ex_log

    if ex_log_ is None:
        gen_confs(fname=in_fname_,
              nconfs_list=nconf_list_,
              stereo=stereo_,
              path=path_,
              ncpu=ncpu_)
    else:
        out_fnames = get_from_exist_log(conf_log=ex_log_, nconfs_list=nconf_list_)