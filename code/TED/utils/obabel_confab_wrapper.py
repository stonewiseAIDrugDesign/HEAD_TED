import os
import functools
import multiprocessing as mp
import subprocess as sp
from .temp_control import TempDir


def obabel_wrapper(input_sdf_list, xcutoff, ecutoff, conf_num, workdir='.'):
    temp_manager = TempDir()
    tmp_dir = temp_manager.__enter__(workdir)
    tmp_input_sdf_path = os.path.join(tmp_dir, 'input.sdf')
    tmp_output_sdf_path = os.path.join(tmp_dir, 'output.sdf')
    with open(tmp_input_sdf_path, 'w') as f:
        for input_sdf in input_sdf_list:
            f.write(input_sdf + '\n')
    try:
        sp.run(f'obabel {tmp_input_sdf_path} -O {tmp_output_sdf_path} --confab --xcutoff {xcutoff} --ecutoff {ecutoff} --conf {conf_num}', shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)
        output_sdf = open(tmp_output_sdf_path, 'r').read()
    except:
        output_sdf = None
    temp_manager.__exit__()
    return output_sdf

def obabel_mp(input_sdf_list_chunk, xcutoff, ecutoff, conf_num, workdir, num_proc):
    obj = functools.partial(obabel_wrapper, xcutoff=xcutoff, ecutoff=ecutoff, conf_num=conf_num, workdir=workdir)
    with mp.Pool(num_proc) as workers:
        res = list(workers.imap(obj, input_sdf_list_chunk))
    return res
    

