import logging
import traceback
from rdkit import Chem
import os
import io
import functools
from multiprocessing import Pool
import multiprocessing
from . import remould_symmetry_function as feature_gen
import time
from tqdm import tqdm

# from common import logger
logger = logging.getLogger('Log')


def get_feature_parallel(input_en):
    new_block_str = input_en[1]
    contents_bytes = bytes(new_block_str, 'utf-8')
    mol_list = Chem.ForwardSDMolSupplier(io.BytesIO(contents_bytes), removeHs=False)

    for temp_mol in mol_list:
        mol_name = str(temp_mol.GetProp('_Name'))
        temp_split = mol_name.split('_')
        mol_id = temp_split[0]
        energy_value = float(temp_mol.GetProp('Energy'))
        ret_feature_list = feature_gen.get_sf_elements(temp_mol)
        if ret_feature_list is None or len(ret_feature_list) == 0:
            return None
    ret_str = ','.join([str(x) for x in ret_feature_list])
    ret_tuple = (input_en[0], temp_split[0], energy_value, ret_str)
    return ret_tuple

def get_feature_parallel_batch(input_en_list):
    res = []
    for input_en in input_en_list:
        res.append(get_feature_parallel(input_en))
    return res

def udf_run(input_list_of_dict: list):
    input_cols = ['sdf_id', 'conformer_sdf']
    output_cols = ['unique_key', 'energy', 'feature_str', 'mol_id']
    start_time = time.time()
    all_sdf_list = []
    idx = -1
    for data_dict in input_list_of_dict:
        try:
            all_sdf_list.append((data_dict[input_cols[0]], data_dict[input_cols[1]]))
            idx = idx + 1
        except Exception:
            data_dict = input_list_of_dict[idx]
            logger.error(f"process error line_no:{idx}")
            logger.error(traceback.format_exc())
            continue
    logger.info(f"input_list sdf len:{len(all_sdf_list)}")

    output_list_of_dict = []
    cpu_num = None
    all_sdf_list_chunk = [all_sdf_list[i:i+100] for i in range(0, len(all_sdf_list), 100)]
    with Pool(cpu_num) as workers:
        _result = list(workers.imap(get_feature_parallel_batch, all_sdf_list_chunk))
    result = []
    for item in _result:
        result.extend(item)

    lowest_energy_dict = {}
    for temp_en in result:
        if temp_en is None:
            continue
        temp_key = temp_en[0]
        lowest_energy_dict[temp_key] = (temp_en[1], temp_en[2], temp_en[3])
    for temp_key in lowest_energy_dict:
        output_data_dict = {}
        output_data_dict[output_cols[0]] = temp_key
        output_data_dict[output_cols[1]] = lowest_energy_dict[temp_key][1]
        output_data_dict[output_cols[2]] = lowest_energy_dict[temp_key][2]
        output_data_dict[output_cols[3]] = lowest_energy_dict[temp_key][0]
        output_list_of_dict.append(output_data_dict)
    logger.info(f'spend time feature generate is:{time.time() - start_time}')
    return output_list_of_dict
