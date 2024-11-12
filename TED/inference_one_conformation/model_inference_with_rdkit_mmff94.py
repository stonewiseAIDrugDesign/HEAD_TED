import logging
import traceback
import os
import time
import gen_inference_data_numpy
import evaluate_with_rdkit_attention
import math

# from common import logger
logger = logging.getLogger('Log')
sch_gen_path = '/home/train_test_data/rdkit_base_dihedral'

molAtomMapNumberStr = 'molAtomMapNumber'


def convert_dihedral_to_25(res_predict_np):
    ret_energy_dict = {}
    for temp_i in range(0, res_predict_np.shape[0]):
        energy_value = float(res_predict_np[temp_i, 0])
        is_nan = math.isnan(energy_value)
        if is_nan:
            continue
        dih_value_key = res_predict_np[temp_i, 1]

        temp_base_split = dih_value_key.split('__')
        mol_id = temp_base_split[0]
        if mol_id not in ret_energy_dict:
            ret_energy_dict[mol_id] = []
        dihedral_value = int(temp_base_split[1])
        if abs(dihedral_value) == 180:
            ret_energy_dict[mol_id].append((mol_id + '__180', energy_value))
            ret_energy_dict[mol_id].append((mol_id + '__-180', energy_value))
        else:
            ret_energy_dict[mol_id].append((dih_value_key, energy_value))
    return ret_energy_dict


def udf_run(input_list_of_dict: list, output_cols: list, functionExtraParam: str):
    task_id = functionExtraParam.get('task_name', 'default_task_name')
    input_cols = ['unique_key', 'energy', 'feature_str', 'mol_id']
    start_time = time.time()
    if os.path.exists(sch_gen_path):
        os.system('rm -rf ' + sch_gen_path)
    os.makedirs(sch_gen_path)
    all_sdf_list = []

    logger.info(f"input_list sdf len:{len(input_list_of_dict)}")
    ###(feature_output_list_of_dict: list, input_cols: list, task_id: str):
    gen_inference_data_numpy.main(input_list_of_dict, input_cols, task_id)
    file_name = task_id
    res_predict_np = evaluate_with_rdkit_attention.main(file_name)
    ret_energy_dict = convert_dihedral_to_25(res_predict_np)
    ###mod_id,unique_key,energy
    output_list_of_dict = []
    for temp_key in ret_energy_dict:
        temp_en_list = ret_energy_dict[temp_key]
        if len(temp_en_list) < 25:
            logger.error(f'have no len in 25:{temp_en_list}')
            continue
        for temp_en in temp_en_list:
            mol_id = temp_key
            energy = temp_en[1]
            unique_key = temp_en[0]
            output_data_dict = {}
            output_data_dict[output_cols[0]] = mol_id
            output_data_dict[output_cols[1]] = unique_key
            output_data_dict[output_cols[2]] = energy
            output_list_of_dict.append(output_data_dict)
    logger.info(f'spend time model is:{time.time() - start_time}')
    return output_list_of_dict
