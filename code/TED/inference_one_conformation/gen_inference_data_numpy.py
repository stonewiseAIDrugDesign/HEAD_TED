import numpy as np
import functools
import multiprocessing
from multiprocessing import Pool
import logging
import os
from tqdm import tqdm

logger = logging.getLogger('Log')


def gen_feature(temp_en_list):
    ret_feature_list = []
    energy_list = []
    origin_ids = []
    sort_dihedrals = [int(x[0].split('__')[1]) for x in temp_en_list]
    sort_numpy = np.array(sort_dihedrals)
    sort_ids = np.argsort(sort_numpy)
    for sort_id in sort_ids:
        temp_en = temp_en_list[sort_id]
        mol_id = str(temp_en[0]).split('__')[0]
        origin_ids.append(temp_en[0])
        feature_list = [float(x) for x in str(temp_en[2]).split(',')]
        ret_feature_list.append(np.array(feature_list))
        energy = float(temp_en[1])
        energy_list.append(energy)
    energy_np = np.array(energy_list)
    energy_np = energy_np - np.min(energy_np)
    energy_np_list = energy_np.tolist()
    return mol_id, ret_feature_list, energy_np_list, origin_ids

def gen_feature_batch(temp_en_list_list):
    res = []
    for temp_en_list in temp_en_list_list:
        res.append(gen_feature(temp_en_list))
    return res

def choose_mol_id(all_id_dict):
    ret_list = []
    mol_id_list = []
    for temp_key in all_id_dict:
        if len(all_id_dict[temp_key]) == 24:
            ret_list.append(all_id_dict[temp_key])
            mol_id_list.append(temp_key)
    return ret_list


def main(feature_output_list_of_dict: list, input_cols: list, task_id: str, sch_gen_path):
    if not os.path.exists(sch_gen_path):
        os.mkdir(sch_gen_path)
    only_one_set = set()
    all_id_dict = {}
    for temp_feature_dict in feature_output_list_of_dict:
        uniq_key = temp_feature_dict[input_cols[0]]
        uniq_key_split = uniq_key.split('__')
        mod_id = uniq_key_split[0]
        angle_int = abs(int(uniq_key_split[1]))
        if angle_int == 180:
            uniq_key = mod_id + '__' + str(angle_int)
        if uniq_key not in only_one_set:
            only_one_set.add(uniq_key)
            if mod_id not in all_id_dict:
                all_id_dict[mod_id] = []
            all_id_dict[mod_id].append(
                (temp_feature_dict[input_cols[0]], temp_feature_dict[input_cols[1]], temp_feature_dict[input_cols[2]]))
    logger.info(f'all molecule size is :{len(all_id_dict)}')
    all_list = choose_mol_id(all_id_dict)

    cpu_num = None
    all_list_chunk = [all_list[i:i+100] for i in range(0, len(all_list), 100)]
    with Pool(cpu_num) as workers:
        _result = list(workers.imap(gen_feature_batch, all_list_chunk))
    result = []
    for item in _result:
        result.extend(item)

    test_x_list = []
    test_y_list = []
    test_y_energy_list = []
    for temp_en in result:
        test_x_list.extend(temp_en[1])
        test_y_list.extend(temp_en[3])
        test_y_energy_list.extend(temp_en[2])
    test_energy_y = np.array(test_y_energy_list)
    test_x = np.array(test_x_list)
    test_id_y = np.array(test_y_list)
    logger.info(f'test_id is:{test_y_list[0:24]},{test_y_list[-24:0]}')
    test_energy_y = test_energy_y.reshape(test_energy_y.shape[0], 1)
    test_id_y = test_id_y.reshape(test_id_y.shape[0], 1)
    test_y = np.concatenate((test_energy_y, test_id_y), axis=1)
    logger.info(f'test_y:{test_y.shape}')
    logger.info(f'test_x:{test_x.shape}')
    np.save(f'{sch_gen_path}/{task_id}_y_test.npy', test_y)
    np.save(f'{sch_gen_path}/{task_id}_X_test.npy', test_x)