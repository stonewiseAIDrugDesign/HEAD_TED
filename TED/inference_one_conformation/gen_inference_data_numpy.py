import numpy as np
import functools
from multiprocessing import Pool
import logging
import os

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


def choose_mol_id(all_id_dict):
    ret_list = []
    mol_id_list = []
    for temp_key in all_id_dict:
        if len(all_id_dict[temp_key]) == 24:
            ret_list.append(all_id_dict[temp_key])
            mol_id_list.append(temp_key)
    # train_mol_id_list, test_mol_id_list = train_test_split(mol_id_list, test_size=0.2, random_state=12345)
    # train_mol_id_set = set(train_mol_id_list)
    # test_mol_id__set = set(test_mol_id_list)
    # logger.info(f'train mol size:{len(train_mol_id_set)}, test size:{len(test_mol_id__set)}')
    return ret_list


base_lacal_path = '/home/train_test_data/rdkit_base_dihedral'


### input_cols = ['unique_key', 'energy', 'feature_str', 'mol_id']
##unique_key,feature_str,energy
def main(feature_output_list_of_dict: list, input_cols: list, task_id: str):
    if not os.path.exists(base_lacal_path):
        os.mkdir(base_lacal_path)
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
    pll_gen = functools.partial(gen_feature, )
    pool = Pool(processes=20)
    result = pool.map(pll_gen, all_list)
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()

    ###如果小24,就将这个id 去除掉。
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
    np.save(f'{base_lacal_path}/{task_id}_y_test.npy', test_y)
    np.save(f'{base_lacal_path}/{task_id}_X_test.npy', test_x)


if __name__ == '__main__':
    # all_table_list = get_table_from_s3(1697534070)
    all_table_list = ['docking_align_qsar.input__csd_test__20231114',
                      'docking_align_qsar.input__torsionnet500_smi__20231010',
                      'docking_align_qsar.input__turing_test_one__20231107',
                      'docking_align_qsar.input__stonewise_dft_001__20231208']
    for temp_base_table in all_table_list:
        main(temp_base_table)
