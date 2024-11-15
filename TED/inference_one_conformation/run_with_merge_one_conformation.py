import logging
import pandas as pd
import time
import argparse

# from common import logger
logger = logging.getLogger('Log')
import uuid

import min_energy_fro_torsion
import scan_xtb_gen_feature
import flexible_scan_with_mmff94
import model_inference_with_rdkit_mmff94

### input_list=input_cols: conformer, dihedral
### output_cols=[mol_id,unique_key,energy]
conformer_key = 'conformer'
dihedral_key = 'dihedral'


def udf_run(input_list_of_dict: list, output_cols: list, functionExtraParam: dict):
    start_time = time.time()
    scan_with_mmff94_output = flexible_scan_with_mmff94.udf_run(input_list_of_dict)
    map_torsion_output = None
    min_energy_output = min_energy_fro_torsion.udf_run(scan_with_mmff94_output)
    scan_with_mmff94_output = None
    feature_output_list_of_dict = scan_xtb_gen_feature.udf_run(min_energy_output)
    min_energy_output = None
    output_list_of_dict = model_inference_with_rdkit_mmff94.udf_run(feature_output_list_of_dict, output_cols,
                                                                    functionExtraParam)
    feature_output_list_of_dict = None
    logger.info(f'all flow time spend time is:{time.time() - start_time}')
    return output_list_of_dict


"""
input path is .csv,the row is conformation,dihedral
model path is /home/train_test_data/model_first_layer_attention.h5
dihedral format:5-8-9-10
conformation name is unique,if have multiply conformations name like test-base-name_1,test_base-name_2.
'_'is used for label the same molecular different conformation. so the base-name should not include '_'
"""


def read_data_to_input_list(data_path: str):
    pd_data = pd.read_csv(data_path)
    input_dict_list = []
    for temp_i in range(0, pd_data.shape[0]):
        temp_dict = {}
        temp_dict[conformer_key] = pd_data.iloc[temp_i, 0]
        temp_dict[dihedral_key] = pd_data.iloc[temp_i, 1]
        input_dict_list.append(temp_dict)
    return input_dict_list


"""
the output is csv format. the column names is: mol_id,unique_key,energy
mol_id is conformation id, the unique_key is consisted of 'mol_id','__' and 'dihedral_value',format is:
'mol_id__dihedral_value'
"""


def output_data_to_csv(output_path: str, output_list_of_dict):
    pd_data = pd.DataFrame(output_list_of_dict)
    pd_data.to_csv(output_path, index=False)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="input data path, it is csv format,one row is conformation sdf,dihedral atom id",
    )
    parser.add_argument(
        "--out-path",
        help="output data path,it is csv format,one row is mod_id,unique_key,energy",
    )
    parser.add_argument(
        "--model-path",
        help="model file path, main model should be TensorFlow's h5 format while the scaaler model should be a .pkl"
    )
    return parser


def run_the_service(args: argparse.Namespace) -> None:
    input_dict_list = read_data_to_input_list(args.data_path)
    unique_id = str(uuid.uuid4()).replace('-', '_')
    print(f'unique id:{unique_id}')
    task_id = f'default_{unique_id}'
    functionExtraParam = {}
    functionExtraParam['task_name'] = task_id
    functionExtraParam['model_path'] = args.model_path
    output_cols = ['mol_id', 'unique_key', 'energy']
    output_dict_list = udf_run(input_dict_list, output_cols, functionExtraParam)
    output_data_to_csv(args.out_path, output_dict_list)


if __name__ == '__main__':
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args()
    run_the_service(args)
