import logging
import traceback
from rdkit import Chem
import math
import os
import io
import functools
from multiprocessing import Pool
import multiprocessing
import pandas as pd
from rdkit.Chem import AllChem
logger = logging.getLogger('Log')
import time

molAtomMapNumberStr = 'molAtomMapNumber'


def get_series_point(temp_mol, temp_point_list):
    new_atom_ids = []
    for temp_index in temp_point_list:
        temp_index_int = int(temp_index)
        temp_atom_attr = temp_mol.GetAtomWithIdx(temp_index_int)
        atom_ngbs = temp_atom_attr.GetNeighbors()
        ngb_count = 0
        nerg_list = []
        for atom_ngb in atom_ngbs:
            atom_id = atom_ngb.GetIdx()
            if str(atom_id) in temp_point_list:
                nerg_list.append(atom_id)
        if len(nerg_list) == 2 or len(nerg_list) == 0:
            continue
        if len(new_atom_ids) == 0:
            new_atom_ids.append(temp_index_int)
            new_atom_ids.append(nerg_list[0])
        else:
            new_atom_ids.append(nerg_list[0])
            new_atom_ids.append(temp_index_int)
    return new_atom_ids


def compute_torsion_int(temp_mol, torsion_points):
    dihedral_angle_deg = AllChem.GetDihedralDeg(temp_mol.GetConformer(), int(torsion_points[0]),
                                                int(torsion_points[1]),
                                                int(torsion_points[2]),
                                                int(torsion_points[3]))
    if dihedral_angle_deg is None or math.isnan(dihedral_angle_deg):
        return None
    dihedral_angle_int = int(dihedral_angle_deg)
    span = 15
    remainder = dihedral_angle_int % span
    if remainder == 0:
        dihedral_angle_gap = dihedral_angle_int
    elif remainder < 8:
        dihedral_angle_gap = dihedral_angle_int - remainder
    else:
        dihedral_angle_gap = dihedral_angle_int - remainder + span
    return dihedral_angle_gap


def MolToSDFBlock(mol):
    res = [Chem.MolToMolBlock(mol)]
    for pn in mol.GetPropNames():
        pv = mol.GetProp(pn)
        res.append('>  <%s>\n%s\n' % (pn, pv))
    res.append('$$$$\n')
    return '\n'.join(res)


def get_conf_sdf_id(one_entity):
    ret_list = []
    contents_bytes = bytes(one_entity[0], 'utf-8')
    mol_list = Chem.ForwardSDMolSupplier(io.BytesIO(contents_bytes), removeHs=False)
    opt_mol = str(one_entity[2])
    for mol in mol_list:
        mol_name = str(mol.GetProp('_Name'))
    dihedral_list = str(one_entity[1]).split('-')
    if mol is None or opt_mol == "" or pd.isna(opt_mol):
        mol.SetProp('Energy', "None")
        return None
    atom_positions = opt_mol.split('\n')
    conf = mol.GetConformer()
    conf_nums = int((len(atom_positions) - 1) / (mol.GetNumAtoms() + 2))
    # print(conf_nums)
    for n in range(conf_nums):
        for i in range(mol.GetNumAtoms()):
            position = atom_positions[n * (mol.GetNumAtoms() + 2) + i + 2].split()
            atom_coor = (round(float(position[1]), 4), round(float(position[2]), 4), round(float(position[3]), 4))
            conf.SetAtomPosition(i, atom_coor)
        string = [k for k in atom_positions[n * (mol.GetNumAtoms() + 2) + 1].split(" ") if k != ""]
        energy = string[1]
        mol.SetProp('Energy', energy)
        temp_dihedral_value = compute_torsion_int(mol, dihedral_list)
        if temp_dihedral_value is None:
            logger.error(f'get the dihedral angle:{mol_name},{dihedral_list}')
            continue
        dihedral_value = str(temp_dihedral_value)
        mol.SetProp('TORSION_ATOMS_FRAGMENT', one_entity[1])
        mol.SetProp('TORSION_VALUE', dihedral_value)
        temp_mol_id = mol_name.split('_')[0]
        new_key = f'{temp_mol_id}__{dihedral_value}'
        ret_list.append((new_key, float(energy), MolToSDFBlock(mol)))
    return ret_list


def is_neutral(mol):
    total_formal_charge = sum([a.GetFormalCharge() for a in mol.GetAtoms()])
    return total_formal_charge == 0.0


"""
### input_cols:(conformer,dihedral,output) 
### output_cols: (sdf_id,energy,conformer_sdf)
"""


def udf_run(input_list_of_dict: list):
    input_cols = ['conformer', 'dihedral', 'output']
    output_cols = ['sdf_id', 'energy', 'conformer_sdf']
    start_time = time.time()
    all_sdf_list = []
    idx = -1
    for data_dict in input_list_of_dict:
        try:
            all_sdf_list.append((data_dict[input_cols[0]], data_dict[input_cols[1]], data_dict[input_cols[2]]))
            idx = idx + 1
        except Exception:
            data_dict = input_list_of_dict[idx]
            logger.error(f"process error line_no:{idx}")
            logger.error(traceback.format_exc())
            continue
    logger.info(f"input_list sdf len:{len(all_sdf_list)}")

    output_list_of_dict = []
    cpu_num = multiprocessing.cpu_count()
    if cpu_num < 4:
        cpu_num = 1
    pll_gen = functools.partial(get_conf_sdf_id, )
    pool = Pool(processes=cpu_num)
    result = pool.map(pll_gen, all_sdf_list)
    pool.close()
    pool.join()
    lowest_energy_dict = {}
    for temp_en_list in result:
        if temp_en_list is None:
            continue
        ###todo (new_key, float(energy), MolToSDFBlock(mol)
        for temp_en in temp_en_list:
            temp_key = temp_en[0]
            if temp_en[0] not in lowest_energy_dict:
                lowest_energy_dict[temp_key] = (temp_en[1], temp_en[2])
            elif temp_en[1] < lowest_energy_dict[temp_key][0]:
                lowest_energy_dict[temp_key] = (temp_en[1], temp_en[2])
    for temp_key in lowest_energy_dict:
        sdf_str = lowest_energy_dict[temp_key][1]
        output_data_dict = {}
        output_data_dict[output_cols[0]] = temp_key
        output_data_dict[output_cols[1]] = lowest_energy_dict[temp_key][0]
        output_data_dict[output_cols[2]] = sdf_str
        output_list_of_dict.append(output_data_dict)
    logger.info(f'spend time min energy is:{time.time() - start_time}')
    return output_list_of_dict
