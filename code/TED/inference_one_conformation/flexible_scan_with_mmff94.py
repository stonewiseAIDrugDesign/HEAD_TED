import logging
import traceback
from rdkit import Chem, RDLogger
import os
import io
import functools
from multiprocessing import Pool
import multiprocessing
import pandas as pd
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem import ChemicalForceFields
from rdkit.Chem import rdMolTransforms
import copy
import time
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
logger = logging.getLogger('Log')
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


def get_conf_sdf_id(temp_en):
    sdf_str = temp_en[0]
    dihedral_str = temp_en[1]
    dihedral_list = [int(x) for x in str(dihedral_str).split('-')]
    angles = range(-180, 182, 15)
    energy = []
    confid = 0
    contents_bytes = bytes(sdf_str, 'utf-8')
    mol_list = Chem.ForwardSDMolSupplier(io.BytesIO(contents_bytes), removeHs=False)
    count_num = 0
    for temp_mol in mol_list:
        count_num = count_num + 1

    write_list = []
    if temp_mol is None:
        logger.error('mol is null')
        return None
    for angle in angles:
        m2 = copy.deepcopy(temp_mol)
        mp2 = AllChem.MMFFGetMoleculeProperties(m2)
        confid += 1
        try:
            ff2 = AllChem.MMFFGetMoleculeForceField(m2, mp2)
        except Exception:
            logger.error('mmff init error')
            return None
        if ff2 is None:
            logger.error(f"the ff2 is null:{temp_mol.GetProp('_Name')}")
            return None
        try:
            ff2.MMFFAddTorsionConstraint(dihedral_list[0], dihedral_list[1], dihedral_list[2], dihedral_list[3], False,
                                        angle - 3.0, angle + 3.0, 1000.0)
            ff2.Minimize()
            ff2_energy = ff2.CalcEnergy()
        except Exception as e:
            logger.error(f"the ff2 minimization meets error in {m2.GetProp('_Name')}: {e}")
            return None
        energy.append(ff2_energy)
        write_list.append(temp_mol.GetProp('_Name'))
        write_list.append(f'energy: {ff2_energy}')
        for i in range(m2.GetNumAtoms()):
            xyz = m2.GetConformer(-1).GetAtomPosition(i)
            write_str = temp_mol.GetAtomWithIdx(i).GetSymbol() + ' ' + str(xyz.x) + ' ' + str(xyz.y) + ' ' + str(xyz.z)
            write_list.append(write_str)
    ret_str = '\n'.join(write_list) + '\n'
    return sdf_str, dihedral_str, ret_str


def get_conf_sdf_id_batch(temp_en_list):
    res = []
    for temp_en in temp_en_list:
        res.append(get_conf_sdf_id(temp_en))
    return res

def is_neutral(mol):
    total_formal_charge = sum([a.GetFormalCharge() for a in mol.GetAtoms()])
    return total_formal_charge == 0.0

def udf_run(input_list_of_dict: list):
    input_cols = ['conformer', 'dihedral']
    output_cols = ['conformer', 'dihedral', 'output']
    start_time = time.time()
    all_sdf_list = []
    idx = -1
    for data_dict in input_list_of_dict:
        try:
            all_sdf_list.append((data_dict[input_cols[0]], data_dict[input_cols[1]]))
            idx = idx + 1
        except Exception:
            data_dict = input_list_of_dict[idx]
            logger.error(f"process error line_no:{idx} content:{data_dict}")
            logger.error(traceback.format_exc())
            continue
    logger.info(f"input_list sdf len:{len(all_sdf_list)}")

    output_list_of_dict = []
    cpu_num = None
    all_sdf_list_chunk = [all_sdf_list[i:i+50] for i in range(0, len(all_sdf_list), 50)]
    with Pool(cpu_num) as workers:
        _result = list(workers.imap(get_conf_sdf_id_batch, all_sdf_list_chunk))
    result = []
    for item in _result:
        result.extend(item)

    for temp_en in result:
        if temp_en is None:
            continue
        sdf_str = temp_en[0]
        point_ids = temp_en[1]
        scan_xyz = temp_en[2]
        output_data_dict = {}
        output_data_dict[output_cols[0]] = sdf_str
        output_data_dict[output_cols[1]] = point_ids
        output_data_dict[output_cols[2]] = scan_xyz
        output_list_of_dict.append(output_data_dict)
    logger.info(f'spend time flexible mmff94 is:{time.time() - start_time}')
    return output_list_of_dict
