import logging

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import pandas as pd
import multiprocessing as mp
import sys
import json
import time
import scipy.interpolate
sys.path.append('.')
from pesfrager import conf_fragmenter
from utils.temp_control import TempDir, split_sdfs
import scipy
import math
import os
import shutil
import pickle
import numpy as np
import subprocess as sp
from typing import Optional, Union
import argparse

RDLogger.DisableLog('rdApp.*') 
logger = logging.getLogger(__name__)
# log settings
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.setLevel("INFO")
logger.addHandler(stream_handler)

class TED():
    """Class to run TED for input molecule conformation(s)"""
    def __init__(self,
                 file_path: str,
                 model_path: str=None,
                 out_csv: str=None,
                 detailed_output: bool=False,
                 id_column: Optional[str]=None,
                 sdf_column: str=None,
                 preparation_num_proc: Optional[int]=None,
                 confgen_path: Optional[str]=None,
                 structconvert_path: Optional[str]=None,
                 augmentation_model: Optional[bool]=True):
        self.file_path = os.path.abspath(file_path)
        if preparation_num_proc is None:
            self.preparation_num_proc = os.cpu_count()
        else:
            self.preparation_num_proc = preparation_num_proc
        self.confgen_path = confgen_path
        if self.confgen_path is not None and structconvert_path is None:
            self.structconvert_path = os.path.join(os.path.split(confgen_path)[0], 'utilities', 'structconvert')
        self.augmentation_model = augmentation_model
        self.model_path = os.path.abspath(model_path)
        self.out_path = os.path.join(os.getcwd(), out_csv)
        if detailed_output:
            self.detailed_out_path = os.path.join(os.getcwd(), out_csv.split('.')[0] + '.json')
        else:
            self.detailed_out_path = None
        self.ids, self.molecules = self.load_molecules_from_file(self.file_path, id_column, sdf_column)

    def load_molecules_from_file(self, file_path, id_column, sdf_column):
        mol_df = pd.read_csv(file_path)
        if id_column is None:
            ids = mol_df.index
        else:
            ids = mol_df.loc[:, id_column].tolist()
        sdf_string = mol_df.loc[:, sdf_column].tolist()
        molecules = []
        for i in range(len(sdf_string)):
            try:
                mol = Chem.MolFromMolBlock(sdf_string[i], removeHs=False)
                molecules.append(mol)
            except Exception:
                molecules.append(None)
                continue
        
        if len(molecules) == 0:
            logger.warning("No molecules loaded") # Warning
        logger.info(f"Loaded {len(molecules)} molecules.")
        return ids, molecules

    def conf_process_mp(self, conf_list):
        with mp.Pool(self.preparation_num_proc) as workers:
            res_list = list(workers.imap(conf_fragmenter, conf_list))
        logger.info(f"Fragmented {len(res_list)} molecules.")
        id_fragment_mapping = {self.ids[i]: (True, dict()) for i in range(len(self.ids))}
        for i in range(len(self.ids)):
            id = self.ids[i]
            conf_res = res_list[i]
            if conf_res is None:
                id_fragment_mapping[id] = (False, dict())
                continue
            for dih_res in conf_res:
                if len(dih_res) < 1:
                    continue
                smikey = None
                for dih_res_select in dih_res:
                    if dih_res_select[-1] is None:
                        continue
                    deg = dih_res_select[-1]
                    dih_quad = '-'.join([str(x) for x in dih_res_select[-2]])
                    smikey = dih_res_select[0]
                    smiles = dih_res_select[1]
                    break
                if smikey is None:
                    id_fragment_mapping[id][1][dih_quad] = False
                else:
                    id_fragment_mapping[id][1][dih_quad] = (smikey, deg)
        return id_fragment_mapping
    
    def _read_smikey(self, smikey):
        mol = Chem.MolFromSmiles(smikey)
        quad = []
        for at in mol.GetAtoms():
            if at.GetAtomMapNum() == 100:
                quad.append(at.GetIdx())
                at.ClearProp('molAtomMapNumber')
        mol = Chem.AddHs(mol)
        mol.SetProp('_Name', smikey)
        ps = AllChem.ETKDG()
        ps.randomSeed = 0xf00d
        AllChem.EmbedMolecule(mol, ps)
        return mol, quad

    def init_fragment_conf(self, smikey_list):
        temp_manager = TempDir()
        tmp_dir = temp_manager.__enter__()
        tmp_input_path = os.path.join(tmp_dir, 'input.sdf')
        tmp_output_path = os.path.join(tmp_dir, 'output.sdf')
        fh = Chem.SDWriter(tmp_input_path)
        raw_mols, quad_list = [], []
        smikey_smi_id_mapping = dict()
        for i, smikey in enumerate(smikey_list):
            smikey_smi_id_mapping[smikey] = i
            try:
                _mol, quad = self._read_smikey(smikey)
            except:
                _mol, quad = None, None
            quad_list.append(quad)
            raw_mols.append(_mol)
            if _mol is None:
                continue
            fh.write(_mol)
        fh.close()

        if self.augmentation_model:
            n_conf = 1
        else:
            n_conf = 20

        if self.confgen_path is None:
            # sdfs = open(tmp_input_path, 'r').read()
            # input_sdf_list = split_sdfs(sdfs)
            # Why split inputs into different pieces? Results of confab method are affected by the order of input structures. :(
            # output_sdf_list = obabel_mp(input_sdf_list, xcutoff=0.5, ecutoff=20, conf_num=n_conf, workdir='.', num_proc=self.preparation_num_proc)
            # with open(tmp_output_path, 'w') as f:
            #     for sdf in output_sdf_list:
            #         f.write(sdf)
            sp.run(f'obabel {tmp_input_path} -O {tmp_output_path} --confab --xcutoff 0.5 --ecutoff 20 --conf {n_conf}', shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)
        else:
            cur_path = os.getcwd()
            os.chdir(tmp_dir)
            sp.run(f'{self.confgen_path} input.sdf -m {n_conf} -LOCAL -NJOBS {self.preparation_num_proc} -optimize -force_field OPLS3e -WAIT', shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)
            sp.run(f'{self.structconvert_path} input-out.maegz output.sdf', shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)
            os.chdir(cur_path)
        
        smikey_init_frag_confs_mapping = dict()
        mols = Chem.SDMolSupplier(tmp_output_path, removeHs=False)
        smikey_quad_mapping = dict()
        for i, mol in enumerate(mols):
            smikey = mol.GetProp('_Name')
            smi_id = smikey_smi_id_mapping[smikey]
            raw_mol = raw_mols[smi_id]
            if smikey not in smikey_init_frag_confs_mapping:
                smikey_init_frag_confs_mapping[smikey] = []
                smikey_quad_mapping[smikey] = []
            mol.SetProp('_Name', str(smi_id) + '_' + str(len(smikey_init_frag_confs_mapping[smikey])))
            smikey_init_frag_confs_mapping[smikey].append(Chem.MolToMolBlock(mol))
            
            new_quad = self.remap_quad(raw_mol, quad_list[smi_id], mol)
            smikey_quad_mapping[smikey].append(new_quad)
        temp_manager.__exit__()
        logger.info(f"Generated initial conformations.")
        return smikey_init_frag_confs_mapping, smikey_quad_mapping

    def remap_quad(self, raw_mol, raw_quad, new_mol):
        match_indexes_list = new_mol.GetSubstructMatches(raw_mol)
        if len(match_indexes_list) == 0:
            return None
        match_indexes = match_indexes_list[0]
        new_quad = [match_indexes[i] for i in raw_quad]
        new_quad = self.get_series_point(new_mol, new_quad)
        return new_quad

    def get_series_point(self, mol, quad):
        new_atom_ids = []
        for i in quad:
            temp_atom_attr = mol.GetAtomWithIdx(i)
            atom_ngbs = temp_atom_attr.GetNeighbors()
            ngb_count = 0
            nerg_list = []
            for atom_ngb in atom_ngbs:
                atom_id = atom_ngb.GetIdx()
                if atom_id in quad:
                    nerg_list.append(atom_id)
            if len(nerg_list) == 2 or len(nerg_list) == 0:
                continue
            if len(new_atom_ids) == 0:
                new_atom_ids.append(i)
                new_atom_ids.append(nerg_list[0])
            else:
                new_atom_ids.append(nerg_list[0])
                new_atom_ids.append(i)
        return new_atom_ids

    def get_PES(self, input):
        dih_deg_id_group, dih_deg_energy_group = input
        deg_energy_pairs = []
        for i in range(len(dih_deg_id_group)):
            _, deg = dih_deg_id_group[i].split('__')
            energy = dih_deg_energy_group[i]
            deg_energy_pairs.append((float(deg), float(energy)))
        sorted_pairs = sorted(deg_energy_pairs, key=lambda x: x[0])
        x, y = [item[0] for item in sorted_pairs], [item[1] for item in sorted_pairs]
        pes = scipy.interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
        return pes
    
    def get_PES_mp(self, df):
        inputs = dict()
        for i in range(len(df)):
            smi_id = df.at[i, 'mol_id']
            if smi_id not in inputs:
                inputs[smi_id] = {'dih_deg_id': [], 'dih_deg_energy': []}
            inputs[smi_id]['dih_deg_id'].append(df.at[i, 'unique_key'])
            inputs[smi_id]['dih_deg_energy'].append(df.at[i, 'energy'])
        
        flatten_inputs, smi_id_list = [], []
        for k, v in inputs.items():
            flatten_inputs.append((v['dih_deg_id'], v['dih_deg_energy']))
            smi_id_list.append(k)
        
        with mp.Pool(self.preparation_num_proc) as workers:
            res = list(workers.imap(self.get_PES, flatten_inputs))
        res_dict = dict()
        for i in range(len(smi_id_list)):
            res_dict[smi_id_list[i]] = res[i]
        return res_dict

    def run(self):
        conf_list = []
        for i, mol in enumerate(self.molecules):
            if mol is None:
                conf_list.append(None)
                continue
            conf_list.append(Chem.MolToMolBlock(mol))
        id_fragment_mapping = self.conf_process_mp(conf_list)
        smikey_list = list(dict.fromkeys([v2[0] for k, v in id_fragment_mapping.items() if v[0] for k2, v2 in v[1].items() if v2]))
        smikey_init_frag_confs_mapping, smikey_quad_mapping = self.init_fragment_conf(smikey_list)
        temp_manager = TempDir()
        temp_dir = temp_manager.__enter__()
        rows = []
        for i in range(len(smikey_list)):
            smikey = smikey_list[i]
            for i in range(len(smikey_init_frag_confs_mapping[smikey])):
                rows.append((smikey_init_frag_confs_mapping[smikey][i], '-'.join([str(x) for x in smikey_quad_mapping[smikey][i]])))
        df = pd.DataFrame(rows, columns=['conformer', 'dihedral'])
        input_file_path = os.path.join(temp_dir, 'tmp_input.csv')
        output_file_path = os.path.join(temp_dir, 'tmp_output.csv')
        df.to_csv(input_file_path, index=False)
        cur_dir = os.getcwd()
        if self.augmentation_model:
            os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inference_one_conformation'))
            sp.run(f'python3 run_with_merge_one_conformation.py --data-path {input_file_path} --out-path {output_file_path} --model-path {self.model_path}', shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)
            os.chdir(cur_dir)
        else:
            os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inference_multiply_conformation'))
            sp.run(f'python3 run_with_merge_multiply_conformation.py --data-path {input_file_path} --out-path {output_file_path} --model-path {self.model_path}', shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)
            os.chdir(cur_dir)
        logger.info(f"Predicted PES curves for {len(smikey_list)} fragments.")
        df = pd.read_csv(output_file_path)

        smi_id_pes_mapping = self.get_PES_mp(df)
        temp_manager.__exit__()
        smikey_pes_mapping = {smikey_list[smi_id]: pes for smi_id, pes in smi_id_pes_mapping.items()}
        id_res_mapping = dict()
        for in_conf_id, dih_collection in id_fragment_mapping.items():
            if dih_collection[0]:
                if in_conf_id not in id_res_mapping:
                    id_res_mapping[in_conf_id] = []
                for dih_quad, dih_res in dih_collection[1].items():
                    if dih_res:
                        smikey, deg = dih_res
                        if smikey in smikey_pes_mapping:
                            energy = float(smikey_pes_mapping[smikey](deg))
                            if energy > 2:
                                id_res_mapping[in_conf_id].append((False, dih_quad, smikey, deg, energy))
                            else:
                                id_res_mapping[in_conf_id].append((True, dih_quad, smikey, deg, energy))

        res = []
        for k, v in id_fragment_mapping.items():
            if v[0]:
                if sum([i[0] for i in id_res_mapping[k]]) < len(id_res_mapping[k]):
                    res.append(0)
                else:
                    if id_fragment_mapping[k][1]:
                        res.append(1)
                    else:
                        res.append(0)
            else:
                res.append(2)

        df = pd.DataFrame({'id': self.ids, 'TED_result': res})
        df.to_csv(self.out_path, index=False)
        if self.detailed_out_path is not None:
            with open(self.detailed_out_path, 'w') as f:
                json.dump(id_res_mapping, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        required=True,
        type=str,
        help="The path to the input CSV file containing molecular conformations. Only .csv format is supported."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="The path to the trained model file to be used for processing."
    )
    parser.add_argument(
        "--augmentation_model",
        action="store_true",
        help="Flag to indicate whether model for loading is an augmentation model."
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default=None,
        help="The name of the column in the input file that contains unique molecule identifiers. Will use the index of rows in csv if not provided. Default is None"
    )
    parser.add_argument(
        "--sdf_column",
        type=str,
        required=True,
        help="The name of the column in the input file containing SDF-format molecular data."
    )
    parser.add_argument(
        "--preparation_num_proc",
        type=int,
        default=None,
        help="The number of processor cores to use for data preparation. Default is determined by the system."
    )
    parser.add_argument(
        "--confgen_path",
        type=str,
        default=None,
        help="The path to the Schrodinger's confgen & confgenx executable. Will use confgen & confgenx instead OpenBabel to perform initial conformation sampling if provided. Default is None."
    )
    parser.add_argument(
        "--structconvert_path",
        type=str,
        default=None,
        help="The path to the Schrodinger's structconvert executable. Will try to get the executable under same directory of confgen if not provided. Default is None."
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        type=str,
        help="The path where the output CSV file with processed data will be saved."
    )
    parser.add_argument(
        "--detailed_output",
        action="store_true",
        help="Flag to enable detailed outputs of the input structures. Detailed outputs include the pass&no-pass flag, quad indicating dihedral angle, torsion fragment smiles, degrees, predicted relative energy of all dihedrals in the query molecules."
    )
    args = parser.parse_args()
    obj = TED(**vars(args))
    obj.run()
