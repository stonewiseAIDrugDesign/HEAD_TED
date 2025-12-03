import os
import pandas as pd
import argparse
from rdkit import Chem
from tools.DrugLikeFilter import druglike_filter_mp
from tools.temp_control import TempDir
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from HEAD.head import HEAD
from TED.ted import TED
import torch
import logging

logger = logging.getLogger('Pipeline_Log')
# log settings
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.setLevel("INFO")
logger.addHandler(stream_handler)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_csv', required=True)
    parser.add_argument('--id_column', required=True)
    parser.add_argument('--pocket_column', required=True)
    parser.add_argument('--conf_column', required=True)
    parser.add_argument('--ref_proteins_path', required=True)
    parser.add_argument('--target_path', required=True)
    args = parser.parse_args()

    file_handler = logging.FileHandler(os.path.join(args.target_path, 'pipeline.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # DrugLikeFilter
    df = pd.read_csv(args.in_csv)
    raw_sdf_list = df[args.conf_column].tolist()
    df['DrugLike_failure'] = druglike_filter_mp(raw_sdf_list)
    _df = df[df['DrugLike_failure']!= 1]
    _df.reset_index(inplace=True, drop=True)
    logger.info(f'Complete DrugLike Filteration. Input Num: {len(df)}, Output Num: {len(_df)}')

    # HEAD 
    protein_id_mol_id_mapping, mol_id_raw_sdf_mapping = dict(), dict()
    for i in range(len(_df)):
        protein_id = _df[args.pocket_column].iloc[i]
        mol_id = _df[args.id_column].iloc[i]
        if protein_id not in protein_id_mol_id_mapping:
            protein_id_mol_id_mapping[protein_id] = []
        protein_id_mol_id_mapping[protein_id].append(mol_id)
        mol_id_raw_sdf_mapping[mol_id] = _df[args.conf_column].iloc[i]
    
    temp_manager = TempDir()
    temp_dir = temp_manager.__enter__(os.path.dirname(os.path.abspath(__file__)))
    head_results = []
    for protein_id, mol_id_list in protein_id_mol_id_mapping.items():
        mols = [Chem.MolFromMolBlock(mol_id_raw_sdf_mapping[mol_id], removeHs=False) for mol_id in mol_id_list]
        ligands_collection = os.path.join(temp_dir, f'{protein_id}_ligands.sdf')
        f = Chem.SDWriter(ligands_collection)
        for m in mols:
            f.write(m)
        f.close()

        head = HEAD(ligands_path=ligands_collection, 
            protein_path=os.path.join(args.ref_proteins_path, f'{protein_id}.pdb'),
            gpu=0,
            )
        head.run(use_info_entropy=False)
        head.run_pose_checking(use_info_entropy=False)

        for i in range(len(head.molecules)):
            mol = head.molecules[i]
            mol_name = mol.GetProp('_Name')
            try:
                ligand_energy = head.ligand_in_pocket_energies[i].sum() 
                mol_energy = head.energies[i].sum()
                hes = head.scores[i]
                ncie = head.binding_energy[i]
            except:
                ligand_energy = None
                mol_energy = None
                ncie = None
                hes = None
            head.records_indices[i]
            head_results.append([mol_name, int(head.records_ligand_pose_indices[i]), int(head.records_indices[i])])
    temp_manager.__exit__()
    head_df = pd.DataFrame(head_results, columns=[args.id_column, 'HEAD_ligand_pocket_interaction_invalidity', 'HEAD_ligand_conformation_invalidity'])
    df = pd.merge(df, head_df, how='left', on=args.id_column)

    logger.info(f"Complete HEAD Filteration. Input Num: {len(_df)}, Output Num: {len(head_df[(head_df['HEAD_ligand_pocket_interaction_invalidity'] != 1) & (head_df['HEAD_ligand_conformation_invalidity'] != 1)])}")

    # Refinement
    # Use the min_sdf column to bypass this step
    df_min = pd.read_csv('example_data/OPLS3e_optimized_conformations.csv')
    df = pd.merge(df, df_min, how='left', on='mol_id')

    # TED
    temp_manger = TempDir()
    temp_dir = temp_manager.__enter__(os.path.dirname(os.path.abspath(__file__)))
    _df = df[(df['DrugLike_failure'] != 1) & (df['HEAD_ligand_pocket_interaction_invalidity'] != 1) & (df['HEAD_ligand_conformation_invalidity'] != 1)]

    _df = _df[[args.id_column, 'min_sdf']]
    ted_input_temp_filepath = os.path.join(temp_dir, 'ted_inputs.csv')
    _df.to_csv(ted_input_temp_filepath, index=False)

    ted = TED(
        file_path=ted_input_temp_filepath,
        model_path='../../data/trained_model_augmentation',
        out_csv=os.path.join(temp_dir, 'TED_results.csv'),
        detailed_output=False,
        id_column=args.id_column,
        sdf_column='min_sdf',
        augmentation_model=True,
    )
    ted.run()

    ted_df = pd.read_csv(os.path.join(temp_dir, 'TED_results.csv'))
    df = pd.merge(df, ted_df, how='left', on=args.id_column)

    logger.info(f"Complete TED Filteration. Input Num: {len(_df)}, Output Num: {len(ted_df[(ted_df['TED_torsion_energy_irrationality'] != 1)])}")
    temp_manager.__exit__()

    df = df[[args.id_column, 'DrugLike_failure', 'HEAD_ligand_pocket_interaction_invalidity', 'HEAD_ligand_conformation_invalidity', 'TED_torsion_energy_irrationality']]
    df.rename(columns={'DrugLike_failure': 'DrugLike', 'HEAD_ligand_pocket_interaction_invalidity': 'Valid_ligand_pocket_interaction (HEAD)', 
    'HEAD_ligand_conformation_invalidity': 'Valid_ligand_conformation (HEAD)',
    'TED_torsion_energy_irrationality': 'Rational_torsion_energy (TED)'}, inplace=True)
    df['DrugLike'] = df['DrugLike'].map({1: 'N', 0: 'Y', -1: 'N/A'})
    df['Valid_ligand_pocket_interaction (HEAD)'] = df['Valid_ligand_pocket_interaction (HEAD)'].map({1: 'N', 0: 'Y', -1: 'N/A'})
    df['Valid_ligand_conformation (HEAD)'] = df['Valid_ligand_conformation (HEAD)'].map({1: 'N', 0: 'Y', -1: 'N/A'})
    df['Rational_torsion_energy (TED)'] = df['Rational_torsion_energy (TED)'].map({1: 'N', 0: 'Y', -1: 'N/A'})

    df.to_csv(os.path.join(args.target_path,'Pipeline_result.csv'), index=False)

