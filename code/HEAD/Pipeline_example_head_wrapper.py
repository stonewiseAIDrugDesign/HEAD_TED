import os
from head import HEAD
import pandas as pd
import os

dude_proteins = '/code/ScreeningPipeline/example_data/HEAD/DUDE_proteins'
pipeline_ligands = '/code/ScreeningPipeline/example_data/HEAD/mol_DrugLike_inputs'
if __name__ == '__main__':
    results = []
    for _, _, files in os.walk(dude_proteins):
        for f in files:
            this_pdb_id = f.split('_')[0]
            this_protein = os.path.join(dude_proteins, f)
            this_ligand = os.path.join(pipeline_ligands, f"{this_pdb_id}_ligands_druglike.sdf")
            print(f">>>>processing {this_pdb_id}")
            head =HEAD(
                ligands_path=this_ligand,
                protein_path=this_protein,
                gpu=0,
            )
            head.run(use_info_entropy=False)
            head.run_pose_checking(use_info_entropy=False)

            for i in range(len(head.molecules)):
                mol = head.molecules[i]
                mol_name = mol.GetProp('_Name') # 1e66_Lingo3DMolv2_1009291
                mol_pdb_id = mol_name.split('_')[0]
                assert mol_pdb_id == this_pdb_id
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
                results.append([mol_name, int(head.records_ligand_pose_indices[i]), int(head.records_indices[i])])
            
    head_df = pd.DataFrame(data=results, columns=['mol_id', 'HEAD_ligand_protein_interaction_invalidity', 'HEAD_ligand_conformation_invalidity'])
    head_df.to_csv('/results/Pipeline_HEAD_output.csv', index=False)