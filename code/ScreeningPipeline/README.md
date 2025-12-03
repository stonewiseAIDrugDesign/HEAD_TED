# Screening Pipeline

This screening pipeline is designed to evaluate AI-generated molecules using a sequential filtering approach. The pipeline incorporates below filters:

*   **Drug-likeness Assessment:** Molecules must meet the following criteria: QED ≥ 0.3 and SAS ≤ 5.
*   **Pocket-Ligand Interaction & Conformation Validity:** Assessed using HEAD.
*   **Torsional Strain Assessment:** Evaluated using TED.

## How to Run

Execute the `reproduce_ScreeningPipeline.sh` script to run a demonstration of the screening pipeline on the Example Data.

The primary script, `Pipeline_main.py`, is designed to enable users to evaluate their own datasets. This script requires the following input parameters:
- **in_csv**: The primary input file containing initial molecular conformations and their unique identifiers, such as `example_data/AI_generated_raw_conformations.csv`.
- **id_column**: The column name in the input CSV that designates unique molecular identifiers.
- **pocket_column**: The column name in the input CSV that designates the pocket name for each ligand.
- **conf_column**: The column name in the input CSV that designates raw molecular conformations.
- **ref_proteins_path**: The file path to the folder storing PDB files of proteins used as external inputs for the HEAD test. These PDB files must be named exactly equal to the pocket names listed in the input CSV.
- **target_path**: The directory where results will be saved.

## Example Data
An example dataset containing 250 molecules is provided in the `example_data` directory. These molecules will be processed through the screening pipeline, with molecules passing the previous filter serving as input for the subsequent filter.

**Important Note:** As discussed in the manuscript, we recommend using force-field-refined ligand conformations as input for TED assessment. In the manuscript, we used OPLS3e from Schrödinger Suites for this refinement. However, due to commercial licensing restrictions, we are unable to provide the Schrödinger Suites software for *in situ* conformational minimization within this demonstration. Therefore, we have also included OPLS3e optimized conformations of the 250 example molecules in the `example_data` directory. These optimized conformations allow users to bypass the *in situ* optimization step in this demonstration.

The `example_data` directory contains the following files:

*   `AI_generated_raw_conformations.csv`: This file contains the initial AI-generated molecular conformations and serves as the primary input for the screening pipeline. 
*   `OPLS3e_optimized_conformations.csv`: This file contains OPLS3e optimized molecular conformations. To bypass *in situ* force field optimization, molecules that pass the DrugLike and HEAD filters are mapped to their corresponding optimized conformations in this file. These optimized conformations are then used as input for the TED analysis.
*   `Proteins_wo_ligands`: This folder houses the PDB files of proteins without ligands. Molecules that pass the DrugLike filter are mapped to their respective proteins in this folder. These proteins serve as external inputs for the ligand-protein interaction evaluation in the HEAD analysis.

The pipeline generates two output files in the `results` folder:
*   `pipeline.log`: Records the execution time for each step, along with the input count and the number of outputs that passed each filtration step.
*   `Pipeline_result.csv`: Contains detailed results for each molecule at every step, with columns named **DrugLike**, **Valid_ligand_pocket_interaction (HEAD)**, **Valid_ligand_conformation (HEAD)**, and **Rational_torsion_energy (TED)**. Values in these columns denote:
    - **Y**: Conformation passes the step.
    - **N**: Conformation fails the step.
    - **N/A**: Conformation is not supported by the tool at this step but is retained for subsequent steps.
    - *nan*: Empty cell indicates exclusion. Conformation was removed in a prior step and thus not included as input for this step.
