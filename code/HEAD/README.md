# High-Energy Atom Detection (HEAD)

**See our paper for details [*Nature Communication*](https://www.nature.com/articles/s41467-026-69303-5)**

HEAD utilizes an AI-derived force field (ANI-2x (http://doi.org/10.1021/acs.jctc.0c00121) in this work) to identify atoms with elevated energy levels caused by implausible neighboring environments.

HEAD supports:
1. ligand conformation validity assessment
2. ligand-protein interaction assessment

## Create Environment

```bash
cd HEAD
# create conda env from yml file
conda env create -f head_env.yml

# activate the environment
conda activate head_env
```

## Running the Ligand Conformation AND(OR) Ligand-Protein Interaction Validity Assessment
After installation, you can run the evaluation pipeline on your own dataset of generated molecules or repeat the experiments conducted in this study.

Below is a code snippet demonstrating the detailed usage of the HEAD toolkit. Alternatively, you can execute it directly using command line commands.

---

### Method 1 (Code snippet)

#### 1. For ligand conformation checking (without binding protein):

Prepare an input file that contains one or multiple molecule conformation(s). **Note** that, HEAD requires conformations with Hydrogen atoms, if the input file does not contain Hydrogens, please set the `add_Hs` to `True` and we use RDKit to complement Hydrogens. 

```python
from head import HEAD

#  Input a csv that stores SDF strings with a column name, e.g., "conformer_sdf". All the content of an .sdf file for a conformation is treated as the SDF string for that specific conformation. For more details, please refer to examples/examples.csv
head = HEAD(
    ligands_path="examples/example.csv",
    csv_column="conformer_sdf",  # the field that stores SDF string of input molecules
    # add_Hs =True, # if the input conformation does not contain the full hydrogen information
)
```
Alternatively, you can provide an `.sdf` file,
```python
# Alternatively, you can input an sdf file that stores one or many molecule conformations
head = HEAD(
    ligands_path="examples/example.sdf",
)
```

Start evalutation for detecting physically implausible conformations, and save the HEAD report to csv file. In addition, the conformation invalidity of all inputs molecules are saved in `head.records_indices`.
```python
# Run the HEAD
head.run(use_info_entropy=True)

# Save the detected results into csv file
head.write_report(output_csv="./head_report.csv")
```

*(Optional)* Plot the atomic-level evaluation result for the conformation,

```python
# Plot the 0-th conformation evaluation result
head.plot(index=0)
```
Then, you should obtain results similar to the example below, which displays the atomic-level details of the HEAD results. The sections circled in red in each conformation correspond to the red bars in the accompanying bar plots.

![Invalid Cases](../../assets/invalid_cases.png "Invalid Cases")
---


#### 2. For ligand conformation and binding pose checking

Prepare an input file that contains one or multiple molecule conformation(s) (e.g., `examples/1b9v_ligands.sdf`) with the binding protein file (e.g., `examples/1b9v_protein_wo_ligand.pdb`).  **Note** that, HEAD requires ligand and protein structures with full Hydrogen atoms.

```python
# We suggest the users use SDF format for ligands and PDB format for protein
head = HEAD(
    ligands_path="examples/1b9v_ligands.sdf",
    protein_path = "examples/1b9v_protein_wo_ligand.pdb",
    residue_cutoff=20, # for speed consideration, the protein is cut as a pocket by considering residues within 20 Angst (by default) of a ligand.
)
```

When ligands and protein are given as inputs, HEAD runs both ligand conformation checking and ligand binding pose checking by `run_pose_checking()` method. In addition, the detected ligand-protein interaction results all inputs ligands with its binding protein are saved in `head.records_ligand_pose_indices`.
```python
# Run the HEAD
head.run_pose_checking()

# (optional) Save the detected results into csv file
head.write_report(output_csv="./head_report.csv", check_pose=True)
```

*(Optional)* Plot the atomic-level evaluation result for the conformation,

```python
# Plot the 14-th liagnd pose evaluation result
head.plot_ligand_dE(index=13) # because the 14th pose is detected as invalid, index starts from 0
```
---

### Method 2 (Command line)

Run the following command for ligand conformation validity checking,
```bash
python head.py --ligands_path examples/example.sdf --write_report --plot
```
Then, you should find the output report (`HEAD_report.csv`) and plot (`HEAD_fig_0.png`) stored under this directory.

Or run the following command for ligand-protein intearction checking,
```bash
python head.py --ligands_path examples/1b9v_ligands.sdf --protein_path examples/1b9v_protein_wo_ligand.pdb --write_report 
```
## Speed

1. Ligand conformation checking: when running HEAD for a large amount of molecule conformations, HEAD takes ***around 50 conformations per second*** on one single GPU (e.g., NVIDIA GeForce RTX 4090).
2. Ligand-protein interaction checking: this normally takes ***around 5~8 ligands bound to one same protein per second*** one one single GPU (e.g., NVIDIA GeForce RTX 4090).

## About the HEAD report
The HEAD report stores both molecule-level and atomic-level information. Please refer the following sections for a better understanding of the report.
- **ligand invalidity**: whether this molecular conformation is valid or not. 
    - `0`: Valid conformation
    - `1`: Invalid conformation
    - `-1`: Unsupported conformation, which may contain elements outside of {H, C, N, O, F, S, Cl}, or may have encountered an unexpected error during loading

- **ligand invalid atoms**: Contains all atomic-level invalid details if the current conformation is detected as invalid; Otherwise, it is `None`. for example,
    - `[(2, 'C', 40.962)]`: This indicates that the No.2 (index starts from 1) Carbon atom is detected as invalid due to the high-energy response `40.962` kcal/mol. (Note that this energy is only a reference and may not be precise.)

- **ligand HES**: Decribes the level of invalidity of this conformation. It is zero if the conformation is valid; Otherwise, it is a non-negative vlaue. The greater the HES is, the more problematic the conformation.

- **ligand atom types**: Stores the atom types for an input conformation.

- **ligand information entropy**: The computed information entropy for the maximum subregion. This approach is ONLY supplementrary for HEAD (see our paper for details).

- **ligand information entropy invalidity**: If `1`, it indicates that an invalid conformation was detected ONLY by the information entropy approach; Otherwise, it is `0`.

- **pose invalidity**: whether this binding pose is valid or not. 
    - `0`: Valid ligand-protein interaction
    - `1`: Invalid ligand-protein interaction
    - `-1`: Unsupported complex, which may contain elements outside of {H, C, N, O, F, S, Cl}, or may have encountered an unexpected error during loading

- **bound ligand energy**: Predicted total energy of ligand in bound state by the MLFF, unit in kcal/mol.

- **isolated ligand energy**: Predicted total energy of ligand in isolated state by the MLFF, unit in kcal/mol.

- **binding energy**: Predicted binding energy of ligand and its binidng protein ($E^{bound}_{complex}$ - $E^{isolated}_{lig}$ - $E^{isolated}_{protein}$) by the MLFF, unit in kcal/mol.

- **pose HES**: Decribes the level of invalidity of the given ligand by comparing its energy difference of a bound state and its isolated state, by $E^{bound}_{lig}$ - $E^{isolated}_{lig}$. It is zero or negative if the conformation is valid; The greater the HES is, the more problematic the pose is.


## (Optional) Using MACE-OFF model

HEAD can be adapted to other MLFF, such as [MACE-OFF](https://github.com/ACEsuit/mace-off/tree/main), which enbales fast and accurate machine learning interatomic potentials with higher order equivariant message passing.

Specifically, MACE-OFF24(M) model is used to check ligand conformation validity and ligand-protein interaction validity evaluation.

**Note that**: Compared with ANI-2x,
1. MACE-OFF24(M) supports more elements of `{H, C, N, O, F, P, S, Cl, Br, I}`.
2. MACE-OFF24(M) can give more accurate binding enegry prediction (based on our testing of Pearson Correlation with MM/GBSA), but ANI-2x is much more faster and requires much smaller GPU meomory.
3. The ligand conformation checking by using MACE-OFF24(M) requires more efforts of tunning the elemental-wise thresholds, so just try it as a toy model:)

```python
# We suggest the users use SDF format for ligands and PDB format for protein
head = HEAD(
    ligands_path="examples/1b9v_ligands.sdf",
    protein_path = "examples/1b9v_protein_wo_ligand.pdb",
    residue_cutoff=8.0, # Cutting a bigger protein pocket may cause OOM, so set it to 8.0 instead.
    mace_model_path='medium', # 'medium' or the path of MACE-OFF24(M) model
)

# Run ligand conformation validity checking
head.run_mace_conf()

# Or run ligand-protein validity checking
head.run_mace_pose()
```

Or using the commands,

```bash
# ligand conformation validity checking
python head.py --ligands_path examples/1b9v_ligands.sdf --write_report --use_mace_off

# Or ligand-protein validity checking
python head.py --ligands_path examples/1b9v_ligands.sdf --protein_path examples/1b9v_protein_wo_ligand.pdb --write_report --use_mace_off
```
