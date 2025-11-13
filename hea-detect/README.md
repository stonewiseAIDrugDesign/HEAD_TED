# High-Energy Atom Detection (HEAD)

**Paper preprint on [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.11.10.622844v1)**

HEAD utilizes an AI-derived force field (ANI-2x (http://doi.org/10.1021/acs.jctc.0c00121) in this work) to identify atoms with elevated energy levels caused by implausible neighboring environments.

## Create Environment

```bash
cd hea-detect/
# create conda env from yml file
conda env create -f head_env.yml

# activate the environment
conda activate head_env

# install setup.py
pip install -e .
```

## Running the Validity Test
After installation, you can run the evaluation pipeline on your own dataset of generated molecules or repeat the experiments conducted in this study.

Below is a code snippet demonstrating the detailed usage of the HEAD toolkit. Alternatively, you can execute it directly using command line commands.

---

### Method 1 (Code snippet)

Prepare an input file that contains one or multiple molecule conformations. **Note** that, HEAD requires conformations with Hydrogen atoms, if the input file does not contain Hydrogens, please set the `add_Hs` to `True` and we use RDKit to complement Hydrogens. 

```python
from hea_detector import HEAD

head = HEAD()
```

Alternatively, you can provide an `.sdf` file,

```python
# Alternatively, you can input an sdf file that stores one or many molecule conformations
head = HEAD(
    
)
```

Start evalutation for detecting physically implausible conformations, and save the HEAD report to csv file.
```python
# Run the HEAD from an input SDF file
head.run(
    file_path="examples/example.sdf",
    # add_Hs =True, # if the input conformation does not contain the full hydrogen information
    use_info_entropy=True
)
# or a .csv file
head.run(
    file_path="examples/example.csv",
    csv_column="conformer_sdf",  # the field that stores SDF string of input molecules
    # add_Hs =True, # if the input conformation does not contain the full hydrogen information
    use_info_entropy=True
)

# or molecules list storing rdkit mol object
head.run(
    mol_list=YOUR_RDKIT_MOl_LIST,  # the field that stores SDF string of input molecules
    # add_Hs =True, # if the input conformation does not contain the full hydrogen information
    use_info_entropy=True
)

# Save the detected results into csv file
head.write_report(output_csv="./head_report.csv")
```

*(Optional)* Plot the atomic-level evaluation result for the conformation,

```python
# Plot the 0-th conformation evaluation result
head.plot(index=0)
```
Then, you should obtain results similar to the example below, which displays the atomic-level details of the HEAD results. The sections circled in red in each conformation correspond to the red bars in the accompanying bar plots.

![Invalid Cases](../assets/invalid_cases.png "Invalid Cases")
---

### Method 2 (Command line)

Run the following command,
```bash
hea-detect --file_path examples/example.sdf --write_report --plot
```
Then, you should find the output report (`HEAD_report.csv`) and plot (`HEAD_fig_0.png`) stored under this directory.

## Speed

When running HEAD for a large amount of molecule conformations, HEAD takes ***around 50 conformations per second*** on one single GPU (e.g., NVIDIA GeForce RTX 3090).

## About the HEAD report
The HEAD report stores both molecule-level and atomic-level information. Please refer the following sections for a better understanding of the report.
- **invalidity**: whether this conformation is valid or not. 
    - `0`: Valid conformation
    - `1`: Invalid conformation
    - `-1`: Unsupported conformation, which may contain elements outside of {H, C, N, O, F, S, Cl}, or may have encountered an unexpected error during loading

- **invalid atoms**: Contains all atomic-level invalid details if the current conformation is detected as invalid; Otherwise, it is `None`. for example,
    - `[(2, 'C', 40.962)]`: This indicates that the No.2 (index starts from 1) Carbon atom is detected as invalid due to the high-energy response `40.962` kcal/mol. (Note that this energy is only a reference and may not be precise.)

- **HES**: Decribes the level of invalidity of this conformation. It is zero if the conformation is valid; Otherwise, it is a non-negative vlaue. The greater the HES is, the more problematic the conformation.

- **atom types**: Stores the atom types for an input conformation.

- **information entropy**: The computed information entropy for the maximum subregion. This approach is ONLY supplementrary for HEAD (see our paper for details).

- **information entropy invalidity**: If `1`, it indicates that an invalid conformation was detected ONLY by the information entropy approach; Otherwise, it is `0`.


