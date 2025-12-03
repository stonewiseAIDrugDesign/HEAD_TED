# Torsional-Energy Descriptor (TED)
TED is primarily powered by a deep learning-based torsion energy prediction model, referred to as the TED-Model.

We provide two distinct models for inference, each tailored to different use cases and types of input data. Please put the **Main Model** and its corresponding **Data Scaling information** under the same directory.
1. The **Base-model** is trained using a combination of GFN2-xTB and DFT (Density Functional Theory) data, processing multiple numbers of initial conformations for each torsion fragments as inputs.

    **Main Model:** [base_model_with_xtb_dft_finetuning.h5](https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/base_model_with_xtb_dft_finetuning.h5) \
    **Data Scaling information:** [train_valid_scale.pkl](https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/train_valid_scale.pkl)

2. The **Augmented-model** is an adaptation of the base model that incorporates data augmentation, processing single initial conformation of each torsion fragments as input.

    **Main Model:** [augmentation_model_with_xtb_dft_finetuning.h5](https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/augmentation_xtb_dft_fine_tune_model.h5) \
    **Data Scaling information:** [train_valid_scale.pkl](https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/train_valid_scale.pkl)

## Running the Test
### Run Test
1. Download the **Base-model** or **Augmented-model** 
2. Place the downloaded files into a new folder. Ensure the filenames remain unchanged. Use Base-model as example.
```
mkdir ./trained_model_base
cp base_xtb_dft_fine_tune_model.h5 ./trained_model_base/
cp train_valid_scale.pkl ./trained_model_base/
```
3. Run the command for test and enjoy! :)
```
python3 ted.py --file_path ./examples/TED_example.csv --id_column mol_id --sdf_column optimized_sdf --model_path ../../data/trained_model_base --out_csv TED_result.csv
```
Or try this command for the Augmented-model. (Included in `example_reproduce.sh` with correct relative path for **one-click reproducible run** in capsule)

```
python3 ted.py --file_path ./examples/TED_example.csv --id_column mol_id --sdf_column optimized_sdf --model_path ../../data/trained_model_augmentation --augmentation_model --out_csv TED_result.csv
```

### Important Notes Before Starting Your Own Job
The input CSV file must contain at least one column with the conformations of each molecule in **SDF V2000** format. Each cell should contain only one conformation. If a column for unique identifiers is provided, please ensure that the identifiers do not contain underscores (`_`).

### Initial Conformation of Torsion Fragments
The provided pipeline has already integrated two methods for generating the initial conformations for torsion fragments.

1. OpenBabel
2. Schrodinger's ConfGen & ConfGenX

If user provide the path to the Schrodinger Series Executable via argument `--confgen_path` and `--structconvert`(optional, used to convert .maegz to readable .sdf files, this will be automatically detected if installed in the same directory as ConfGen), the script will use ConfGen & ConfGenX. If these paths are not provided, OpenBabel will be used instead. 


## Speed
When using the TED-Model for inference, only CPUs are required. The Augmented-Model takes approximately 180 seconds to process 7000 torsion fragments on a machine with 48 cores, whereas the Base-Model takes around 880 seconds. Note that the majority of the processing time is spent on the initial conformation sampling. If OpenBabel is used for conformation generation, the overall time required will be approximately halved.

## About the TED report
### Brief Result
The output of the **TED-Model** is a `.csv` file containing two columns: `id` and `TED_result`. 

- If the `--id_column` argument is not provided, the `id` in the output CSV will be set to the index of the input CSV.
- If the `--id_column` argument is provided, the `id` will correspond to the values in the specified column.

The `TED_result` column contains flags indicating whether the molecular conformation passed the evaluation:
- `0` indicates the conformation passed.
- `1` indicates the conformation did not pass.
- `-1` indicates the molecule could not be processed.

### Detailed Outputs
If the `--detailed_output` argument is provided, detailed information for each dihedral angle of every query molecule will be included and saved as a `.json` file. This file will have the same filename as the brief output CSV. 

Each record for a dihedral angle will include:
- A flag indicating whether the evaluation passed.
- A quad joined by `-` representing the atom indices that define the specific dihedral in the query molecule.
- The SMILES string of the torsion fragment being investigated.
- The degree of the dihedral angle.
- The predicted relative energy (with the minimum energy set to zero).

An example of the detailed output can be found in `examples/TED_example_detailed_output.json`.