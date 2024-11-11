# Assessing Conformation Validity and Rationality of Deep Learning-Generated 3D Molcules

---

Recent advancements in artificial intelligence (AI) have opened new frontiers in 3D molecule generation, with applications in drug design, materials science, and more. However, evaluating the quality of generated 3D conformations remains challenging due to limitations in current methods. This project presents an open-source solution to address these limitations by combining speed with quantum mechanical (QM)-level accuracy.

## Overview

Most current evaluation methods for AI-generated 3D molecules rely either on empirical geometric metrics, which may overlook subtle conformational issues, or on molecular mechanics (MM) energy calculations, which often lack accuracy and atomic/torsional detail. This project introduces a two-stage approach to improve upon existing evaluation techniques:

1. **Validity Test**: <u>H</u>igh-<u>E</u>nergy <u>A</u>tom <u>D</u>etection ([HEAD](https://github.com/stonewiseAIDrugDesign/HEAD_TED/blob/main/HEAD/README.md)) utilizes an AI-derived force field to identify atoms with elevated energy levels caused by implausible neighboring environments.
2. **Rationality Test**: <u>T</u>orsional <u>E</u>nergy <u>D</u>escriptor (TED) applies a deep learning model trained with density functional theory (DFT)-level accuracy to detect torsional energies, specifically around rotatable bonds.

Our method has been tested on five prominent AI-driven 3D molecule generation models, namely Lingo3DMol, Pocket2Mol, PocketFlow, TargetDiff and PMDM across 101 targets in the Directory of Useful Decoys-Enhanced (DUD-E) dataset. 

![Schematic plot](./assets/schematic_plot.png "HEAD & TED")

## Installation

To use this package, clone the repository and install the dependencies:

```bash
git clone https://github.com/stonewiseAIDrugDesign/HEAD_TED.git

```

## Usage
After installation, you can run the evaluation pipeline on your own dataset of generated molecules or replicate the experiments conducted in this study.

### 1. Running the Validity Test

```python
from head import HEAD

# Input a csv that stores sdf strings with column name, e.g., conformer_sdf
head = HEAD(
    file_path='example.csv',
    csv_column="conformer_sdf",  # the field that stores sdf string of input molecules
    gpu=0
)

# Or input an sdf file that stores one of many molecule conformations
head = HEAD(
    file_path='example.sdf',
    gpu=0
)

head.run(use_info_entropy=True)
head.write_report(output_csv="./head_report.csv")
```

### 2. Running the Rationality Test
```python

# to be update
```

## Example Datasets
To facilitate testing, example datasets are provided. You can also download the released GM5K, GM1K or generated molecules by each model via the link: to be uploaded.

## Applications and Validation
This evaluation framework has been applied to five recent AI-driven models for 3D molecule generation and evaluated on 101 targets in the DUD-E dataset. The approach can be adapted to other datasets or AI models for molecule generation.

## Citation
If you find our code useful, please cite:

more to come...

## License
This project is licensed under the MIT License.




