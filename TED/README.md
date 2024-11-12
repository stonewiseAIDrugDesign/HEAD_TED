# Torsional-energy descriptor
TED (torsional energy descriptor), is mainly composed by a deep learning-based torsion energy prediction model (referred to as the TED-Model henceforth).  
we have two models for inference:  
first model input is multiple conformations, the model name is 'base_model_with_xtb_dft_finetuning.h5'.  
another model input is one conformation ,the model name is 'augmentation_xtb_dft_fine_tune_model.h5'.  
##  model 1
### environment configurations:  
sw_torsion_dnn/requirements.txt.
### model url:  
https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/base_model_with_xtb_dft_finetuning.h5  
https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/train_valid_scale.pkl  
### example input file:  
TED/test_data_multiple_conformations.csv  
### run script:  
```
mkdir /home/train_test_data  
cp base_model_with_xtb_dft_finetuning.h5 /home/train_test_data  
cp train_valid_scale.pkl /home/train_test_data  
cd inference_multiply_conformation  
python run_with_merge_multiply_conformation.py --data-path /home/test_data_multiple_conformations.csv --out-path /home/out.csv  
```
## model 2
### environment configurations:  
sw_torsion_dnn/requirements.txt.    
### model url:  
https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/augmentation_xtb_dft_fine_tune_model.h5    
https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/train_valid_scale.pkl   
### example input file:  
TED/test_data_one_conformation.csv  
### run script:  
```
mkdir /home/train_test_data  
cp augmentation_xtb_dft_fine_tune_model.h5 /home/train_test_data  
cp train_valid_scale.pkl /home/train_test_data  
cd inference_one_conformation  
python run_with_merge_one_conformation.py --data-path /home/test_data_one_conformation.csv --out-path /home/out.csv  
```
## Description of input 
The input is csv format,the column names is:conformation,dihedral  
dihedral format:5-8-9-10  
The ID of a conformation must be unique. If a molecule has multiple conformations, they should be named as base-name_0, base-name_1, etc  
The underscore (_) is used to label different conformations of the same molecule, so the base-name should not contain an underscore  
base_mame is the molecule id  
## Description of output
The output is csv format. the column names is: mol_id,unique_key,energy  
The mol_id represents the molecular ID.  
The unique_key is composed of mol_id, double underscores (__), and dihedral_value, formatted as:   
mol_id__dihedral_value  
The energy represents the relative energy for each dihedral, with the minimum energy set to zero.  
## Conformation generation method
There are two methods to obtain multiple conformations:  
1.  Schrodinger ConfGenX  
2.  OpenBabel  
### ConfGenX cmd:  
```
confgenx {input.mae} -m {conformation_size} -LOCAL -HOST localhost:32  -NJOBS 32 -optimize -force_field OPLS3e -WAIT
```
### Openbabel cmd:  
```
obabel {input.sdf} -O {output.sdf} --confab --xcutoff 0.5 --ecutoff 30 --conf {conformation_size}  
```
