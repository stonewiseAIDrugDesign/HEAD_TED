we have two model for inference  
one input is multiple conformation, the model name is 'base_model_with_xtb_dft_finetuning.h5'  
another input is one conformation ,the model name is 'augmentation_xtb_dft_fine_tune_model.h5'  

#first model run with those configurations
environment configurations from sw_torsion_dnn/requirements.txt.  
model url:  
https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/base_model_with_xtb_dft_finetuning.h5  
https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/train_valid_scale.pkl  
example input file:  
TED/test_data_multiple_conformations.csv  
run script:  
```
mkdir /home/train_test_data  
copy base_model_with_xtb_dft_finetuning.h5 /home/train_test_data  
copy train_valid_scale.pkl /home/train_test_data  
cd inference_multiply_conformation  
python run_with_merge_multiply_conformation.py --data-path /home/test_data_multiple_conformations.csv --out-path /home/out.csv  
```
###second model run with those configurations
model url:  
https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/augmentation_xtb_dft_fine_tune_model.h5    
https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/train_valid_scale.pkl   
example input file:  
TED/test_data_one_conformation.csv  
run script:  
```
mkdir /home/train_test_data  
copy augmentation_xtb_dft_fine_tune_model.h5 /home/train_test_data  
copy train_valid_scale.pkl /home/train_test_data  
cd inference_one_conformation  
python run_with_merge_one_conformation.py --data-path /home/test_data_one_conformation.csv --out-path /home/out.csv  
```
input path is .csv,the row is conformation,dihedral  
dihedral format:5-8-9-10  
conformation name is unique,if have multiple conformations name like base-name_1,test_base-name_2.  
'_'is used for label the same molecular different conformation. so the base-name should not include '_'  
base_mame is the mol_id  


the output is csv format. the column names is: mol_id,unique_key,energy  
mol_id is conformation id, the unique_key is consisted of 'mol_id','__' and 'dihedral_value',format is: 
'mol_id__dihedral_value'  

get the multiple conformations have two methods  
one is Schrodinger confgenx   
cmd:
confgenx {input.mae} -m {conformer_size} -LOCAL -HOST localhost:32  -NJOBS 32 -optimize -force_field OPLS3e -WAIT  
another is openbabel  
obabel {input.sdf} -O {output.sdf} --confab --xcutoff 0.5 --ecutoff 30 --conf {conformer_size}  
