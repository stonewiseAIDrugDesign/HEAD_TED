from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score import sascorer
import multiprocessing as mp
import logging
import os
import time
import argparse
import pandas as pd

RDLogger.DisableLog('rdApp.*') 
logger = logging.getLogger(__name__)
# log settings
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.setLevel("INFO")
logger.addHandler(stream_handler)

def druglike_filter(molblock):
    try:
        mol = Chem.MolFromMolBlock(molblock)
    except:
        mol = None
    
    if mol is None:
        return -1

    qed = Descriptors.qed(mol)
    sas = sascorer.calculateScore(mol)
    if qed >= 0.3 and sas <= 5:
        return 0
    else:
        return 1

def druglike_filter_mp(molblock_list):
    with mp.Pool(None) as workers:
        res = list(workers.imap(druglike_filter, molblock_list))
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        required=True,
        type=str,
        help="The path to the input CSV file containing molecular conformations. Only .csv format is supported."
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
        "--out_csv",
        required=True,
        type=str,
        help="The path where the output CSV file with processed data will be saved."
    )
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)
    molblock_list = df[args.sdf_column].tolist()
    
    res = druglike_filter_mp(molblock_list)

    df['DrugLike_failure'] = res
    df = df[[args.id_column, 'DrugLike_failure']]
    df.to_csv(args.out_csv, index=False)