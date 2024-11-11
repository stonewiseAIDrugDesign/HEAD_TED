"""High-energy atom detection (HEAD) class for physically implausiable molecule conformation(s) check"""

import os
import argparse
import logging
import warnings

import torch
import torchani
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
from typing import Optional, Union

from scipy.spatial import distance_matrix
from torchani.temp_var import Temp
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import (
    MolFromMol2File,
    MolFromMolBlock,
    MolFromMolFile,
    MolFromPDBFile,
    MolFromSmiles,
    SDMolSupplier,
)

warnings.filterwarnings(action="ignore")
logger = logging.getLogger(__name__)
# log settings
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.setLevel("INFO")
logger.addHandler(stream_handler)

class HEAD:
    """Class to run HEAD for input molecule conformation(s)"""
    
    def __init__(self,
                 file_path: str,
                 csv_column: Optional[str]=None, 
                 add_Hs=False,
                 remove_Hs=False,
                 sanitize=False,
                 gpu: int=0): 
        """ init 
        Args:
            use_info_entropy: whether to use information entropy 
        
        Notes:
            - 
        
        """
        
        self.file_path =  file_path
        # init model
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        self.model, self.consts = self.load_ani2x_model()
        
        # init molecules loading
        self.molecules = self.load_molecules_from_file(
            self.file_path, 
            sanitize=sanitize,
            csv_column=csv_column,
            add_Hs=add_Hs,
            remove_Hs=remove_Hs
        )

        self.num_mols = len(self.molecules)
        # init head
        self._initialize_head()
        
        # 
        pass
    
    def _initialize_head(self) -> None:
         # init
        self.energies = []
        self.records_indices = np.zeros(self.num_mols) # 0: valid mol, 1: invalid mol, -1: unsupport mol
        self.invalid = []

        self.scores = np.zeros(self.num_mols)
        self.all_atom_types = []
        self.information_entropy = np.zeros(self.num_mols)
        self.information_entropy_invalid_ids = np.zeros(self.num_mols)
        
        self.hc = 0.8 # information entropy cutoff
        self.info_entropy_energy_threshold = 60 # unit: kcal/mol
    
    def load_ani2x_model(self):
        """pre-load ANI-2x model for atomic energies predictions"""
        try:
            path = os.path.dirname(os.path.realpath(__file__))
        except NameError:
            path = os.getcwd()
        
        # Load ANI-2x model
        const_file = os.path.join(path, "torchani/resources/ani-2x_8x/rHCNOSFCl-5.1R_16-3.5A_a8-4.params")
        consts = torchani.neurochem.Constants(const_file)
        model = torchani.models.ANI2x(periodic_table_index=False, model_index=None).to(self.device)
        logger.info(f"Device: {self.device}")
        logger.info("Loading ANI-2x model done.")
        
        return model, consts
    
    def atomic_enegry_classifier(
        self,
        species, 
        atomic_energies, 
        pos,
        use_info_entropy, # whether or not use inforamtion entropy for detecting invalid region
        ):

        pos = np.array(pos)
        # Description: 
        # HEAD works largely based on the "atomic-energy partition" principle. where "atomic-energy partition" principle, 
        # i.e., E_tot = \sum_{i} E_i, saying that: (i) each E_i is independent of others, it only depends on its receptive field; 
        # (ii) Any physically implausible region (with its receptive field) of a given conformation results in a high energy response. 
        # Thus, HEAD detects these high energies of different regions by seven pre-defined elemental enegry thresholds, 
        # the obtain of these energy thresholds are detailded in paper
        
        # hyperparameter (unit: kcal/mol): defines the atomic energy lower bound of finding the largest "invalid region"
        h_criteria = 10.
        
        # init of information entropy
        h=0 
        
        # init of largest subcomponent graph index based on real-space atom-pair distances
        subcomponent_index = None
        
        # information entropy judgement is a supplementary of HEAD, which detect "additional" invalid region based on information entropy of the largest region
        if use_info_entropy:
            rc = 2. # cut-off radius
            # temp init.
            max_sub_energies = 0
            index = np.where(atomic_energies > h_criteria)[0]
            
            if len(atomic_energies[index]) > 1: # subgraph contains at least two atom.
                # pairwise distance
                dist = distance_matrix(pos[index], pos[index]) 
                for p_dist in dist: 
                    # each subregion index
                    cur_sub_index = np.where(p_dist < rc)[0] 
                    if np.any(atomic_energies[index[cur_sub_index]]):
                        if atomic_energies[index[cur_sub_index]].sum() > max_sub_energies: 
                            # select index of the greatest sub-region enegry
                            max_sub_energies = atomic_energies[index[cur_sub_index]].sum()
                            # record index
                            subcomponent_index = index[cur_sub_index] #
            
                if subcomponent_index is not None and len(subcomponent_index) > 1:
                    # compute information entropy
                    p = (atomic_energies[subcomponent_index] - h_criteria)/ (atomic_energies[subcomponent_index] - h_criteria).sum()
                    h = p * np.log(p)
                    h = -1 * h.sum()
                    # normalize
                    h = h / np.log(len(p))
                else:
                    h = 0
            else:
                h=0
        
        # elemental energy thresholds
        elemental_thresholds = {"C": 32.81, "H": 34.06, "O": 59.861, "N": 37.277, "Cl": 84.893, "F": 81.9, "S": 248.151}
        
        unstable_idx = []
        unstable_energies = []
        scores=[]
        
        for i in range(len(species)):
            if atomic_energies[i] > elemental_thresholds[species[i]]:
                unstable_idx.append(i)
                unstable_energies.append(atomic_energies[i])
                scores.append(atomic_energies[i]-elemental_thresholds[species[i]])
        
        # Compute High-Energy Score (HES)
        # Method 1. R.M.S (E_i - E_{i, Z_i}) 
        # high_energy_score = np.sqrt(np.mean(np.array(scores)**2)) if len(scores) != 0 else 0
        
        # Method 2. MAE
        # high_energy_score = np.mean(np.abs(np.array(scores))) if len(scores) != 0 else 0
        
        # Method 3. SUM
        high_energy_score = np.sum(np.abs(np.array(scores))) if len(scores) != 0 else 0
        
        return np.array(unstable_idx), np.array(unstable_energies), high_energy_score, h, subcomponent_index
        
        
    def run(self, use_info_entropy=True):
        self.start_time = time()
        if use_info_entropy:
            logger.info(
                "Use information entropy method as a supplementary for the detection of invalid molecule conformations."
            )
            
        for idx, mol in enumerate(self.molecules):
            if mol is None:
                # consider as unsupported molecules
                self.records_indices[idx] = -1 # unsupport mol status stored as -1
                self.energies.append(None)
                self.invalid.append(None)
                self.all_atom_types.append(None)
                continue
            
            conf = mol.GetConformer()
            atom_types = []
            positions = []
            
            for i, atom in enumerate(mol.GetAtoms()):
                atom_types.append(atom.GetSymbol())
                positions.append([conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z])

            Temp.SPECIES = np.array(atom_types)
            coordinates = torch.tensor(positions, requires_grad=True).unsqueeze(0).to(self.device).to(torch.float32)
            
            try:
                species = self.consts.species_to_tensor(atom_types).unsqueeze(0).to(self.device)
            except KeyError:
                # support: C, H, N, O, S, F, Cl
                # unsupport mol status as -1: that may contain unsupported species
                self.records_indices[idx] = -1
                self.energies.append(None)
                self.invalid.append(None)
                self.all_atom_types.append(np.array(atom_types))
                continue
        
            energy = self.model((species, coordinates)).energies
            atomic_energies = Temp.ATOMIC_ENERGIES
            
            atomic_energies = hartree_to_kcal_per_mol(atomic_energies)
            atomic_energies = atomic_energies.squeeze()
            
            self.energies.append(atomic_energies)
            self.all_atom_types.append(np.array(atom_types))
            
            (
                unstable_idx, unstable_energies, high_energy_score, h, subcomponent_index
            ) = self.atomic_enegry_classifier(
                atom_types, 
                atomic_energies, 
                positions, 
                use_info_entropy=use_info_entropy
            )
            
            if use_info_entropy: self.information_entropy[idx] = h

            if len(unstable_idx) != 0:
                # detected invalid atoms
                self.records_indices[idx] = 1 # invalid mol status considered as 1
                self.scores[idx] = high_energy_score
                
                # record the atomic invalidness
                temp_res = []
                for i in range(len(unstable_idx)):
                    temp_res.append((unstable_idx[i]+1, atom_types[unstable_idx[i]], round(unstable_energies[i], 3)))
                self.invalid.append(temp_res)
                
            elif h > self.hc and atomic_energies[subcomponent_index].sum() > self.info_entropy_energy_threshold:
                # detected invalid atoms according to the information entropy supplementary method
                self.records_indices[idx] = 1 
                self.scores[idx] = atomic_energies[subcomponent_index].sum()
                
                # record the id
                self.information_entropy_invalid_ids[idx] = 1 # information entropy supplementary staus stored as 1 
                
                # record the atomic invalidness
                temp_res = []
                for k in range(len(subcomponent_index)):
                    temp_res.append((subcomponent_index[k]+1, atom_types[subcomponent_index[k]], round(atomic_energies[subcomponent_index[k]], 3)))
                self.invalid.append(temp_res)
                
            else:
                # detected valid mol, the mol status considered as 0
                self.invalid.append(None)

        logger.info("Detecting invalid conformations completed.")
        logger.info("========================Detection Results==========================")
        logger.info(f"Total molecules: {self.num_mols}")
        logger.info(f"Unsupported molecules: {np.where(self.records_indices == -1)[0].shape[0]}")
        logger.info(f"Invalid molecules: {np.where(self.records_indices == 1)[0].shape[0]}")
        # logger.info(f"Information entropy supplementary count: {len(np.where(self.information_entropy_invalid_ids>0)[0])}")
        logger.info(f"Time cost: {round((time() - self.start_time ) / 60, 2)} min")
    
    def load_molecules_from_file(
        self,
        file_path: str,
        sanitize=False,
        remove_Hs=False,
        add_Hs=False,
        csv_column: Optional[str]=None):
        """Load molecule(s) from a file, support sdf, pdb or sdf strings stored in csv file"""
        file_extension = file_path.split(".")[-1].lower()

        if file_extension == "pdb":
            # Load PDB
            if add_Hs:
                mol = Chem.MolFromPDBFile(file_path, removeHs=True, sanitize=sanitize)
                mol = Chem.AddHs(mol, addCoords=True)
            else:
                mol = Chem.MolFromPDBFile(file_path, removeHs=remove_Hs, sanitize=sanitize)
            if mol is None:
                logger.warning(f"Failed to load molecule from {file_path}")   
            molecules = [mol]
        elif file_extension == "sdf":
            # Load SDF
            if add_Hs:
                suppl = Chem.SDMolSupplier(file_path, removeHs=True, sanitize=sanitize)
            else:
                suppl = Chem.SDMolSupplier(file_path, removeHs=remove_Hs, sanitize=sanitize)
            molecules = []
            for mol in suppl:
                if mol is not None:
                    if add_Hs:
                        mol = Chem.AddHs(mol, addCoords=True)
                molecules.append(mol)    
        elif file_extension == "csv":
            mol_df = pd.read_csv(file_path)
            molecule_strings = mol_df.loc[:, csv_column].to_list()
            molecules =  []
            for i in range(len(molecule_strings)):
                try:
                    mol = MolFromMolBlock(molecule_strings[i], removeHs=remove_Hs, sanitize=sanitize)
                    if add_Hs:
                        mol = Chem.AddHs(mol, addCoords=True)
                    
                    molecules.append(mol)
                except Exception:
                    molecules.append(None)
                    continue
                
        if len(molecules) == 0:
            logger.warning("No molcules loaded") # warning
        
        logger.info(f"Loaded {len(molecules)} molecules.")
        return molecules
        
    
    def write_report(self, output_csv="/HEAD_result.csv"):
        res=[]
        for i in range(self.num_mols):
            # [mol_id, invalidity, invalid atoms, atomic energies, hes, atom types, information entropy, self.information_entropy_invalid_ids[i]]
            res.append([   
                i, 
                int(self.records_indices[i]), 
                self.invalid[i], 
                self.energies[i], 
                self.scores[i], 
                self.all_atom_types[i], 
                self.information_entropy[i], 
                int(self.information_entropy_invalid_ids[i])
            ])
            
        head_res = pd.DataFrame(data=res, columns=["mol_id", "invalidity", "invalid atoms", "atomic energies", "HES", "atom types", "information entropy", "information entropy invalidity"])
        head_res.to_csv(output_csv, index=False)
        logger.info(f"Write report to {output_csv} completed.")
        logger.info("========================How to check the report====================")
        logger.info(f"invalidity: whether this conformation is valid or not (0: valid | 1: invalid | -1: unsupported ).")
        logger.info(f"invalid atoms: detailed atomic-level invalidness, if invalid, each entry represents (atom index, atom type, detected high atomic energy (unit: kcal/mol)), else None")
        logger.info(f"HES: a score descibing the invalidness, and the higher the score, the more invalid the conformation is.")
        logger.info(f"atom types: atom types.")
        logger.info(f"information entropy: the computed information entropy for the maximum subregion, this approach is ONLY a supplementrary for HEAD.")
        logger.info(f"information entropy invalidity: if 1, invalid conformation ONLY detected by information entropy approach, else 0.")
    
    
    def plot(self, index=0, save_fig=False, fig_path=None):
        if self.invalid[index] is not None:
            irrat_index = [irrat[0] for irrat in self.invalid[index]]
        else:
            irrat_index = []
            
        rat_index = []
        rat_e = []
        irrat_e = []
        xs = list(range(len(self.energies[index])))

        for i, atom in enumerate(self.all_atom_types[index]):
            if i+1 not in irrat_index:
                rat_index.append(i+1)
                rat_e.append(self.energies[index][i])
            else:
                irrat_e.append(self.energies[index][i])


        fig = plt.figure(figsize=(10, 5))
        labels = ['$'+atom+'_{'+str(i+1)+'}$' for i, atom in enumerate(self.all_atom_types[index])]
        plt.bar(np.array(rat_index)-1, rat_e, edgecolor='k', color='#90BEE0')
        plt.bar(np.array(irrat_index)-1, irrat_e, edgecolor='k', color='#E57B7F')
        
        plt.xticks(xs, labels, rotation=90)
        plt.ylabel('Atomic energy (kcal/mol)', fontsize=15)
        
        fig.tight_layout()
        plt.grid(which='major')
        plt.show()
        if save_fig:
            if fig_path is None:
                fig_path = f"HEAD_fig_{0}.png"
                
            plt.savefig(fig_path, dpi=300)
                
        
        
        
# convert unit
def ev_to_hartree(energy):
    return energy/27.211396641308  

def ev_to_kcal_per_mol(energy):
    return energy * 23.0621

def hartree_to_kcal_per_mol(energy):
    return energy * 627.509608

def test_case():
    # log settings
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.setLevel("INFO")
    logger.addHandler(stream_handler)
    
    head = HEAD(
        file_path="/home/jovyan/atomic_energy_metric/paper_data/model_comparison/GM4K_final.csv",
        csv_column="conformer_sdf",  
        gpu=1
    )
    
    head.run(use_info_entropy=True)
    head.write_report()
    
    
    
def main():
    pass

if __name__ == "__main__":
    test_case()