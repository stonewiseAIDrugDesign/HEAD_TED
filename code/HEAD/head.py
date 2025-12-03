"""High-energy atom detection (HEAD) class for physically implausiable molecule conformation(s) & ligand pose checking"""
import os
import argparse
import logging
import warnings
import tempfile

import torch
import sys
sys.path.append(os.path.dirname(__file__))
import torchani
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
import ase

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
from ase import build


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
                 ligands_path: str,
                 csv_column: Optional[str]=None, 
                 protein_path: Optional[str]=None,
                 not_cut_pocket: Optional[bool]=False,
                 binding_energy_threshold: Optional[float]=20,
                 add_Hs: Optional[bool]=False,
                 remove_Hs: Optional[bool]=False,
                 sanitize: Optional[bool]=False,
                 gpu: Optional[int]=0,
                 residue_cutoff: Optional[float]=20.0,
                 mace_model_path: Optional[str]= None,
                 ): 
        """ init. 
        Args:
            ligands_path: The path that contains input molecule conformations, support: .sdf, .pdb, .csv.
            csv_column: The column name that specifies location of sdf strings stored in csv file. This only works when using csv file as input.
            add_Hs: Add Hydrogen atoms by RDKit if necessary. Note, this function will remove the original Hydrogens by default.
            protein_path: The path for the protein structure that the input ligands bind to, support: .pdb format. Note that, the protein should contain full hydrogen atoms,
            not_cut_pocket: Do not cut the pocket from the protein (residues that <20 Angst as the pocket) by the first ligand provided. Setting this param to `True` will significantly slow down the checking speed.
            binding_energy_threshold: Criteria for pose checking: pose is considered as invalid if E^{bound}_{mol} - E^{isolated}_{mol} > binding_energy_threshold. 
            remove_Hs: Whether to remove Hydrogens when loading conformations.
            sanitize: Whether to use RDKit sanitiziation when loading conformations.
            gpu: Specify the GPU to run, default `cuda:0` if cuda is available.
            
        Notes: HEAD requires each input conformation with correct Hydrogen atoms provided. If the inputs do not contain Hydrogen atoms, simply set `add_Hs` to `True`.
        """
        # base settings
        self.ligands_path = ligands_path
        self.protein_path = protein_path
        self.not_cut_pocket = not_cut_pocket
        
        # init model
        self.device = f'cuda:{gpu}' if torch.cuda.is_available else 'cpu'
        # self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        if mace_model_path is not None:
            self.use_mace = True
            # Warning: the following elemental thresholds are roughly determined and only for demo
            self.thresholds = {'N': -60.7, 'C': -128.0, 'O': -50.0, 'H': -45.3}
            self.model = self.load_mace_model(mace_model_path, self.device)
        else:
            self.model, self.consts = self.load_ani2x_model()
        
        self.device = torch.device(self.device)
        # init molecules loading
        self.molecules = self.load_molecules_from_file(
            self.ligands_path, 
            sanitize=sanitize,
            csv_column=csv_column,
            add_Hs=add_Hs,
            remove_Hs=remove_Hs
        )
        self.num_mols = len(self.molecules)
        
        # init complex if provided
        self.pocket = None
        self.protein = None
        self.complexes = []
        self.all_ligand_indices = []
        self.check_pose = True if self.protein_path is not None else False
        self.residue_cutoff = 8.0 if mace_model_path is not None else residue_cutoff
            
        if self.check_pose:
            self.protein = self.prepare_protein(self.protein_path)
            assert self.protein is not None, "Failed to load the protein, please check the protein file."
            
            if not self.not_cut_pocket:
                for i in range(self.num_mols):
                    temp_complex, _, _ = self._prepare_single_complex(self.molecules[i], self.protein)
                    if temp_complex is not None:
                        try:
                            # cutting pocket from given protein
                            self.pocket, _ = self.cut_pocket(temp_complex, self.residue_cutoff)
                        except Exception:
                            continue
                        else:
                            logger.info(
                                f"Successfully prepared protein-ligand complex by using No. {i+1} ligand, this complex is used for cutting pocket."
                            )
                            break
            else:
                self.pocket = self.protein
                
            self.complexes, self.all_ligand_indices = self.prepare_complex(self.molecules, self.pocket)
        
        # init head
        # self._initialize_head()
        
    def _initialize_head(self) -> None:
        # init: store all necessary results for future analysis use
        self.energies = [] # predicted all atomic energies by ANI-2x: [[E_{atom1}^{system1}, E_{atom2}^{system1},... ], [E_{atom1}^{system2}, E_{atom2}^{system2},...], ...]
        self.records_indices = np.zeros(self.num_mols) # detected validity labels: 0: valid mol, 1: invalid mol, -1: unsupport mol
        self.invalid = [] # detected details for molecular conformation, e.g., (1, C, 100) indicating the 1st carbon is detected as invalid atom with atomic energy 100 kcal/mol
        self.scores = np.zeros(self.num_mols) # init
        self.all_atom_types = [] # store all atomic types, e.g., [['C', 'N', ...], ['O', 'C', 'H',...], ...]
        self.information_entropy = np.zeros(self.num_mols) # init
        self.information_entropy_invalid_ids = np.zeros(self.num_mols) # init for detected invalid indices by information entropy criteria
        self.hc = 0.8 # information entropy cutoff
        self.info_entropy_energy_threshold = 60 # unit: kcal/mol
        
        if self.check_pose:
            self.binding_energy_threshold = 20 # unit: kcal/mol, default: 10
            self.complex_energies = [] # store all atomic energies of each atom of the complex (both ligand atom and residue atom)
            self.records_ligand_pose_indices = np.zeros(self.num_mols) # init for detected pose validity labels: 0: valid mol, 1: invalid mol, -1: unsupport mol
            self.binding_energy = []
            self.records_ligand_indices = np.zeros(self.num_mols) # init for detected ligand validity labels: 0: valid mol, 1: invalid mol, -1: unsupport mol
            self.records_pocket_indices = np.zeros(self.num_mols) # init for detected pocket validity labels: 0: valid mol, 1: invalid mol, -1: unsupport mol
            self.invalid_ligand = [] # detected details for ligand invalidity
            self.invalid_pocket = [] # detected details for pocket invalidity. Q: Why do we have this? See paper for details.
            self.ligand_in_pocket_energies = [] # store all atomic energies of each atom of the bound molecule (i.e., ligand)
            self.all_ligand_atom_types = [] # store all atomic types in ligand, e.g., [['C', 'N', ...], ['O', 'C', 'H',...], ...]
            self.all_pocket_atom_types = [] # store all atomic types in pocket, e.g., [['C', 'N', ...], ['O', 'C', 'H',...], ...]
            self.ligand_scores = np.zeros(self.num_mols) # init for dectected ligand scores
            self.ligand_information_entropy = np.zeros(self.num_mols) # init
            self.ligand_information_entropy_invalid_ids = np.zeros(self.num_mols) # init
            self.pocket_energies = None
            
    def load_mace_model(self, mace_model_path, device):
        from mace.calculators import mace_off
        calc = mace_off(model=mace_model_path, device=device) # small, medium, large, or path to MACE-OFF24
        
        return calc
    
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
    
    def prepare_protein(self, protein_path):
        with open(protein_path, 'r') as pf:
            lines = pf.readlines()
        
        # clear the protein by removing non-standard residues
        temp_str = ""
        for i_l, l in enumerate(lines):
            if l.startswith("ATOM"):
                temp_str+=l       
        prepared_protein = Chem.MolFromPDBBlock(temp_str, removeHs=False, sanitize=False)
        
        return prepared_protein
    
    def _prepare_single_complex(self, ligand_mol, protein_mol):
        # prepare complex from given molecule and the protein
        try:
            cur_complex = Chem.CombineMols(ligand_mol, protein_mol)
            lig_indices = list(range(ligand_mol.GetNumAtoms()))
            protein_indices = list(range(ligand_mol.GetNumAtoms(), protein_mol.GetNumAtoms() + ligand_mol.GetNumAtoms()))
        except Exception as e:
            logger.warning(f"Error whenprocessing complex: {e}")
            cur_complex = None
            lig_indices = []
            protein_indices = []
            # raise  # Re-raise the exception for the caller to handle
        
        return cur_complex, lig_indices, protein_indices
    
    def cut_pocket(self, complex_mol, residue_cutoff=20.0):
        # For speed consideration, cut a pocket by considering all residues with 20 Angst from ligand.
        _out_pocket = "./pocket_by_head.pdb" # temporary file showing the cut pocket
        pdb_string = Chem.MolToPDBBlock(complex_mol)
        
        with tempfile.NamedTemporaryFile(mode='w+t', suffix='.pdb', delete=True) as temp_file:
            # Write PDB string to temporary file
            temp_file.write(pdb_string)
            temp_file.flush()  # Ensure data is written
            temp_file.seek(0)  # Go back to start
            # Load from temporary file (alternative approach)
            comp = mda.Universe(temp_file.name)
            
        pkt = comp.select_atoms(f"protein and around {residue_cutoff} resname UNL")
        logger.info(f"Cut pocket with {residue_cutoff} Angst residues from given protein.")
        pkt_sel = ''
        for res in pkt.residues:
            # print(f"Residue: {res.resname} {res.resid}, Chain: {res.segid}")
            if res == pkt.residues[0]:
                pkt_sel = pkt_sel + f"(resid {res.resid} and segid {res.segid})"
            else:
                pkt_sel = pkt_sel + f" or (resid {res.resid} and segid {res.segid})"
        cut_pkt = comp.select_atoms(pkt_sel)
        cut_pkt.write(_out_pocket)
        pocket = Chem.MolFromPDBFile(_out_pocket, sanitize=False, removeHs=False)
        
        return pocket, _out_pocket
    
    def prepare_complex(self, molecules, pocket):
        # prepare complex from given molecule and the protein
        assert pocket is not None, "Failed to load the protein, please check the protein file."
        complexes = []
        all_lig_indices = []
        for i, mol in enumerate(molecules):
            if mol is not None:
                cur_complex, lig_indices, _ = self._prepare_single_complex(mol, pocket)
                complexes.append(cur_complex)
                all_lig_indices.append(len(lig_indices))
                
        return complexes, all_lig_indices
    
    def atomic_enegry_classifier(
        self,
        species, 
        atomic_energies, 
        pos,
        use_info_entropy, # whether or not use inforamtion entropy for detecting invalid region
        ):
        # Description: 
        # HEAD works largely based on the "atomic-energy partition" principle. where "atomic-energy partition" principle, 
        # i.e., E_tot = \sum_{i} E_i, saying that: (i) each E_i is independent of others, it only depends on its receptive field; 
        # (ii) Any physically implausible region (with its receptive field) of a given conformation results in a high energy response. 
        # Thus, HEAD detects these high energies of different regions by seven pre-defined elemental enegry thresholds, 
        # the obtain of these energy thresholds are detailded in paper
        
        pos = np.array(pos)
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
        
        # store temporary results
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
    
    def _run_single_case(self, mol, use_info_entropy=False):
        """
            input: mol, use_info_entropy
            output: valid, invalidity, atomic_energies, atom_types, score,
        """
        # init outputs
        invalidity = 0
        invalid_atoms = None
        high_energy_score = -1
        atomic_energies, atom_types = None, None
        information_entropy_invalid = 0
        information_entropy = 0
    
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
            energy = self.model((species, coordinates)).energies 
            atomic_energies = Temp.ATOMIC_ENERGIES
            atomic_energies = hartree_to_kcal_per_mol(atomic_energies)
            atomic_energies = atomic_energies.squeeze()
            
            (
                unstable_idx, unstable_energies, high_energy_score, information_entropy, subcomponent_index
            ) = self.atomic_enegry_classifier(
                atom_types, 
                atomic_energies, 
                positions, 
                use_info_entropy=use_info_entropy
            )
        except Exception:
            # support: C, H, N, O, S, F, Cl
            # unsupport mol status as -1: that may contain unsupported species
            invalidity = -1
            return invalidity, invalid_atoms, atomic_energies, high_energy_score, np.array(atom_types), information_entropy, information_entropy_invalid
        
        if len(unstable_idx) != 0:
            # detected invalid atoms
            invalidity = 1 # invalid mol status considered as 1
            # record the atomic invalidness
            invalid_atoms = []
            for i in range(len(unstable_idx)):
                invalid_atoms.append((unstable_idx[i]+1, atom_types[unstable_idx[i]], round(unstable_energies[i], 3)))
        elif information_entropy > self.hc and atomic_energies[subcomponent_index].sum() > self.info_entropy_energy_threshold:
            # detected invalid atoms according to the information entropy supplementary method
            invalidity = 1 
            high_energy_score = atomic_energies[subcomponent_index].sum()
            # record the id
            information_entropy_invalid = 1 # information entropy supplementary staus stored as 1 
            # record the atomic invalidness
            invalid_atoms = []
            for k in range(len(subcomponent_index)):
                invalid_atoms.append((subcomponent_index[k]+1, atom_types[subcomponent_index[k]], round(atomic_energies[subcomponent_index[k]], 3)))
        
        return (
                invalidity, 
                invalid_atoms, 
                atomic_energies, 
                high_energy_score, 
                np.array(atom_types), 
                information_entropy, 
                information_entropy_invalid
            )


    def run(self, use_info_entropy=True):
        """Run HEAD"""
        # init
        self._initialize_head()
        
        logger.info("==========Running HEAD for ligand conformations checking===========")
        self.start_time = time()
        if use_info_entropy:
            logger.info(
                "Use information entropy method as a supplementary for the detection of invalid molecule conformations."
            )
            
        for idx, mol in enumerate(self.molecules):
            if mol is None:
                # consider as unsupported molecules
                self.records_indices[idx] = -1 # unsupport mol status stored as -1
                self.invalid.append(None)
                self.energies.append(None)
                self.all_atom_types.append(None)
                self.scores[idx] = -1
                self.information_entropy[idx] = 0
                self.information_entropy_invalid_ids[idx] = 0
                continue
            
            (
                invalidity, 
                invalid_atoms, 
                atomic_energies, 
                high_energy_score, 
                atom_types, 
                information_entropy, 
                information_entropy_invalid
            ) = self._run_single_case(mol, use_info_entropy=use_info_entropy)
            
            self.records_indices[idx] = invalidity # unsupport mol status stored as -1
            self.invalid.append(invalid_atoms)
            self.energies.append(atomic_energies)
            self.all_atom_types.append(atom_types)
            self.scores[idx] = high_energy_score
            self.information_entropy[idx] = information_entropy
            self.information_entropy_invalid_ids[idx] = information_entropy_invalid
        
        logger.info("Detecting invalid conformations completed.")
        logger.info("========================Detection Results==========================")
        logger.info(f"Total molecules: {self.num_mols}")
        logger.info(f"Unsupported molecules: {np.where(self.records_indices == -1)[0].shape[0]}")
        logger.info(f"Invalid molecules: {np.where(self.records_indices == 1)[0].shape[0]}")
        # logger.info(f"Information entropy supplementary count: {len(np.where(self.information_entropy_invalid_ids>0)[0])}")
        logger.info(f"Time cost: {round((time() - self.start_time ) / 60, 2)} min")


    def run_pose_checking(self, use_info_entropy=False):
        """Run HEAD for pose checking"""
        # run pose checking will first run molecular conformation checking
        self.run(use_info_entropy=use_info_entropy)
        
        logger.info("================Running HEAD for pose checking=====================")
        start_time = time()

        for idx, mol in enumerate(self.complexes):
            if mol is None:
                # consider as unsupported molecules
                self.records_ligand_indices[idx] = -1
                self.records_pocket_indices[idx] = -1
                self.complex_energies.append(None)
                self.ligand_in_pocket_energies.append(None)
                self.invalid_ligand.append(None)
                self.invalid_pocket.append(None)
                self.all_ligand_atom_types.append(None)
                self.all_pocket_atom_types.append(None)
                continue
            cur_lig_max_index = self.all_ligand_indices[idx]
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
                self.records_ligand_indices[idx] = -1
                self.records_pocket_indices[idx] = -1
                self.complex_energies.append(None)
                self.ligand_in_pocket_energies.append(None)
                self.invalid_ligand.append(None)
                self.invalid_pocket.append(None)
                self.all_ligand_atom_types.append(np.array(atom_types)[:cur_lig_max_index])
                self.all_pocket_atom_types.append(np.array(atom_types)[cur_lig_max_index:])
                continue
            
            # for energy
            energy = self.model((species, coordinates)).energies
            atomic_energies = Temp.ATOMIC_ENERGIES
            atomic_energies = hartree_to_kcal_per_mol(atomic_energies)
            atomic_energies = atomic_energies.squeeze()
            self.complex_energies.append(atomic_energies)
            lig_atomic_energies = atomic_energies[:cur_lig_max_index]
            pkt_atomic_energies = atomic_energies[cur_lig_max_index:]
            self.ligand_in_pocket_energies.append(lig_atomic_energies)

            # for element type
            atom_types = np.array(atom_types)
            lig_atom_types = atom_types[:cur_lig_max_index]
            pkt_atom_types = atom_types[cur_lig_max_index:]
            self.all_ligand_atom_types.append(lig_atom_types)
            self.all_pocket_atom_types.append(pkt_atom_types)
            
            # for positions
            positions = np.array(positions)
            lig_positions = positions[:cur_lig_max_index, :]
            pkt_positions = positions[cur_lig_max_index:, :]
            
            # run ligand validity classifier
            (
                lig_unstable_idx, lig_unstable_energies, lig_high_energy_score, lig_h, lig_subcomponent_index
            ) = self.atomic_enegry_classifier(
                lig_atom_types, 
                lig_atomic_energies, 
                lig_positions, 
                use_info_entropy=use_info_entropy,
            )
            if use_info_entropy: self.ligand_information_entropy[idx] = lig_h
            
            # run pocket validity classifier
            (
                pkt_unstable_idx, pkt_unstable_energies, pkt_high_energy_score, _, _
            ) = self.atomic_enegry_classifier(
                pkt_atom_types, 
                pkt_atomic_energies, 
                pkt_positions, 
                use_info_entropy=False, # When using head for pocket validity checking, information entropy is not supported. Switching to normal mode.
            )
            
            if len(lig_unstable_idx) != 0:
                # detected invalid atoms for ligand
                self.records_ligand_indices[idx] = 0 # default as 0
                self.ligand_scores[idx] = lig_high_energy_score
                
                # record the atomic invalidness
                temp_res = []
                for i in range(len(lig_unstable_idx)):
                    temp_res.append((lig_unstable_idx[i]+1, lig_atom_types[lig_unstable_idx[i]], round(lig_unstable_energies[i], 3)))
                    self.records_ligand_indices[idx] = 1 # invalid mol status considered as 1
                    
                if self.records_ligand_indices[idx] == 1:
                    self.invalid_ligand.append(temp_res)
                else:
                    self.invalid_ligand.append(None)
            elif lig_h > self.hc and lig_atomic_energies[lig_subcomponent_index].sum() > self.info_entropy_energy_threshold:
                # detected invalid atoms according to the information entropy supplementary method
                self.records_ligand_indices[idx] = 1 
                self.ligand_scores[idx] = lig_atomic_energies[lig_subcomponent_index].sum()
                
                # record the id
                self.ligand_information_entropy_invalid_ids[idx] = 1 # information entropy supplementary staus stored as 1 
                
                # record the atomic invalidness
                temp_res = []
                for k in range(len(lig_subcomponent_index)):
                    temp_res.append((lig_subcomponent_index[k]+1+cur_lig_max_index, pkt_atom_types[lig_subcomponent_index[k]], round(atomic_energies[lig_subcomponent_index[k]], 3)))
                self.invalid_ligand.append(temp_res)
            else:
                # detected valid mol, the mol status considered as 0
                self.invalid_ligand.append(None)
                
            if len(pkt_unstable_idx) != 0:
                # detected invalid atoms for ligand
                # self.records_ligand_indices[idx] = 0 # default as 0
                
                # record the atomic invalidness
                temp_res = []
                for i in range(len(pkt_unstable_idx)):
                    temp_res.append((pkt_unstable_idx[i]+1+cur_lig_max_index, pkt_atom_types[pkt_unstable_idx[i]], round(pkt_unstable_energies[i], 3)))
                    self.records_pocket_indices[idx] = 1 # invalid mol status considered as 1
                    
                if self.records_pocket_indices[idx] == 1:
                    self.invalid_pocket.append(temp_res)
                else:
                    self.invalid_pocket.append(None)
            else:
                # detected valid mol, the mol status considered as 0
                self.invalid_pocket.append(None)

        # check the pose validity based on above results
        self._pose_checking(use_info_entropy=use_info_entropy)
        
        logger.info("Detecting invalid poses completed.")
        logger.info("========================Detection Results==========================")
        logger.info(f"Total complexes: {self.num_mols}")
        logger.info(f"Unsupported complexes: {np.where(self.records_ligand_pose_indices == -1)[0].shape[0]}")
        logger.info(f"Invalid poses: {np.where(self.records_ligand_pose_indices == 1)[0].shape[0]}")
        # logger.info(f"Information entropy supplementary count: {len(np.where(self.information_entropy_invalid_ids>0)[0])}")
        logger.info(f"Time cost: {round((time() - start_time ) / 60, 2)} min")
    
    
    def _pose_checking(self, use_info_entropy=False):
        assert len(self.complexes) == self.num_mols, "Molecular conformation and ligand-pocket complex mismatch!"
        # compute protein pocket validity results
        (
            invalidity, 
            invalid_atoms, 
            atomic_energies, 
            high_energy_score, 
            atom_types, 
            information_entropy, 
            information_entropy_invalid
        ) = self._run_single_case(self.pocket, use_info_entropy=use_info_entropy)
        self.pocket_energies = atomic_energies
        
        for idx, mol in enumerate(self.complexes):
            if self.records_ligand_indices[idx] == -1:
                # unsupported complex conformation
                self.records_ligand_pose_indices[idx] = -1
                self.binding_energy.append(None)
            else:
                if self.invalid[idx] is None and self.invalid_ligand[idx] is not None:
                    # Case 1: molecular conformation is valid, while the ligand in protein pocket is detected as invalid, thus invalid pose
                    self.records_ligand_pose_indices[idx] = 1
                elif self.invalid[idx] is not None and self.invalid_ligand[idx] is not None:
                    # Case2: molecular conformation and the ligand in protein pocket are both detected as invalid, check whether there is a new invalid atom or not
                    invalid_d = {}
                    invalid_indices = []
                    for iv in self.invalid[idx]:
                        invalid_d[iv[0]]=0
                        invalid_indices.append(int(iv[0])-1)
                    
                    for iiv in self.invalid_ligand[idx]:
                        try:
                            invalid_d[iiv[0]]
                        except KeyError:
                            self.records_ligand_pose_indices[idx] = 1 # newly detected invalid atom caused by the pocket environment, thus invalid pose
                            break
                
                # if self.invalid[idx] is not None and self.invalid_ligand[idx] is None:
                #     # Case 3: detected corner cases
                #     pass

                # if self.ligand_in_pocket_energies[idx].sum() - self.energies[idx].sum() > (self.binding_energy_threshold / 2):
                self.binding_energy.append(self.complex_energies[idx].sum() - self.energies[idx].sum() - atomic_energies.sum())
                if  self.complex_energies[idx].sum() - self.energies[idx].sum() - atomic_energies.sum() > self.binding_energy_threshold: # E_{bind} = E(complex) - E(ligand) - E(pocket)
                    # Additional energy criteria: binding energy checking, molecule in bound state is energetically unfavorable due compared with isolated state 
                    self.records_ligand_pose_indices[idx] = 1
                    
                # Supplementary checking for ligand pose by checking protein pocket, from the pocket's view
                if invalid_atoms is None and self.invalid_pocket[idx] is not None:
                    self.records_ligand_pose_indices[idx] = 1
                elif invalid_atoms is not None and self.invalid_pocket[idx] is not None:
                    cur_lig_max_index = self.all_ligand_indices[idx]
                    invalid_d = {}
                    for iv in invalid_atoms:
                        invalid_d[iv[0]]=0
                            
                    for iiv in self.invalid_pocket[idx]:
                        try:
                            invalid_d[iiv[0]-cur_lig_max_index]
                        except KeyError:
                            self.records_ligand_pose_indices[idx] = 1
                            break

    def run_mace_toy_model(self)-> None:
        if not self.check_pose:
            self.run_mace_conf()
        else:
            self.run_mace_pose()
            
    def run_mace_pose(self):
        """Run HEAD for pose checking"""
        # run pose checking will first run molecular conformation checking
        self.run_mace_conf()
        calc = self.model
        complexes = self.complexes
        start_time = time()

        # pocket 
        pocket = rdkitMol2ASEAtoms(self.pocket)    
        pocket.calc = calc
        pocket_e_tot = pocket.get_potential_energy()
        # print("Inference:", round(time() - start, 3), "sec | total energy: ", e_tot)
        pocket_node_energies = pocket.calc.results['node_energy']
        pocket_node_energies = ev_to_kcal_per_mol(pocket_node_energies)
        pocket_atom_types = pocket.get_chemical_symbols()
        self.pocket_energies = pocket_node_energies
        
        for i, mol in enumerate(complexes):
            cur_lig_max_index = self.all_ligand_indices[i]
            complex = rdkitMol2ASEAtoms(mol)    
            complex.calc = calc
            complex_e_tot = complex.get_potential_energy()
            complex_node_energies = complex.calc.results['node_energy']
            complex_node_energies = ev_to_kcal_per_mol(complex_node_energies)
            complex_atom_types = complex.get_chemical_symbols()
            self.complex_energies.append(complex_node_energies)
            lig_atomic_energies = complex_node_energies[:cur_lig_max_index]
            # pkt_atomic_energies = complex_node_energies[cur_lig_max_index:]
            self.ligand_in_pocket_energies.append(lig_atomic_energies)
            try:
                complex = rdkitMol2ASEAtoms(mol)    
                complex.calc = calc
                complex_e_tot = complex.get_potential_energy()
                complex_node_energies = complex.calc.results['node_energy']
                complex_node_energies = ev_to_kcal_per_mol(complex_node_energies)
                complex_atom_types = complex.get_chemical_symbols()
                self.complex_energies.append(complex_node_energies)
                
                # self.all_atom_types.append(atom_types)
            except Exception:
                # unsupported case
                self.complex_energies.append(None)
                self.binding_energy.append(None)
                self.records_ligand_pose_indices[i] = -1
                self.ligand_in_pocket_energies.append(None)
                continue
            else:
                bind_e = complex_node_energies.sum() - self.energies[i].sum() - pocket_node_energies.sum()
                self.binding_energy.append(bind_e)
                if bind_e > 20:
                    self.records_ligand_pose_indices[i] = 1
                else:
                    self.records_ligand_pose_indices[i] = 0
        
        logger.info("Detecting invalid poses completed.")
        logger.info("========================Detection Results==========================")
        logger.info(f"Total complexes: {self.num_mols}")
        logger.info(f"Unsupported complexes: {np.where(self.records_ligand_pose_indices == -1)[0].shape[0]}")
        logger.info(f"Invalid poses: {np.where(self.records_ligand_pose_indices == 1)[0].shape[0]}")
        # logger.info(f"Information entropy supplementary count: {len(np.where(self.information_entropy_invalid_ids>0)[0])}")
        logger.info(f"Time cost: {round((time() - start_time ) / 60, 2)} min")    
    
    def run_mace_conf(self) -> None:
        # init
        self._initialize_head()
        start_time = time()
        
        calc = self.model
        molecules = self.molecules
        
        for i, mol in enumerate(molecules):
            try:
                atoms = rdkitMol2ASEAtoms(mol)    
                atoms.calc = calc
                e_tot = atoms.get_potential_energy()
                # print("Inference:", round(time() - start, 3), "sec | total energy: ", e_tot)
                node_energies = atoms.calc.results['node_energy']
                node_energies = ev_to_kcal_per_mol(node_energies)
                atom_types = atoms.get_chemical_symbols()
                self.energies.append(node_energies)
                self.all_atom_types.append(atom_types)
            except Exception:
                # unsupported case
                self.energies.append(None)
                self.records_indices[i]=-1
                self.invalid.append(None)
                self.scores[i]=None
                self.all_atom_types.append(None)
                continue
            else:
                cur_invalid_flag = 0
                cur_invalid = []
                cur_score = 0
                for j, ele in enumerate(atom_types):
                    try:
                        if node_energies[j] > self.thresholds[ele]:
                            cur_invalid.append(({j+1}, ele, node_energies[j]))
                            cur_invalid_flag = 1
                            cur_score +=  node_energies[j] - self.thresholds[ele]
                    except KeyError:
                        continue
                if cur_invalid_flag == 1:
                    self.invalid.append(cur_invalid)
                else:
                    self.invalid.append(None)
                    
                self.scores[i] = cur_score
                self.records_indices[i] = cur_invalid_flag
        print(f">>>Detecting invalid conformations completed.")
        print(">>>========================Detection Results==========================")
        print(f">>>Total molecules: {len(self.molecules)}")
        print(f">>>Unsupported molecules: {np.where(np.array(self.records_indices) == -1)[0].shape[0]}")
        print(f">>>Invalid molecules: {np.where(np.array(self.records_indices) == 1)[0].shape[0]}")
        print(f">>>Time cost: {round((time() - start_time ) / 60, 2)} min")
    
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
                        try:
                            mol = Chem.AddHs(mol, addCoords=True)
                        except RuntimeError:  # catch RuntimeError: Pre-condition Violation
                            molecules.append(None)
                            continue
                molecules.append(mol)    
        elif file_extension == "csv":
            # SDF string
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
        
    
    def write_report(self, output_csv="./HEAD_result.csv", check_pose=False):
        """Save HEAD results to csv file"""
        
        res=[]
        if not check_pose:
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
                
            head_res = pd.DataFrame(
                data=res, 
                columns=[
                    "mol_id", 
                    "ligand invalidity", 
                    "ligand invalid atoms", 
                    "ligand atomic energies", 
                    "ligand HES", 
                    "ligand atom types", 
                    "ligand information entropy", 
                    "ligand information entropy invalidity"
                ]
            )
            output_path = os.path.dirname(os.path.abspath(output_csv))
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
            head_res.to_csv(output_csv, index=False)
            logger.info(f"Write report to {output_csv} completed.")
            logger.info("========================How to check the report====================")
            logger.info(f"ligand invalidity: whether this conformation is valid or not (0: valid | 1: invalid | -1: unsupported ).")
            logger.info(f"ligand invalid atoms: detailed atomic-level invalidness, if invalid, each entry represents (atom index, atom type, detected high atomic energy (unit: kcal/mol)), else None")
            logger.info(f"ligand HES: a score descibing the invalidness, and the higher the score, the more invalid the conformation is.")
            logger.info(f"ligand atom types: atom types.")
            logger.info(f"ligand information entropy: the computed information entropy for the maximum subregion, this approach is ONLY a supplementrary for HEAD.")
            logger.info(f"ligand information entropy invalidity: if 1, invalid conformation ONLY detected by information entropy approach, else 0.")
        
        else:
            for i in range(self.num_mols):
                # [mol_id, invalidity, invalid atoms, atomic energies, hes, atom types, information entropy, self.information_entropy_invalid_ids[i]]
                try:
                    ligand_energy = self.ligand_in_pocket_energies[i].sum() 
                    mol_energy = self.energies[i].sum()
                    hes = self.scores[i]
                    ncie = self.binding_energy[i]
                except Exception:
                    ligand_energy = None
                    mol_energy = None
                    ncie = None
                    hes = None
                
                res.append([   
                    i, 
                    int(self.records_indices[i]), 
                    self.invalid[i], 
                    self.energies[i], 
                    self.scores[i], 
                    self.all_atom_types[i], 
                    self.information_entropy[i], 
                    int(self.information_entropy_invalid_ids[i]),
                    int(self.records_ligand_pose_indices[i]),
                    ligand_energy, 
                    mol_energy, 
                    ncie, 
                    hes,
                ])
                
            head_res = pd.DataFrame(
                data=res, 
                columns=[
                    "mol_id", 
                    "ligand invalidity", 
                    "ligand invalid atoms", 
                    "ligand atomic energies", 
                    "ligand HES", 
                    "ligand atom types", 
                    "ligand information entropy", 
                    "ligand information entropy invalidity",
                    "pose invalidity",
                    "bound ligand energy",
                    "isolated ligand energy",
                    "binding energy",
                    "pose HES",
                    ])
            output_path = os.path.dirname(os.path.abspath(output_csv))
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
            head_res.to_csv(output_csv, index=False)
            logger.info(f"Write report to {output_csv} completed.")
            logger.info("========================How to check the report====================")
            logger.info(f"ligand invalidity: whether this conformation is valid or not (0: valid | 1: invalid | -1: unsupported ).")
            logger.info(f"ligand invalid atoms: detailed atomic-level invalidness, if invalid, each entry represents (atom index, atom type, detected high atomic energy (unit: kcal/mol)), else None")
            logger.info(f"ligand HES: a score descibing the invalidness, and the higher the score, the more invalid the conformation is.")
            logger.info(f"ligand atom types: atom types.")
            logger.info(f"ligand information entropy: the computed information entropy for the maximum subregion, this approach is ONLY a supplementrary for HEAD.")
            logger.info(f"ligand information entropy invalidity: if 1, invalid conformation ONLY detected by information entropy approach, else 0.")
            logger.info(f"pose invalidity: whether this binding pose is valid or not (0: valid | 1: invalid | -1: unsupported ).")
            logger.info(f"bound ligand energy: predicted total energy of bound ligand (in kcal/mol).")
            logger.info(f"isolated ligand energy: predicted total energy of isolated ligand (in kcal/mol).")
            logger.info(f"pose HES: a score descibing the invalidness, and the higher the score, the more invalid the pose is.")
    
    def plot(self, index=0, save_fig=False, fig_path=None, check_pose=False):
        """Plot HEAD results at atomic-level"""
        if check_pose and not self.check_pose:
            logger.info(
                "No pose checking results detected."
                )
            return
        # elif not check_pose:
        #     invalid = self.invalid
        #     energies = self.energies
        #     all_atom_types = self.all_atom_types
        # else:
        #     invalid = self.invalid_ligand
        #     energies = self.ligand_in_pocket_energies
        #     all_atom_types = self.all_ligand_atom_types
        invalid = self.invalid
        energies = self.energies
        all_atom_types = self.all_atom_types
        
        if invalid[index] is not None:
            irrat_index = [irrat[0] for irrat in invalid[index]]
        else:
            irrat_index = []
            
        rat_index = []
        rat_e = []
        irrat_e = []
        xs = list(range(len(energies[index])))

        for i, atom in enumerate(all_atom_types[index]):
            if i+1 not in irrat_index:
                rat_index.append(i+1)
                rat_e.append(energies[index][i])
            else:
                irrat_e.append(energies[index][i])


        fig = plt.figure(figsize=(10, 5))
        labels = ['$'+atom+'_{'+str(i+1)+'}$' for i, atom in enumerate(all_atom_types[index])]
        plt.bar(np.array(rat_index)-1, rat_e, edgecolor='k', color='#90BEE0')
        plt.bar(np.array(irrat_index)-1, irrat_e, edgecolor='k', color='#E57B7F')
        
        plt.xticks(xs, labels, rotation=90)
        plt.ylabel('Atomic Energy (kcal/mol)', fontsize=15)
        
        fig.tight_layout()
        plt.grid(which='major')
        plt.show()
        if save_fig:
            if fig_path is None:
                fig_path = f"HEAD_fig_{index}.png"
                
            plt.savefig(fig_path, dpi=300)
            logger.info(f"The HEAD plot was saved to {fig_path}.")
                

    def plot_ligand_dE(self, index=0, save_fig=False, fig_path=None):
        """Plot HEAD results at atomic-level"""
        if not self.check_pose:
            logger.info(
                "No pose checking results detected."
                )
            return
        
        # invalid = self.invalid_ligand
        dEs = self.ligand_in_pocket_energies[index] - self.energies[index]
        all_atom_types = self.all_atom_types
        
        # if invalid[index] is not None:
        #     irrat_index = [irrat[0] for irrat in invalid[index]]
        # else:
        #     irrat_index = []
        
        irrat_index=np.where(np.array(dEs) > 4)[0]
        irrat_index +=1 # index starts from 1
        rat_index = []
        rat_e = []
        irrat_e = []
        xs = list(range(len(self.ligand_in_pocket_energies[index])))

        for i, atom in enumerate(all_atom_types[index]):
            if i+1 not in irrat_index:
                rat_index.append(i+1)
                rat_e.append(dEs[i])
            else:
                irrat_e.append(dEs[i])

        fig, ax = plt.subplots(figsize=(10, 5))
        labels = ['$'+atom+'_{'+str(i+1)+'}$' for i, atom in enumerate(all_atom_types[index])]
        ax.bar(np.array(rat_index)-1, rat_e, edgecolor='k', color='#90BEE0')
        ax.bar(np.array(irrat_index)-1, irrat_e, edgecolor='k', color='#E57B7F')
        ax.text(0.85, 0.94, '$E_{bind}$:'+f' {round(sum(dEs), 1)} kcal/mol', fontsize='16', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        ax.set_xticks(xs, labels, rotation=90)
        ax.set_ylabel('Atomic Energy Difference (kcal/mol)', fontsize=15)
        
        fig.tight_layout()
        # plt.grid(which='major')
        if save_fig:
            if fig_path is None:
                fig_path = f"HEAD_fig_pose_checking_{index}.png"
                
            plt.savefig(fig_path, dpi=300)
            logger.info(f"The HEAD plot was saved to {fig_path}.")
        plt.show()
        
        
# convert unit
def ev_to_hartree(energy):
    return energy/27.211396641308  

def ev_to_kcal_per_mol(energy):
    return energy * 23.0621

def hartree_to_kcal_per_mol(energy):
    return energy * 627.509608

def test_case():
    head = HEAD(
        ligands_path="/home/jovyan/atomic_energy_metric/paper_data/model_comparison/GM4K_final.csv",
        csv_column="conformer_sdf",  
        gpu=0
    )
    
    head.run(use_info_entropy=True)
    head.write_report()
    
def rdkitMol2ASEAtoms(rdkit_mol):
    # Ensure the molecule has a conformer (3D coordinates)
    conf = rdkit_mol.GetConformer()
    # Extract atomic numbers and positions
    atomic_numbers = [atom.GetAtomicNum() for atom in rdkit_mol.GetAtoms()]
    positions = [conf.GetAtomPosition(i) for i in range(rdkit_mol.GetNumAtoms())]
    # Convert positions to a list of [x, y, z] coordinates
    positions = [[pos.x, pos.y, pos.z] for pos in positions]
    # Create ASE Atoms object
    ase_atoms = ase.Atoms(numbers=atomic_numbers, positions=positions)
    return ase_atoms

def main(args):
    extention = args.ligands_path.split('.')[-1].lower()
    check_pose = True if args.protein_path is not None else False
    if args.protein_path is not None and args.not_cut_pocket:
        warnings.warn("Checking the pose by using full protein atoms will take a much longer time. Please set `not_cut_pocket` to False instead.", Warning)

    use_info_entropy = False if args.not_use_info_entropy else True
    
    mace_model_path = None
    if args.use_mace_off:
        logger.info("Using MACE-OFF24 for HEAD checking.")
        mace_model_path = args.mace_model_path

    if extention == 'csv' and args.csv_column is None:
            raise RuntimeError("Please provide csv column name that stores sdf string when using csv as an input.")
        
    head = HEAD(
                ligands_path=args.ligands_path,
                csv_column=args.csv_column,  
                protein_path=args.protein_path,
                not_cut_pocket=args.not_cut_pocket,
                gpu=args.gpu,
                add_Hs=args.add_Hs,
                remove_Hs=args.remove_Hs,
                sanitize=args.sanitize,
                mace_model_path=mace_model_path,
            )
             

    if args.use_mace_off:
        if check_pose:
            head.run_mace_pose()
        else:
            head.run_mace_conf()
    else:
        
        if check_pose :
            head.run_pose_checking(use_info_entropy=use_info_entropy)
        else:
            head.run(use_info_entropy=use_info_entropy)
    
    if args.write_report:
        head.write_report(output_csv=args.output_csv, check_pose=check_pose)
    
    if args.plot:
        if check_pose:
            head.plot_ligand_dE(
                index=args.plot_index,
                save_fig=True,
                fig_path=args.fig_save_to_path
            )
        else:
            head.plot(
                index=args.plot_index,
                save_fig=True,
                fig_path=args.fig_save_to_path
            )
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ligands_path",
        default=None,
        type=str,
        help="The path that contains input molecule conformations, support: .sdf, .pdb, .csv. When using csv file, the conformation should be stored as sdf string, and the column name should be provided.",
    )
    parser.add_argument(
        "--protein_path",
        default=None,
        type=str,
        help="The path for the protein structure that the input ligands bind to, support: .pdb format."
    )
    parser.add_argument(
        "--not_cut_pocket",
        action="store_true",
        help="Do not cut the pocket from the protein (residues that <6 Angst as the pocket) by the first ligand provided.",
    )
    parser.add_argument(
        "--csv_column",
        default=None,
        type=str,
        help="The column name that specifies location of sdf strings stored in csv file. This is only working when using csv file as input."
    ) 
    parser.add_argument(
        "--add_Hs",
        action="store_true",
        help="Add Hydrogen atoms by RDKit if necessary. Note, this function removes the original Hydrogens by default.",
    )
    parser.add_argument(
        "--remove_Hs",
        action="store_true",
        help="Whether to remove Hydrogens when loading conformations. **HEAD requires each input conformation with correct Hydrogen atoms provided.",
    )
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Whether to use RDKit sanitiziation when loading conformations.",
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="Specify the GPU to run, default cuda:0 if cuda is available.",
    )
    parser.add_argument(
        "--not_use_info_entropy",
        action="store_false",
        help="Whether to use information entropy rules as a supplementary for HEAD evaluation.",
    )
    parser.add_argument(
        "--write_report",
        action="store_true",
        help="Whether to use RDKit sanitiziation when loading conformations.",
    )
    parser.add_argument(
        "--output_csv",
        default="./HEAD_result.csv",
        type=str,
        help="The path for saving HEAD results."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot atomic-level HEAD result for given conformation index.",
    )
    parser.add_argument(
        "--plot_index",
        default=0,
        type=int,
        help="Which conformation to plot.",
    )
    parser.add_argument(
        "--fig_save_to_path",
        default=None,
        type=str,
        help="Save the plot to the given figure path.",
    )
    parser.add_argument(
        "--use_mace_off",
        action="store_true",
        help="Whether to use MACE-OFF24(M) for HEAD checking",
    )
    parser.add_argument(
        "--mace_model_path",
        default='medium',
        type=str,
        help="Path or type for MACE-OFF24",
    )
    args = parser.parse_args()
    main(args)