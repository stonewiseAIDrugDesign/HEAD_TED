from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np

class BaseMolObject():
    def __init__(self, mol) -> None:
        self.raw_mol = mol
        self.atomicNum_list = [at.GetAtomicNum() for at in mol.GetAtoms()]
        self.expHsNum_list = [at.GetNumExplicitHs() for at in mol.GetAtoms()]
        self.chrg_list = [at.GetFormalCharge() for at in mol.GetAtoms()]
        self.__mol2connect_mats()

    def __mol2connect_mats(self):
        self.connect_mat = np.zeros((self.raw_mol.GetNumAtoms(), self.raw_mol.GetNumAtoms()))    
        for bd in self.raw_mol.GetBonds():
            i = bd.GetBeginAtomIdx()
            j = bd.GetEndAtomIdx()
            bd_order = bd.GetBondTypeAsDouble()
            self.connect_mat[i, j] = bd_order
            self.connect_mat[j, i] = bd_order
        