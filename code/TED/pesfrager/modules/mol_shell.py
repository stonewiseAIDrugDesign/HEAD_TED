from typing import Any
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
import numpy as np
from .rdkit_utils import *
from ..utils.basic_mol_shell import *

functional_groups = {
    "hydrazine": ["[NX3;!R][NX3;!R]"],
    "hydrazone": ["[NX3;!R][NX2;!R]"],
    "nitric oxide": ["[N;!R]-[O;!R]"],
    "amide": ["[#7;!R][#6;!R](=[#8])", "[#7;!R][#6;!R](-[O-;!R])"],
    "urea": ["[NX3;!R][CX3;!R](=[OX1])[NX3;!R]"],
    # "aldehyde": ["[CX3H1](=O)[#6]"],
    "carbonyl": ["[#6;!R](=O)"],
    "sulfoxide": ["[#16X3;!R]=[OX1;!R]", "[#16X3+;!R][OX1-;!R]"],
    "sulfonyl": ["[#16X4;!R](=[OX1])(=[OX1])"],
    "sulfonyl aromatic": ["[#16;R](-[!#6])(-[!#6])"],
    "sulfinic acid": ["[#16X3;!R](=[OX1])[OX2H,OX1H0-;!R]"],
    "sulfinamide": ["[#16X4;!R](=[OX1])(=[OX1])([NX3R0;!R])"],
    "sulfonic acid": ["[#16X4;!R](=[OX1])(=[OX1])[OX2H,OX1H0-;!R]"],
    "phosphine oxide": ["[PX4;!R](=[OX1])([#6;!R])([#6;!R])([#6;!R])"],
    "phosphonate": ["[P;!R](=[OX1])([OX2H,OX1-;!R])([OX2H,OX1-;!R])"],
    "phosphate": ["[PX4;!R](=[OX1])([#8;!R])([#8;!R])([#8;!R])"],
    "carboxylic acid": ["[CX3;!R](=O)[OX1H0-,OX2H1;!R]"],
    "nitro": ["[NX3+;!R](=O)[O-;!R]", "[NX3;!R](=O)=O"],
    "ester": ["[CX3;!R](=O)[OX2H0;!R]"],
    "tri-halides": ["[#6;!R]([F,Cl,I,Br])([F,Cl,I,Br])([F,Cl,I,Br])"]
}
## New rules added
## 1. When the terminal atom is a carbonyl, it should be capped by a methyl group.
# q_carbonyl_match = rdqueries.AtomNumEqualsQueryAtom(6)
# q_carbonyl_match.ExpandQuery(rdqueries.AtomNumEqualsQueryAtom(8), # Oxygen atom
#               Chem.BondType.DOUBLE, # Double bond
#               False) # Not in same ring
## 2. Keep the first atom in all ortho substitutents relative to the torsion quartet atoms, 
## if the atom is part of a ring or functional groups defined above, include all the atoms instead.
rigid_groups = {
    "amide": ["[#7;!R][#6;!R](=[#8])", "[#7;!R][#6;!R](-[O-;!R])"],
    "sulfinamide": ["[#16X4;!R](=[OX1])(=[OX1])([NX3R0;!R])"],
    "sulfonic acid": ["[#16X4;!R](=[OX1])(=[OX1])[OX2H,OX1H0-;!R]"],
    "sulfinic acid": ["[#16X3;!R](=[OX1])[OX2H,OX1H0-;!R]"],
    "sulfonyl": ["[#16X4;!R](=[OX1])(=[OX1])"],
    "sulfonyl aromatic": ["[#16;R](-[!#6])(-[!#6])"],
    "tri-halides": ["[#6;!R]([F,Cl,I,Br])([F,Cl,I,Br])([F,Cl,I,Br])"],
    "nitro": ["[NX3+;!R](=O)[O-;!R]", "[NX3;!R](=O)=O"]
}
functional_pats = {k: [Chem.MolFromSmarts(x) for x in v] for k, v in functional_groups.items()}
for k, v in functional_pats.items():
    for p in v:
        if p is None:
            print(k)

rigid_pats = {k: [Chem.MolFromSmarts(x) for x in v] for k, v in rigid_groups.items()}
for k, v in functional_pats.items():
    for p in v:
        if p is None:
            print(k)

class MTFObject():
    def __init__(self, mol_object, quad) -> None:
        self.mol_object = mol_object
        self.ref_quartet = quad
        self.MTF_at_ids = []
        self.cap_heteroat_ids = []
        self.quartet = None

    # def add_fragment_info(self, MTF_at_ids, cap_heteroat_ids=None):
    #     self.MTF_at_ids.extend(MTF_at_ids)
    #     if cap_heteroat_ids is not None:
    #         self.cap_heteroat_ids.extend(cap_heteroat_ids)
    #     if all([id in self.MTF_at_ids for id in self.ref_quartet]):
    #         self.quartet = [self.MTF_at_ids.index(id) for id in self.ref_quartet]

class MolObject(BaseMolObject):
    def __init__(self, mol) -> None:
        super().__init__(mol)
        self.ringsys = GetRingSystems(mol)
        # at_id belongs to which ring
        self.ring_dict = create_ring_dict(self.ringsys)
        # ring_id contains which id
        self.swap_ring_dict = swap_dict(self.ring_dict)
        self.__get_shortest_path_mat()
        self.__get_functional_group_ids()
        self.__get_at_neighbors_ids()
        self.__get_at_neighbors_h_dummy()

    def __get_shortest_path_mat(self):
        self.dist_matrix = rdmolops.GetDistanceMatrix(self.raw_mol)

    def __get_functional_group_ids(self):
        self.functional_groups = [f for k, v in functional_pats.items() for p in v for f in self.raw_mol.GetSubstructMatches(p)]

    def __get_at_neighbors_ids(self):
        self.at_neighbors_ids = [set([nei.GetIdx() for nei in at.GetNeighbors()]) for at in self.raw_mol.GetAtoms()]
    
    def __get_at_neighbors_h_dummy(self):
        self.at_neighbors_h_dummy = [[nei.GetIdx() for nei in at.GetNeighbors() if nei.GetAtomicNum() == 1 or 0] for at in self.raw_mol.GetAtoms()]
