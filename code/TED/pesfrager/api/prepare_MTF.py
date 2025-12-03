from ..modules import *
import numpy as np
from rdkit.Chem import rdchem, rdMolTransforms
import copy
import time

def get_hMTF_feature(mol_object, half_quad):
    basic_fragment_half_ids = get_half_basic_fragment_ids(mol_object.raw_mol, half_quad)
    fragment_ids = include_ring_functional_groups(mol_object, basic_fragment_half_ids)
    ortho_ids, ortho_ringsys_ids = find_ortho_at_ids(mol_object, half_quad, basic_fragment_half_ids, mol_object.ringsys)
    fragment_ids = update_fragment_at_ids_by_ortho(mol_object, ortho_ids, ortho_ringsys_ids, fragment_ids)
    return fragment_ids

def get_MTF_feature(mol_object, quad):
    half_quad_1 = quad[:2]
    half_quad_2 = quad[2:]

    # half features
    fragment_ids_1 = get_hMTF_feature(mol_object, half_quad_1)
    fragment_ids_2 = get_hMTF_feature(mol_object, half_quad_2)

    fragment_ids = list(set(fragment_ids_1) | set(fragment_ids_2))
    res = get_afids_from_at_ids(mol_object, fragment_ids, cap_heteroat=True)
    mtf_object = MTFObject(mol_object, quad)

    mtf_object.MTF_at_ids = res['MTF_at_ids']
    if 'cap_heteroat_ids' in res:
        mtf_object.cap_heteroat_ids = res['cap_heteroat_ids']
    mtf_object.quartet = [mtf_object.MTF_at_ids.index(id) for id in mtf_object.ref_quartet]
    return mtf_object

def get_frag_from_mtf_object(mtf_object, cap_heteroats=True):
    mol = Chem.RWMol()
    for id in mtf_object.MTF_at_ids:
        mol.AddAtom(Chem.Atom(mtf_object.mol_object.atomicNum_list[id]))
    
    # Rebuild Mol Object
    sub_connect_mat = mtf_object.mol_object.connect_mat[np.ix_(mtf_object.MTF_at_ids, mtf_object.MTF_at_ids)]
    rec_sub_connect_mat = np.triu(sub_connect_mat, k=1)
    for i, j in zip(*np.nonzero(rec_sub_connect_mat)):
        bond_order = sub_connect_mat[i][j]
        if bond_order == 1.0:
            bond_type = rdchem.BondType.SINGLE
        elif bond_order == 2.0:
            bond_type = rdchem.BondType.DOUBLE
        elif bond_order == 3.0:
            bond_type = rdchem.BondType.TRIPLE
        elif bond_order == 1.5:
            bond_type = rdchem.BondType.AROMATIC
        mol.AddBond(int(i), int(j), bond_type)

    # Add Methyl group
    if cap_heteroats:
        i = len(mtf_object.MTF_at_ids)
        for at_id in mtf_object.cap_heteroat_ids:
            if mtf_object.mol_object.atomicNum_list[at_id] != 6:
                mol.AddAtom(Chem.Atom(6))
                # Connect methyl group to fragment
                mol.AddBond(mtf_object.MTF_at_ids.index(at_id), i, rdchem.BondType.SINGLE)
                i += 1

    # Add Explicit Hydrogens
    for i, at_id in enumerate(mtf_object.MTF_at_ids):
        mol.GetAtomWithIdx(i).SetNumExplicitHs(mtf_object.mol_object.expHsNum_list[at_id])
        mol.GetAtomWithIdx(i).SetFormalCharge(mtf_object.mol_object.chrg_list[at_id])
    #TODO: Hydrogens
    smi = None
    # Convert to canonical SMIKEYs and return the mapping of atom idx
    mol_noH = Chem.RemoveHs(mol)
    smi = Chem.MolToSmiles(mol_noH)

    # Get dihedral deg if conformers are given
    confs = mtf_object.mol_object.raw_mol.GetConformers()
    deg = None
    if len(confs) > 0:
        deg = rdMolTransforms.GetDihedralDeg(confs[0], *mtf_object.ref_quartet)
    
    # Get Atom Idx Mapping between smiles Mol and input Mol
    smi_mol = Chem.MolFromSmiles(smi)
    at_id_mapping = list(mol.GetSubstructMatches(smi_mol)[0])

    new_quartet = [at_id_mapping.index(id) for id in mtf_object.quartet]
    return smi, new_quartet, deg