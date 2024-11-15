from .rdkit_utils import *
from rdkit import Chem
import copy


def get_half_basic_fragment_ids(mol, half_quad):
    """
    half_quad: a tuple of 2 integers, (x, y), x - the atom id of incoming terminal atom, y - the atom id of rotatable bond out terminal atom
    """
    res = [half_quad[0], half_quad[1]]
    [res.append(nei.GetIdx()) for id in [half_quad[0], half_quad[1]] for nei in mol.GetAtomWithIdx(id).GetNeighbors() if nei.GetIdx() not in half_quad and nei.GetAtomicNum() != 1 and nei.GetAtomicNum() != 0]
    return res

def include_ring_functional_groups(mol_object, frag_at_ids):
    # include ring
    res = copy.deepcopy(frag_at_ids)
    [res.extend(r) for id in frag_at_ids for r in mol_object.ringsys if id in r]
    res = set(res)

    # include functional groups
    seen_ids = []
    # functional_groups_ids = set([f for k, v in functional_pats.items() for p in v for f in mol.GetSubstructMatches(p)])
    [seen_ids.extend(f) for id in res for f in mol_object.functional_groups if id in f]
    seen_ids = set(seen_ids)

    # include the further ring if one atom in the functional group is also a part of ring system
    seen_ids2 = []
    [seen_ids2.extend(r) for id in seen_ids for r in mol_object.ringsys if id in r]
    seen_ids2 = set(seen_ids2)
    seen_ids = seen_ids|seen_ids2
    return list(res|seen_ids)

def get_st_at_id_in_ring(mol_object, basic_at_ids, oppo_rot_at_id):
    ring_id_ats_dict = dict()
    for id in basic_at_ids:
        if id in mol_object.ring_dict:
            if mol_object.ring_dict[id] not in ring_id_ats_dict:
                ring_id_ats_dict[mol_object.ring_dict[id]] = []
            ring_id_ats_dict[mol_object.ring_dict[id]].append(id)

    ring_in_basic_frag_st_id = set()

    for k, v in ring_id_ats_dict.items():
        _len_path_pairs = [(int(mol_object.dist_matrix[oppo_rot_at_id][id]), id) for id in v]
        ring_in_basic_frag_st_id.add(min(_len_path_pairs, key=lambda x: x[1])[1])
    return list(ring_in_basic_frag_st_id)

def get_ortho_ids(mol, ring_in_basic_frag_st_id, ringsys):
    """
    Get the atom indices at ortho location for all atoms in ring_in_basic_frag_st_id
    """
    ortho_ids, ortho_ringsys_ids = [], []
    for id in ring_in_basic_frag_st_id:
        for ringsys_ids in ringsys:
            if id in ringsys_ids:
                neis = mol.GetAtomWithIdx(id).GetNeighbors()
                for nei in neis:
                    if nei.GetIdx() in ringsys_ids:
                        ortho_ids.append(nei.GetIdx())
                        ortho_ringsys_ids.append(ringsys_ids)
    return ortho_ids, ortho_ringsys_ids

def find_ortho_at_ids(mol_object, half_quartet, basic_at_ids, ringsys=None):
    """
    Return 
    1. the indices of atoms at ortho positions
    2. their corresponding atom indices of the rings they belong to 
    according to the half quartet and basic fragment atom ids.
    """
    # Create a dictionary with atom id as keys and ring system index as values'
    ring_in_basic_frag_st_id = get_st_at_id_in_ring(mol_object, basic_at_ids, oppo_rot_at_id=half_quartet[0])

    # Get connection atom indices of substitutes on ring systems
    ortho_ids, ortho_ringsys_ids = get_ortho_ids(mol_object.raw_mol, ring_in_basic_frag_st_id, ringsys)
    return ortho_ids, ortho_ringsys_ids

def update_fragment_at_ids_by_ortho(mol_object, ortho_ids, ortho_ringsys_ids, frag_at_ids):
    """
    Use after ortho atom ids determined in advance
    """
    # res = copy.deepcopy(frag_at_ids)
    # Get substitutes bond atom pairs
    bd_at_pairs = [(id, nei.GetIdx()) for i, id in enumerate(ortho_ids) for nei in mol_object.raw_mol.GetAtomWithIdx(id).GetNeighbors() if nei.GetIdx() not in ortho_ringsys_ids[i] and nei.GetAtomicNum() != 1]
    # if no substitutes on ortho-positions, return the input fragment atom ids
    if len(bd_at_pairs) == 0:
        return frag_at_ids
    
    # Include the first atom in substitute
    sub_at_ids = [x[1] for x in bd_at_pairs]
    sub_at_ids = include_ring_functional_groups(mol_object, sub_at_ids)
    frag_at_ids.extend(sub_at_ids)
    return frag_at_ids

def cap_heteroat_ids(mol, cap_at_ids, cap_atomic_num=[7, 8, 16], cap_carbonyl=True):
    # cap_at_ids = set(cap_at_ids)
    cap_atomic_num = set(cap_atomic_num)

    res = []
    for id in cap_at_ids:
        at = mol.GetAtomWithIdx(id)
        at_num = at.GetAtomicNum()
        if at_num in cap_atomic_num:
            res.append(id)
        if cap_carbonyl:
            if at_num == 6:
                neis = at.GetNeighbors()
                for nei in neis:
                    if nei.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(nei.GetIdx(), id).GetBondType() == Chem.BondType.DOUBLE:
                        res.append(id)
    return res

def get_afids_from_at_ids(mol_object, in_frag_at_ids, cap_heteroat=True):
    frag_at_ids = in_frag_at_ids
    res = dict()
    for id in in_frag_at_ids:
        frag_at_ids.extend(mol_object.at_neighbors_h_dummy[id])
    res['MTF_at_ids'] = list(set(frag_at_ids))

    # cap_ats: the indices of atoms in bd_at_pairs which are included in frag_at_ids as well
    if cap_heteroat:
        cap_at_ids = [id for id in frag_at_ids if not mol_object.at_neighbors_ids[id].issubset(set(frag_at_ids))]
        
        heteroat_ids_in_mol = cap_heteroat_ids(mol_object.raw_mol, cap_at_ids)
        res['cap_heteroat_ids'] = heteroat_ids_in_mol
    return res
