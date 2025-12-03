from rdkit import Chem
import numpy as np
import copy

def find_bonds(mol):
    # smrts = "[!$(*#*)!D1]-,=;!@[!$(*#*)&!D1]"
    smrts = "[!$(*#*)!D1&$(*([!#1])[!#1])]-,=;!@[!$(*#*)&!D1&$(*([!#1])[!#1])]"
    pat = Chem.MolFromSmarts(smrts)
    return mol.GetSubstructMatches(pat)

def get_quartet(mol, rot_bd_at1_id, rot_bd_at2_id):
    res = []
    neis = mol.GetAtomWithIdx(rot_bd_at1_id).GetNeighbors()
    rot_bd_at1_nei_ids = [x.GetIdx() for x in neis if x.GetIdx() != rot_bd_at2_id and x.GetSymbol() != "H"]

    neis = mol.GetAtomWithIdx(rot_bd_at2_id).GetNeighbors()
    rot_bd_at2_nei_ids = [x.GetIdx() for x in neis if x.GetIdx() != rot_bd_at1_id and x.GetSymbol() != "H"]

    for i in rot_bd_at1_nei_ids:
        for j in rot_bd_at2_nei_ids:
            res.append([i, rot_bd_at1_id, rot_bd_at2_id, j])
    return res

def get_src_id(mol, id):
    neis_ids = [nei.GetIdx() for nei in mol.GetAtomWithIdx(id).GetNeighbors()]
    assert len(neis_ids) == 1
    return neis_ids[0]

def give_dummy_atom(mol, at_id):
    at = mol.GetAtomWithIdx(at_id)
    if at.GetAtomicNum() == 1:
        emol = Chem.RWMol(mol)
        emol.ReplaceAtom(at_id, Chem.Atom(0))
        mol = Chem.Mol(emol)
        for atom in mol.GetAtoms():
            atom.UpdatePropertyCache()
        return mol
    else:
        raise ValueError("Wrong Atom Index. Please use a hydrogen atom instead.")

# def GetRingSystems(mol, includeSpiro=True, includeExo=True):
#     """_summary_

#     Args:
#         mol (_type_): _description_
#         includeSpiro (bool, optional): _description_. Defaults to True.
#         includeExo (bool, optional): _description_. Defaults to True.

#     Returns:
#         _type_: _description_
#     """
#     ri = mol.GetRingInfo()
#     systems = []
#     for ring in ri.AtomRings():
#         ringAts = set(ring)
#         if includeExo:
#             # Loop over the ring atoms
#             for at in ring:
#                 # Get the atom object
#                 atom = mol.GetAtomWithIdx(at)
#                 # Loop over the bonds of the atom
#                 for bond in atom.GetBonds():
#                     # Check if the bond is a double bond
#                     if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
#                         # Get the other atom index
#                         oth_at = bond.GetOtherAtomIdx(at)
#                         # Check if the other atom is not in the current ring
#                         if oth_at not in ringAts:
#                             # Check if the other atom is in another ring
#                             if ri.NumAtomRings(oth_at) > 0:
#                                 # Loop over the other rings
#                                 for oth_ring in ri.AtomRings():
#                                     # Check if the other ring contains the other atom
#                                     if oth_at in oth_ring:
#                                         # Check if the other ring is connected to the current ring
#                                         if len(set(ring) & set(oth_ring)) > 0:
#                                             # Add the other atom to the ring atoms
#                                             ringAts.add(oth_at)
#                             else:
#                                 ringAts.add(oth_at)
#         nSystems = []
#         for system in systems:
#             nInCommon = len(ringAts.intersection(system))
#             if nInCommon and (includeSpiro or nInCommon>1):
#                 ringAts = ringAts.union(system)
#             else:
#                 nSystems.append(system)
#         nSystems.append(ringAts)
#         systems = nSystems
#     return systems

def GetRingSystems(mol, includeSpiro=True, includeExo=True, includeConjugated=True, includeHydrogens=False):
    connect_mat = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()))    
    for bd in mol.GetBonds():
        i = bd.GetBeginAtomIdx()
        j = bd.GetEndAtomIdx()
        bd_order = bd.GetBondTypeAsDouble()
        connect_mat[i, j] = bd_order
        connect_mat[j, i] = bd_order
    
    ri = mol.GetRingInfo().AtomRings()
    ri_connect_mat = np.zeros((len(ri), len(ri)))
    ri_adj_mat = np.zeros((len(ri), len(ri)))
    for i in range(len(ri) - 1):
        for j in range(1, len(ri)):
            if i != j:
                inter_num = len(set(ri[i]).intersection(set(ri[j])))
                ri_connect_mat[i][j] = inter_num
                ri_connect_mat[j][i] = inter_num
                for at_i in ri[i]:
                    for at_j in np.nonzero(connect_mat[at_i])[0].tolist():
                        if at_j in ri[j]:
                            bd_order = mol.GetBondBetweenAtoms(at_i, at_j).GetBondTypeAsDouble()
                            ri_adj_mat[i][j] = bd_order
                            ri_adj_mat[j][i] = bd_order
                # for at_i in ri[i]:
                #     print(np.nonzero(ri_adj_mat[at_i]))
    # triu_ri_connect_mat = np.triu(ri_connect_mat, k=1)

    n =  len(ri_connect_mat)
    visited = [False] * n
    systems = []
    def dfs(node, current_system):
        visited[node] = True
        current_system.append(node)
        for neighbor in range(n):
            if includeSpiro:
                threshold = 0
            else:
                threshold = 1
            if includeConjugated:
                if (ri_connect_mat[node][neighbor] > threshold or ri_adj_mat[node][neighbor] > 1.0) and not visited[neighbor]:
                    dfs(neighbor, current_system)
            else:
                if ri_connect_mat[node][neighbor] > threshold and not visited[neighbor]:
                    dfs(neighbor, current_system)
        
    for i in range(n):
        if not visited[i]:
            system = []
            dfs(i, system)
            systems.append(system)
    systems_ats = []
    for system in systems:
        system_ats = []
        for ri_id in system:
            system_ats.extend(list(ri[ri_id]))
        systems_ats.append(system_ats)

    if includeExo:
        _systems_ats = []
        for system_ats in systems_ats:
            _system_ats = system_ats
            for at_i in _system_ats:
                for at_j in np.nonzero(connect_mat[at_i])[0].tolist():
                    if connect_mat[at_i][at_j] > 1 and at_j not in system_ats:
                        _system_ats.append(at_j)
            _systems_ats.append(_system_ats)
        systems_ats = _systems_ats

    if includeHydrogens:
        _systems_ats = []
        for system_ats in systems_ats:
            _system_ats = system_ats
            for at_i in _system_ats:
                _system_ats.extend([nei.GetIdx() for nei in mol.GetAtomWithIdx(at_i).GetNeighbors() if nei.GetAtomicNum() == 1])
            _systems_ats.append(_system_ats)
        systems_ats = _systems_ats
    systems_ats = [list(set(x)) for x in systems_ats]
    return systems_ats

def create_ring_dict(ringsys):
    ring_dict = dict()
    for i, ringsys_ids in enumerate(ringsys):
        for id in ringsys_ids:
            ring_dict[id] = i
    return ring_dict

def swap_dict(in_dict):
    out_dict = dict()
    for k, v in in_dict.items():
        if isinstance(v, list):
            for i in v:
                if i not in out_dict:
                    out_dict[i] = set()
                out_dict[i].add(k)
        if v not in out_dict:
            out_dict[v] = set()
        out_dict[v].add(k)
    return out_dict

def find_bonds_between_frag_mol(mol, frag_at_ids):
    # (x, y): x, the index of atom in fragment. y, the index of atom outside fragment
    frag_at_ids = set(frag_at_ids)
    return [(id, nei.GetIdx()) for id in frag_at_ids for nei in mol.GetAtomWithIdx(id).GetNeighbors() if nei.GetIdx() not in frag_at_ids]

def fix_radical_electron(mol):
    flag = True
    i = 0
    while flag: 
        for at in mol.GetAtoms():
            if at.GetNumRadicalElectrons() > 0:
                num_radi = at.GetNumRadicalElectrons()
                num_Hs = at.GetTotalNumHs()
                at.SetNumRadicalElectrons(0)
                at.SetNumExplicitHs(num_radi + num_Hs)
        mol = Chem.RemoveHs(mol)
        flag = any([at.GetNumRadicalElectrons() > 0 for at in mol.GetAtoms()])
        i += 1
        if i > 20:
            break
    return mol

def generate_smikey(smi, quartet):
    mol = Chem.MolFromSmiles(smi)
    for i, at_id in enumerate(quartet):
        mol.GetAtomWithIdx(at_id).SetProp('molAtomMapNumber', str(100))
    return Chem.MolToSmiles(mol)

def read_molblock(molblock_string):
    mol = Chem.MolFromMolBlock(molblock_string, removeHs=False)
    if mol is None:
        return None
    molblock_lines = molblock_string.splitlines()
    props_dict = dict()
    flag = False
    for l in molblock_lines:
        l_items = l.split()
        if len(l_items) > 0:
            if flag:
                props_dict[prop_key] = l
                flag = False

            if l_items[0] == ">":
                prop_key = l_items[1][1:-1]
                flag = True
                
    for k, v in props_dict.items():
        mol.SetProp(k, v)

    return mol
