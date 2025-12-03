from ..modules import *

def enumerate_quartet(mol):
    rot_bd_at_pairs = find_bonds(mol)

    res = []
    for at_pair in rot_bd_at_pairs:
        quads = get_quartet(mol, at_pair[0], at_pair[1])
        res.append(quads)

    return res