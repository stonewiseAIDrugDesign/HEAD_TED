from ..api.prepare_MTF import *
from ..api.quartet_wrapper import *

def conf_fragmenter(conf):
    if conf is None:
        return None
    try:
        mol = Chem.MolFromMolBlock(conf)
    except:
        mol = None
    if mol is None:
        return None
    quartets = enumerate_quartet(mol)
    mol_object = MolObject(mol)
    res = []
    for quad_group in quartets:
        failed_inner_res, good_inner_res = [], []
        for quad in quad_group:
            try:
                mtf_object = get_MTF_feature(mol_object, quad)
                # frag, _ = get_frag_from_at_ids(mol, in_frag_at_ids=mtf_feature['MTF_at_ids'], heteroat_ids_in_mol=mtf_feature['cap_atom_ids'])
                frag_smi, quartet, deg = get_frag_from_mtf_object(mtf_object)
                good_inner_res.append((generate_smikey(frag_smi, quartet), frag_smi, quartet, mtf_object.ref_quartet, deg))
            except:
                failed_inner_res.append((quad, None))
        if len(good_inner_res) > 0:
            good_inner_res = sorted(good_inner_res, key=lambda x: (len(x[1]), x[2][0], x[2][1], x[2][2], x[2][3]))
        inner_res = good_inner_res + failed_inner_res
        res.append(inner_res)
    return res

# def conf_single_batch(conf_list, i=0):
#     res = []
#     for conf in conf_list:
#         if conf is None or conf == '':
#             res.append(None)
#             continue
#         try:
#             mol = read_molblock(conf)
#         except:
#             mol = None
#         if mol is None:
#             res.append(None)
#             continue

#         tmp_output = None
#         try:
#             tmp_output = enumerate_MTFs(mol)
#         except Exception as e:
#             pass
#         if tmp_output is not None:
#             res.append(tmp_output)
#         else:
#             res.append(None)
#     return res, i
    