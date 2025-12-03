import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D


def get_torsion_oeatom_list(mol, tag="TORSION_ATOMS_FRAGMENT"):
    torsion_list = mol.GetProp(tag)
    try:
        torsion_atoms_idx = list(map(int, torsion_list.split('-')))
        return torsion_atoms_idx
    except Exception as e:
        print(e)
        return None


def deep_traversal(molecule, atom_index, visited_atoms):
    if atom_index in visited_atoms:
        return

    visited_atoms.add(atom_index)
    atom = molecule.GetAtomWithIdx(atom_index)
    for neighbor in atom.GetNeighbors():
        neighbor_index = neighbor.GetIdx()
        deep_traversal(molecule, neighbor_index, visited_atoms)


def OEGetSubtree(bgnAtom, endAtom, mol):
    traverse_start_id = endAtom.GetIdx()
    visited_atoms = set()
    visited_atoms.add(bgnAtom.GetIdx())
    deep_traversal(mol, traverse_start_id, visited_atoms)
    ret_list = []
    for temp_id in visited_atoms:
        if temp_id == bgnAtom.GetIdx() or temp_id == traverse_start_id:
            continue
        ret_list.append(mol.GetAtomWithIdx(temp_id))
    return ret_list


def bond_distance2(mol, atom_a_index, atom_b_index):
    coord_a = mol.GetConformer().GetAtomPosition(atom_a_index)
    coor_b = mol.GetConformer().GetAtomPosition(atom_b_index)
    ret_dist = (coord_a[0] - coor_b[0]) ** 2 + (coord_a[1] - coor_b[1]) ** 2 + (coord_a[2] - coor_b[2]) ** 2
    return ret_dist


def get_torsion_oebond(mol, tag="TORSION_ATOMS_FRAGMENT"):
    torsion_atoms = get_torsion_oeatom_list(mol, tag)
    try:
        return mol.GetBondBetweenAtoms(torsion_atoms[1], torsion_atoms[2])
    except Exception as e:
        print(e)
        return None


def GetPairwiseDistanceMatrix(icoords, jcoords):
    '''
    input: two sets of coordinates, icoords, jcoords; each of which are a list
           of OEDoubleArray(3) containing x, y, and z component
    output:
         xij - the x component of the distance matrix
         yij - the y component of the distance matrix
         zij - the z component of the distance matrix
         rij - the distance matrix
         rij2 - square of the distance matrix
    '''
    nullRet = [None, None, None, None, None]

    ni = len(icoords)
    nj = len(jcoords)

    try:
        iArrayX = np.array([c[0] for c in icoords])
        iArrayY = np.array([c[1] for c in icoords])
        iArrayZ = np.array([c[2] for c in icoords])
        iArrayX = np.repeat(iArrayX, nj)
        iArrayY = np.repeat(iArrayY, nj)
        iArrayZ = np.repeat(iArrayZ, nj)

        iArrayX = iArrayX.reshape(ni, nj)
        iArrayY = iArrayY.reshape(ni, nj)
        iArrayZ = iArrayZ.reshape(ni, nj)

        jArrayX = np.array([c[0] for c in jcoords])
        jArrayY = np.array([c[1] for c in jcoords])
        jArrayZ = np.array([c[2] for c in jcoords])
        jArrayX = np.repeat(jArrayX, ni)
        jArrayY = np.repeat(jArrayY, ni)
        jArrayZ = np.repeat(jArrayZ, ni)

        jArrayX = jArrayX.reshape(nj, ni)
        jArrayY = jArrayY.reshape(nj, ni)
        jArrayZ = jArrayZ.reshape(nj, ni)

        jArrayX = np.transpose(jArrayX)
        jArrayY = np.transpose(jArrayY)
        jArrayZ = np.transpose(jArrayZ)

        ijArrayX = jArrayX - iArrayX
        ijArrayY = jArrayY - iArrayY
        ijArrayZ = jArrayZ - iArrayZ
        rijArraySq = (ijArrayX * ijArrayX) + (ijArrayY * ijArrayY) + (ijArrayZ * ijArrayZ)
        rijArray = np.sqrt(rijArraySq)

        return ijArrayX, ijArrayY, ijArrayZ, rijArray, rijArraySq
    except:
        return nullRet


def GetThetaIJKMatrix(iCoords, jCoords, kCoords):
    '''
    Using the given input, calculates a matrix of angles ijk
    iCoords -> OEDoubleArray containing x, y, and z component of the reference coordinate
    jCoordsList -> list of N OEDoubleArrays, each OEDoubleArray is of size 3
    kCoordsList -> list of M OEDoubleArrays, each OEDoubleArray is of size 3
    return a N-by-M matrix of angle theta_ijk
    '''
    jiArrayX, jiArrayY, jiArrayZ, rjiArray, rjiArraySq \
        = GetPairwiseDistanceMatrix(jCoords, iCoords)
    jkArrayX, jkArrayY, jkArrayZ, rjkArray, rjkArraySq \
        = GetPairwiseDistanceMatrix(jCoords, kCoords)

    if jCoords == kCoords:
        rjkArray = np.eye(len(jCoords)) + np.sqrt(rjkArraySq)
    else:
        rjkArray = np.sqrt(rjkArraySq)

    if jCoords == iCoords:
        rjiArray = np.eye(len(jCoords)) + np.sqrt(rjiArraySq)
    else:
        rjiArray = np.sqrt(rjiArraySq)

    jiArrayX = jiArrayX / rjiArray
    jiArrayY = jiArrayY / rjiArray
    jiArrayZ = jiArrayZ / rjiArray
    rjkArray = np.select([rjkArray != 0], [rjkArray], default=0.01)
    jkArrayX = jkArrayX / rjkArray
    jkArrayY = jkArrayY / rjkArray
    jkArrayZ = jkArrayZ / rjkArray

    dotProduct = (jiArrayX * jkArrayX) + (jiArrayY * jkArrayY) + (jiArrayZ * jkArrayZ)
    dotProduct = np.select([dotProduct <= -1.0, dotProduct >= 1.0, np.abs(dotProduct) < 1.0],
                           [-0.999, 0.999, dotProduct])

    theta_ijk = np.arccos(dotProduct)

    return theta_ijk


def GetThetaIJKLMatrix(mol, iAtoms, jAtom, kAtom, lAtoms, transform=True):
    '''
    Using the given input, calculates a matrix of torsion angles around jk
    jAtom, kAtom -> OEAtombase, middle two atoms of the torsion
    iAtoms -> list of N OEAtombase
    lAtoms -> list of M OEAtombase
    return a N-by-M matrix of angle theta_ijkl
    '''
    torsions = []
    for iAtom in iAtoms:
        for lAtom in lAtoms:
            # print(f"{iAtom.GetIdx()}, {jAtom.GetIdx()}, {kAtom.GetIdx()},{lAtom.GetIdx()}")
            tor_angle = AllChem.GetDihedralDeg(mol.GetConformer(), iAtom.GetIdx(), jAtom.GetIdx(), kAtom.GetIdx(),
                                               lAtom.GetIdx())
            if not transform:
                torsions.append(tor_angle)
            else:
                torsions.append((math.pi + tor_angle) / 4.0)

    theta_ijkl = np.array(torsions)
    theta_ijkl = theta_ijkl.reshape(len(iAtoms), len(lAtoms))

    return theta_ijkl


class SymmetryFunction:
    def __init__(self):
        self.rcMax = 8.0  # distance cutoff for symmetry functions
        self.ita = 0.0001
        self.rcMin = 1.0
        self.rcIncr = 0.5
        self.rsVec = [0.0]
        self.theta_s_Vec = [0.0]
        self.rsVec_tor = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
        self.theta_s_Vec_tor = [0.0]
        self.rcRadVec = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0]
        self.rcAngVec = [4.5]
        self.rcTorVec = [2.5, 3.5, 5.0, 10.0]
        self.rs = 0.0  # parameter determining shape of the function
        self.itaVec = [0.0001]  # parameter determining shape of the function
        self.lambda1 = 0.5  # parameter for angular symmetry function
        self.chi = 0.5  # parameter for angular symmetry function
        self.elemList = [1, 6, 7, 8, 9, 16, 17, "pc", "nc"]

    def GetEnvAtomCoords(self, elem, refAtom, envMol, envAtoms):
        elemEnvList = []
        for envAtom in envAtoms:
            if envAtom == refAtom:
                continue

            if elem == 'pc' and envAtom.GetFormalCharge() >= 1:
                elemEnvList.append(envAtom)
            elif elem == 'nc' and envAtom.GetFormalCharge() <= -1:
                elemEnvList.append(envAtom)
            elif envAtom.GetAtomicNum() == elem:
                elemEnvList.append(envAtom)

        coordsList = []
        for elemEnvAtom in elemEnvList:
            coords = envMol.GetConformer().GetAtomPosition(elemEnvAtom.GetIdx())
            coordsList.append(coords)
        return coordsList

    def GetTorsionEnvAtoms(self, elem, bgnAtom, endAtom, envMol):
        elemEnvList = []
        for envAtom in OEGetSubtree(bgnAtom, endAtom, envMol):
            if elem == 'pc' and envAtom.GetFormalCharge() >= 1:
                elemEnvList.append(envAtom)
            elif elem == 'nc' and envAtom.GetFormalCharge() <= -1:
                elemEnvList.append(envAtom)
            elif envAtom.GetAtomicNum() == elem:
                elemEnvList.append(envAtom)

        coordsList = []
        ####遍历bond的一个起始点的原子
        for elemEnvAtom in elemEnvList:
            coords = envMol.GetConformer().GetAtomPosition(elemEnvAtom.GetIdx())
            coordsList.append(coords)
        return elemEnvList, coordsList

    def CalculateTorsionSymmetryFunction(self, envMol, num_iter):
        '''
        Takes refAtom coordinates from refMol as reference and calculates the angular symmetry
        function using envMol atoms

        Functional form is described in the DFT-NN review article by Behler, page 30, equations 25 and 26
        '''
        tsf = []
        elemList = self.elemList
        nullRet = []

        bond = get_torsion_oebond(envMol)
        jAtom = bond.GetBeginAtom()
        jcoords = envMol.GetConformer().GetAtomPosition(jAtom.GetIdx())
        kAtom = bond.GetEndAtom()
        kcoords = envMol.GetConformer().GetAtomPosition(kAtom.GetIdx())

        # tsf.append(bond.GetBgn().GetAtomicNum() * bond.GetEnd().GetAtomicNum());
        for inum, iElem in enumerate(elemList):
            if num_iter == 1:
                iAtoms, icoords = self.GetTorsionEnvAtoms(iElem, bond.GetBeginAtom(), bond.GetEndAtom(), envMol)
            else:
                iAtoms, icoords = self.GetTorsionEnvAtoms(iElem, bond.GetEndAtom(), bond.GetBeginAtom(), envMol)
            if len(icoords) == 0:
                for ita in self.itaVec:
                    for rc in self.rcTorVec:
                        for num1, _ in enumerate(elemList):
                            if num1 < inum:
                                continue
                            tsf.append(0.0)
                continue
            _, _, _, rij, _ = GetPairwiseDistanceMatrix(icoords, [jcoords])

            for lnum, lElem in enumerate(elemList):
                if lnum < inum:
                    continue
                if num_iter == 1:
                    lAtoms, lcoords = self.GetTorsionEnvAtoms(lElem, bond.GetEndAtom(), bond.GetBeginAtom(), envMol)
                else:
                    lAtoms, lcoords = self.GetTorsionEnvAtoms(lElem, bond.GetBeginAtom(), bond.GetEndAtom(), envMol)
                if len(lcoords) == 0:
                    for ita in self.itaVec:
                        for rc in self.rcTorVec:
                            tsf.append(0.0)
                    continue
                _, _, _, rkl, _ = GetPairwiseDistanceMatrix([kcoords], lcoords)
                _, _, _, ril, _ = GetPairwiseDistanceMatrix(icoords, lcoords)

                theta_ijkl = GetThetaIJKLMatrix(envMol, iAtoms, jAtom, kAtom, lAtoms)
                # angular symmetry function
                for ita in self.itaVec:
                    for rc in self.rcTorVec:
                        rijMat = np.repeat(rij, rkl.size)
                        rijMat = rijMat.reshape(rij.size, rkl.size)
                        rklMat = np.repeat(rkl, rij.size)
                        rklMat = rklMat.reshape(rkl.size, rij.size)
                        rklMat = np.transpose(rklMat)

                        fcRij = np.select([rijMat <= rc, rijMat > rc],
                                          [0.5 * (np.cos(np.pi * rijMat / rc) + 1.0), 0.0])
                        fcRkl = np.select([rklMat <= rc, rklMat > rc],
                                          [0.5 * (np.cos(np.pi * rklMat / rc) + 1.0), 0.0])
                        fcRil = np.select([ril <= rc, ril > rc], [0.5 * (np.cos(np.pi * ril / rc) + 1.0), 0.0])
                        exponent = ita * (np.square(rijMat) + np.square(rklMat) + np.square(ril))
                        term1 = np.power((1 + self.lambda1 * np.cos(theta_ijkl)), self.chi)
                        term2 = np.exp(-exponent)
                        term3 = (fcRij * fcRkl) * fcRil
                        sumIL = np.sum(term1 * term2 * term3)
                        coeff = np.power(2, 1 - self.chi) * sumIL
                        tsf.append(coeff * jAtom.GetAtomicNum() * kAtom.GetAtomicNum())

        a, b, c, d = get_torsion_oeatom_list(envMol)
        tsf.append(bond_distance2(envMol, a, d))
        tsf.append(bond_distance2(envMol, b, c))
        tsf.append(AllChem.GetDihedralDeg(envMol.GetConformer(), a, b, c, d))
        a_atom = envMol.GetAtomWithIdx(a)
        b_atom = envMol.GetAtomWithIdx(b)
        c_atom = envMol.GetAtomWithIdx(c)
        d_atom = envMol.GetAtomWithIdx(d)
        tsf.append(a_atom.GetAtomicNum() * d_atom.GetAtomicNum())
        tsf.append(b_atom.GetAtomicNum() * c_atom.GetAtomicNum())

        return tsf

    def GetTorsionCenterAsOEMol(self, mol):
        refCoords = [0.0, 0.0, 0.0]
        try:
            ###torsion_atom 是list,对应的id为1,2,3,4
            torsion_atoms = get_torsion_oeatom_list(mol)
            bgnCoords = mol.GetConformer().GetAtomPosition(torsion_atoms[1])
            endCoords = mol.GetConformer().GetAtomPosition(torsion_atoms[2])
            refCoords[0] = (bgnCoords[0] + endCoords[0]) / 2.0
            refCoords[1] = (bgnCoords[1] + endCoords[1]) / 2.0
            refCoords[2] = (bgnCoords[2] + endCoords[2]) / 2.0
        except Exception as e:
            print(e)
            return None

        refMol = Chem.MolFromSmiles("C")
        conformer = AllChem.Conformer(refMol.GetNumAtoms())
        conformer.SetAtomPosition(0, Point3D(refCoords[0], refCoords[1], refCoords[2]))
        refMol.AddConformer(conformer)
        return refMol

    ###rsf.append(0.0)  # radial
    ##asf.append(0.0)  # angular

    def CalculateSymmetryFunction(self, envMol):
        '''
        Takes refAtom coordinates from refMol as reference and calculates the angular symmetry
        function using envMol atoms

        Functional form is described in the DFT-NN review article by Behler, page 30, equations 25 and 26
        '''
        
        refMol = self.GetTorsionCenterAsOEMol(envMol)
        
        _, b, c, _ = get_torsion_oeatom_list(envMol)
        
        refAtom = refMol.GetAtomWithIdx(0)

        rsf = []
        asf = []
        elemList = self.elemList
        nullRet = [[], []]

        icoords = refMol.GetConformer().GetAtomPosition(0)

        for jnum, jElem in enumerate(elemList):
            jcoords = self.GetEnvAtomCoords(jElem, refAtom, envMol, envMol.GetAtoms())
            if len(jcoords) == 0:
                for ita in self.itaVec:
                    for rc in self.rcRadVec:
                        rsf.append(0.0)  # radial
                        #print(f'radial distance is zero')
                    for rc in self.rcAngVec:
                        for num1, _ in enumerate(elemList):
                            if num1 < jnum:
                                continue
                            asf.append(0.0)  # angular
                continue
            # ijX, ijY, ijZ, rij, rij2 = GetPairwiseDistanceMatrix([icoords], jcoords)
            _, _, _, rij, _ = GetPairwiseDistanceMatrix([icoords], jcoords)

            for ita in self.itaVec:
                expArg = ita * ((rij - self.rs) * (rij - self.rs))
                expTerm = np.exp(-expArg)

                # radial symmetry function
                for rc in self.rcRadVec:
                    fc = np.select([rij <= rc, rij > rc], [0.5 * (np.cos(np.pi * rij / rc) + 1.0), 0.0])
                    prod = expTerm * fc
                    coeff = np.sum(prod)
                    rsf.append(
                        coeff * envMol.GetAtomWithIdx(b).GetAtomicNum() * envMol.GetAtomWithIdx(c).GetAtomicNum())

            for knum, kElem in enumerate(elemList):
                if knum < jnum:
                    continue
                kcoords = self.GetEnvAtomCoords(kElem, refAtom, envMol, envMol.GetAtoms())
                if len(kcoords) == 0:
                    for ita in self.itaVec:
                        for rc in self.rcAngVec:
                            asf.append(0.0)  # angular
                    continue
                _, _, _, rik, _ = GetPairwiseDistanceMatrix([icoords], kcoords)
                _, _, _, rjk, _ = GetPairwiseDistanceMatrix(jcoords, kcoords)

                theta_ijk = GetThetaIJKMatrix([icoords], jcoords, kcoords)
                # angular symmetry function
                for ita in self.itaVec:
                    for rc in self.rcAngVec:
                        rijMat = np.repeat(rij, rik.size)
                        rijMat = rijMat.reshape(rij.size, rik.size)
                        rikMat = np.repeat(rik, rij.size)
                        rikMat = rikMat.reshape(rik.size, rij.size)
                        rikMat = np.transpose(rikMat)

                        fcRij = np.select([rijMat <= rc, rijMat > rc],
                                          [0.5 * (np.cos(np.pi * rijMat / rc) + 1.0), 0.0])
                        fcRik = np.select([rikMat <= rc, rikMat > rc],
                                          [0.5 * (np.cos(np.pi * rikMat / rc) + 1.0), 0.0])
                        fcRjk = np.select([rjk <= rc, rjk > rc], [0.5 * (np.cos(np.pi * rjk / rc) + 1.0), 0.0])
                        exponent = ita * (np.square(rijMat) + np.square(rikMat) + np.square(rjk))
                        term1 = np.power((1 + self.lambda1 * np.cos(theta_ijk)), self.chi)
                        term2 = np.exp(-exponent)
                        term3 = (fcRij * fcRjk) * fcRik
                        sumJK = np.sum(term1 * term2 * term3)
                        coeff = np.power(2, 1 - self.chi) * sumJK
                        asf.append(
                            coeff * envMol.GetAtomWithIdx(b).GetAtomicNum() * envMol.GetAtomWithIdx(c).GetAtomicNum())

        return rsf, asf


def get_sf_elements(mol):
    sfObj = SymmetryFunction()
    rsf, asf = sfObj.CalculateSymmetryFunction(mol)
    tsf1 = sfObj.CalculateTorsionSymmetryFunction(mol, 1)
    tsf2 = sfObj.CalculateTorsionSymmetryFunction(mol, 2)

    tsf = []
    for elem1, elem2 in zip(tsf1, tsf2):
        tsf.append(elem1 + elem2)

    sf_elements = rsf
    sf_elements.extend(asf)
    sf_elements.extend(tsf)

    return sf_elements