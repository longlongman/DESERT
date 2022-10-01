from rdkit import Chem
import numpy as np
from rdkit.Chem import rdMolTransforms
from copy import deepcopy
import re
import random
from functools import cmp_to_key

PLACE_HOLDER_ATOM = 80 # Hg

def find_parts_bonds(mol, parts):
    ret_bonds = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            i_part = parts[i]
            j_part = parts[j]
            for i_atom_idx in i_part:
                for j_atom_idx in j_part:
                    bond = mol.GetBondBetweenAtoms(i_atom_idx, j_atom_idx)
                    if bond is None:
                        continue
                    ret_bonds.append((i_atom_idx, j_atom_idx))
    return ret_bonds

def get_other_atom_idx(mol, atom_idx_list):
    ret_atom_idx = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in atom_idx_list:
            ret_atom_idx.append(atom.GetIdx())
    return ret_atom_idx

def get_rings(mol):
    rings = []
    for ring in list(Chem.GetSymmSSSR(mol)):
        ring = list(ring)
        rings.append(ring)
    return rings

def get_bonds(mol, bond_type):
    bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() is bond_type:
            bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    return bonds

def get_center(mol, confId=-1):
    conformer = mol.GetConformer(confId)
    center = np.mean(conformer.GetPositions(), axis=0)
    return center

def trans(x, y, z):
    translation = np.eye(4)
    translation[:3, 3] = [x, y, z]
    return translation

def centralize(mol, confId=-1):
    mol = deepcopy(mol)
    conformer = mol.GetConformer(confId)
    center = get_center(mol, confId)
    translation = trans(-center[0], -center[1], -center[2])  
    rdMolTransforms.TransformConformer(conformer, translation)
    return mol

def canonical_frag_smi(frag_smi):
    frag_smi = re.sub(r'\[\d+\*\]', '[*]', frag_smi)
    canonical_frag_smi = Chem.CanonSmiles(frag_smi)
    return canonical_frag_smi

def get_surrogate_frag(frag):
    frag = deepcopy(frag)
    m_frag = Chem.RWMol(frag)
    for atom in m_frag.GetAtoms():
        if atom.GetSymbol() == '*':
            atom_idx = atom.GetIdx()
            m_frag.ReplaceAtom(atom_idx, Chem.Atom(PLACE_HOLDER_ATOM))
    Chem.SanitizeMol(m_frag)
    return m_frag

def get_align_points(frag1, frag2):
    align_point1 = np.zeros((frag1.GetNumAtoms(), 3))
    align_point2 = np.zeros((frag2.GetNumAtoms(), 3))
    frag12frag2 = dict()
    frag22farg1 = dict()
    order1 = list(Chem.CanonicalRankAtoms(frag1, breakTies=True))
    order2 = list(Chem.CanonicalRankAtoms(frag2, breakTies=True))
    con1 = frag1.GetConformer()
    con2 = frag2.GetConformer()
    for i in range(len(order1)):
        frag_idx1 = order1.index(i)
        frag_idx2 = order2.index(i)
        assert frag1.GetAtomWithIdx(frag_idx1).GetSymbol() == frag2.GetAtomWithIdx(frag_idx2).GetSymbol()
        atom_pos1 = list(con1.GetAtomPosition(frag_idx1))
        atom_pos2 = list(con2.GetAtomPosition(frag_idx2))
        align_point1[i] = atom_pos1
        align_point2[i] = atom_pos2
        frag12frag2[frag_idx1] = frag_idx2
        frag22farg1[frag_idx2] = frag_idx1
    return align_point1, align_point2, frag12frag2, frag22farg1

def get_atom_mapping_between_frag_and_surrogate(frag, surro):
    con1 = frag.GetConformer()
    con2 = surro.GetConformer()
    pos2idx1 = dict()
    pos2idx2 = dict()
    for atom in frag.GetAtoms():
        pos2idx1[tuple(con1.GetAtomPosition(atom.GetIdx()))] = atom.GetIdx()
    for atom in surro.GetAtoms():
        pos2idx2[tuple(con2.GetAtomPosition(atom.GetIdx()))] = atom.GetIdx()
    frag2surro = dict()
    surro2frag = dict()
    for key in pos2idx1.keys():
        frag_idx = pos2idx1[key]
        surro_idx = pos2idx2[key]
        frag2surro[frag_idx] = surro_idx
        surro2frag[surro_idx] = frag_idx
    return frag2surro, surro2frag

def get_tree(adj_dict, start_idx, visited, iter_num):
    ret = [start_idx]
    visited.append(start_idx)
    for i in range(iter_num):
        if (not i in visited) and ((start_idx, i) in adj_dict):
            ret.append(get_tree(adj_dict, i, visited, iter_num))
    visited.pop()
    return ret

def get_tree_high(tree):
    if len(tree) == 1:
        return 1
    
    subtree_highs = []
    for subtree in tree[1:]:
        subtree_high = get_tree_high(subtree)
        subtree_highs.append(subtree_high)
    
    return 1 + max(subtree_highs)

def tree_sort_cmp(a_tree, b_tree):
    a_tree_high = get_tree_high(a_tree)
    b_tree_high = get_tree_high(b_tree)

    if a_tree_high < b_tree_high:
        return -1
    if a_tree_high > b_tree_high:
        return 1
    return random.choice([-1, 1])

def tree_linearize(tree, res):
    res.append(tree[0])
    
    subtrees = tree[1:]
    subtrees.sort(key=cmp_to_key(tree_sort_cmp))
    
    for subtree in subtrees:
        if subtree != subtrees[-1]:
            res.append('b')
            tree_linearize(subtree, res)
            res.append('e')
        else:
            tree_linearize(subtree, res)
