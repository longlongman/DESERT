from rdkit import Chem
from utils import get_rings, get_other_atom_idx, find_parts_bonds
from rdkit.Chem.rdchem import BondType

class RING_R_Fragmenizer():
    def __init__(self):
        self.type = 'RING_R_Fragmenizer'

    def bonds_filter(self, mol, bonds):
        filted_bonds = []
        for bond in bonds:
            bond_type = mol.GetBondBetweenAtoms(bond[0], bond[1]).GetBondType()
            if not bond_type is BondType.SINGLE:
                continue
            f_atom = mol.GetAtomWithIdx(bond[0])
            s_atom = mol.GetAtomWithIdx(bond[1])
            if f_atom.GetSymbol() == '*' or s_atom.GetSymbol() == '*':
                continue
            if mol.GetBondBetweenAtoms(bond[0], bond[1]).IsInRing():
                continue
            filted_bonds.append(bond)
        return filted_bonds
    
    def get_bonds(self, mol):
        bonds = []
        rings = get_rings(mol)
        if len(rings) > 0:
            for ring in rings:
                rest_atom_idx = get_other_atom_idx(mol, ring)
                bonds += find_parts_bonds(mol, [rest_atom_idx, ring])
            bonds = self.bonds_filter(mol, bonds)
        return bonds
    
    def fragmenize(self, mol, dummyStart=1):
        rings = get_rings(mol)
        if len(rings) > 0:
            bonds = []
            for ring in rings:
                rest_atom_idx = get_other_atom_idx(mol, ring)
                bonds += find_parts_bonds(mol, [rest_atom_idx, ring])
            bonds = self.bonds_filter(mol, bonds)
            if len(bonds) > 0:
                bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]
                bond_ids = list(set(bond_ids))
                dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
                break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
                dummyEnd = dummyStart + len(dummyLabels) - 1
            else:
                break_mol = mol
                dummyEnd = dummyStart - 1
        else:
            break_mol = mol
            dummyEnd = dummyStart - 1
        return break_mol, dummyEnd
