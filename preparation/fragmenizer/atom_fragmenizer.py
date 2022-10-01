from rdkit import Chem

class ATOM_Fragmenizer():
    def __init__(self):
        self.type = 'Atom_Fragmenizers'
    
    def get_bonds(self, mol):
        bonds = mol.GetBonds()
        return list(bonds)
    
    def fragmenize(self, mol, dummyStart=1):
        bonds = self.get_bonds(mol)
        if len(bonds) != 0:
            bond_ids = [bond.GetIdx() for bond in bonds]
            dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
            break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
            dummyEnd = dummyStart + len(dummyLabels) - 1
        else:
            break_mol = mol
            dummyEnd = dummyStart - 1
        return break_mol, dummyEnd
