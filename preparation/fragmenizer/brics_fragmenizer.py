from rdkit import Chem
from rdkit.Chem.BRICS import FindBRICSBonds

class BRICS_Fragmenizer():
    def __inti__(self):
        self.type = 'BRICS_Fragmenizers'
    
    def get_bonds(self, mol):
        bonds = [bond[0] for bond in list(FindBRICSBonds(mol))]
        return bonds
    
    def fragmenize(self, mol, dummyStart=1):
        # get bonds need to be break
        bonds = [bond[0] for bond in list(FindBRICSBonds(mol))]
        
        # whether the molecule can really be break
        if len(bonds) != 0:
            bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]

            # break the bonds & set the dummy labels for the bonds
            dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
            break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
            dummyEnd = dummyStart + len(dummyLabels) - 1
        else:
            break_mol = mol
            dummyEnd = dummyStart - 1

        return break_mol, dummyEnd
