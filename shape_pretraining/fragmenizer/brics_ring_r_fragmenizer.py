# from fragmenizer import BRICS_Fragmenizer, RING_R_Fragmenizer
from rdkit import Chem

from .brics_fragmenizer import BRICS_Fragmenizer
from .ring_r_fragmenizer import RING_R_Fragmenizer

class BRICS_RING_R_Fragmenizer():
    def __init__(self):
        self.type = 'BRICS_RING_R_Fragmenizer'
        self.brics_fragmenizer = BRICS_Fragmenizer()
        self.ring_r_fragmenizer = RING_R_Fragmenizer()
    
    def fragmenize(self, mol, dummyStart=1):
        brics_bonds = self.brics_fragmenizer.get_bonds(mol)
        ring_r_bonds = self.ring_r_fragmenizer.get_bonds(mol)
        bonds = brics_bonds + ring_r_bonds

        if len(bonds) != 0:
            bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]
            bond_ids = list(set(bond_ids))
            dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
            break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
            dummyEnd = dummyStart + len(dummyLabels) - 1
        else:
            break_mol = mol
            dummyEnd = dummyStart - 1

        return break_mol, dummyEnd
