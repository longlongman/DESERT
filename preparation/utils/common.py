# van der Waals radius
ATOM_RADIUS = {
    'C': 1.908,
    'F': 1.75,
    'Cl': 1.948,
    'Br': 2.22,
    'I': 2.35,
    'N': 1.824,
    'O': 1.6612,
    'P': 2.1,
    'S': 2.0,
    'Si': 2.2, # not accurate
    'H': 1.0
}

# atomic number
ATOMIC_NUMBER = {
    'C': 6,
    'F': 9,
    'Cl': 17,
    'Br': 35,
    'I': 53,
    'N': 7,
    'O': 8,
    'P': 15,
    'S': 16,
    'Si': 14,
    'H': 1
}

ATOMIC_NUMBER_REVERSE = {v: k for k, v in ATOMIC_NUMBER.items()}
