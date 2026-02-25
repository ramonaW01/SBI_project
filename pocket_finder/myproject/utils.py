print("Importing utils")

import numpy as np

# 3-letter to 1-letter amino acid code lookup
AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# Kyte-Doolittle Scale: Positive = Hydrophobic, Negative = Hydrophilic
KD_SCALE = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
    'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
    'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
    'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}

def create_search_grid(protein_atoms: list, spacing: float = 1.0) -> np.ndarray:
    """
    Creates a uniform 3D grid around the protein bounding box.
    """
    BUFFER = 5.0
    coords = np.array([atom.get_coord() for atom in protein_atoms])
    
    min_coords = coords.min(axis=0) - BUFFER
    max_coords = coords.max(axis=0) + BUFFER
    
    x = np.arange(min_coords[0], max_coords[0], spacing)
    y = np.arange(min_coords[1], max_coords[1], spacing)
    z = np.arange(min_coords[2], max_coords[2], spacing)
    
    grid = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
    return grid
