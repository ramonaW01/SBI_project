"""
Utility Constants and Geometric Helpers
---------------------------------------
This module contains biological lookup tables (amino acid mappings, 
hydrophobicity scales) and geometric functions for 3D grid generation.
"""

print("IMPORTING UTILS...") 

import numpy as np 

# =================================================================
# BIOLOGICAL LOOKUP TABLES
# =================================================================

# Standard 3-letter to 1-letter amino acid code mapping
AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# Kyte-Doolittle Hydrophobicity Scale
# Positive values indicate hydrophobic residues; negative values indicate hydrophilic.
# Used to identify hydrophobic patches common in ligand-binding pockets.
KD_SCALE = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
    'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
    'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
    'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}

# =================================================================
# GEOMETRIC ALGORITHMS
# =================================================================

def create_search_grid(protein_atoms: list, spacing: float = 1.0) -> np.ndarray:  
    """
    Generates a uniform 3D grid of points surrounding the protein structure.
    
    The grid acts as the search space for pocket detection. A buffer is added 
    to the protein bounding box to ensure surface-level cavities are captured.

    Args:
        protein_atoms (list): A list of Bio.PDB.Atom objects.
        spacing (float): The distance (in Angstroms) between grid points.

    Returns:
        np.ndarray: A (N, 3) array containing the (x, y, z) coordinates of grid points.

    """
    # Print confirmation for internal package tracking
    print("--- Generating 3D Search Grid ---")

    # 5.0 Angstrom buffer to cover the entire protein surface and hydration shell
    BUFFER = 5.0

    # Extract atomic coordinates 
    coords = np.array([atom.get_coord() for atom in protein_atoms]) 
    
    # Define the boundaries of the protein "box"
    min_coords = coords.min(axis=0) - BUFFER
    max_coords = coords.max(axis=0) + BUFFER

    # Generate discrete steps for each axis    
    x = np.arange(min_coords[0], max_coords[0], spacing)
    y = np.arange(min_coords[1], max_coords[1], spacing)
    z = np.arange(min_coords[2], max_coords[2], spacing)

    # Create the 3D coordinate cloud using vectorized meshgrid operations
    grid = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
    
    return grid
