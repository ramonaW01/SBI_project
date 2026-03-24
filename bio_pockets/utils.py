"""
Utility Constants and Geometric Helpers
 
This module contains biological lookup tables (amino acid mappings, 
hydrophobicity scales) and geometric functions for 3D grid generation.

Core Components:
    • AA_3_TO_1: Standard three-letter to one-letter amino acid code mapping
    • KD_SCALE: Kyte-Doolittle hydrophobicity scale for residue properties
    • create_search_grid(): Generate uniform 3D grid for pocket detection

Biological Context:
    Hydrophobic pockets are preferred sites for hydrophobic drug binding.
    The KD scale identifies residues that form lipophilic binding surfaces.
    Grid spacing determines the precision vs. speed trade-off in cavity detection.

Dependencies:
    • NumPy: Array operations and numerical computations
"""

print("IMPORTING UTILS...") 

import numpy as np 

 
# BIOLOGICAL LOOKUP TABLES
 

# Standard 3-letter to 1-letter amino acid code mapping
# Used to convert between BioPython's 3-letter notation (from PDB files)
# and standard single-letter sequence notation (used in alignment databases).
#
# Example: 'ALA' (alanine, 3-letter) → 'A' (1-letter)
#          'GLY' (glycine, 3-letter) → 'G' (1-letter)
#
# This mapping covers all 20 standard amino acids.
AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# Kyte-Doolittle Hydrophobicity Scale (Normalized Values)
 
# Scale Range: -4.5 (most hydrophilic) to +4.5 (most hydrophobic)
#
# Biological Significance:
#     • Positive values (> 0): Hydrophobic residues
#       These tend to bury themselves in protein interior or line lipophilic pockets
#       Examples: ILE (+4.5), VAL (+4.2), LEU (+3.8) - often form drug binding sites
#     
#     • Negative values (< 0): Hydrophilic/charged residues  
#       These prefer surface exposure and water interaction
#       Examples: ARG (-4.5), LYS (-3.9), ASP (-3.5)
#
# Application in Pocket Detection:
#     Pockets with high average hydrophobicity are preferred sites for binding
#     small hydrophobic molecules (drugs, lipids). This scale helps rank pockets
#     by their chemical suitability for ligand binding.
#
# Reference: Kyte, J. & Doolittle, R.F. (1982). A simple method for displaying 
#            the hydropathic character of a protein. Journal of Molecular Biology
KD_SCALE = {
    'ILE': 4.5,   # Most hydrophobic - strongly prefers nonpolar environment
    'VAL': 4.2,   # Very hydrophobic
    'LEU': 3.8,   # Very hydrophobic
    'PHE': 2.8,   # Hydrophobic (aromatic)
    'CYS': 2.5,   # Hydrophobic (disulfide bonds alter behavior)
    'MET': 1.9,   # Slightly hydrophobic
    'ALA': 1.8,   # Slightly hydrophobic
    'GLY': -0.4,  # Neutral (smallest amino acid)
    'THR': -0.7,  # Slightly hydrophilic
    'SER': -0.8,  # Slightly hydrophilic
    'TRP': -0.9,  # Hydrophobic aromatic (but with polar NH)
    'TYR': -1.3,  # Aromatic with polar hydroxyl
    'PRO': -1.6,  # Special case - affects structure
    'HIS': -3.2,  # Charged/polar (ionizable at pH ~6)
    'GLU': -3.5,  # Negatively charged
    'GLN': -3.5,  # Polar
    'ASP': -3.5,  # Negatively charged
    'ASN': -3.5,  # Polar
    'LYS': -3.9,  # Positively charged - very hydrophilic
    'ARG': -4.5   # Most hydrophilic - strongly positively charged
}
 
# GEOMETRIC ALGORITHMS
 

def create_search_grid(protein_atoms: list, spacing: float = 1.0) -> np.ndarray:
    """
    Generate a uniform 3D lattice of search points surrounding the protein structure.
    
    This grid represents the continuous 3D space around the protein where cavity
    detection will search for pocket points. Grid points serve as candidates for
    geometric analysis: each point is tested to determine if it represents an
    empty cavity location (not clashing with protein atoms, near surface, etc.).
    
    Grid Generation Strategy:
        1. Compute protein bounding box (min/max coordinates of all atoms)
        2. Apply buffer to include surface-level cavities (typically 5.0 Å)
        3. Generate regular lattice points at specified spacing within box
        4. Return as 2D array for efficient batch processing
    
    Configuration Parameters:
        spacing (float): Distance in Ångströms between adjacent grid points
            • Smaller spacing (0.5-0.8 Å): Higher precision, detects finer cavities
                                            Slower computation, more memory usage
            • Medium spacing (1.0-1.5 Å): Good balance (RECOMMENDED for most proteins)
                                          Default: 1.0 Å
            • Larger spacing (2.0+ Å): Fast but may miss small pockets
                                       Use only for very large proteins
            
            Empirical rule: spacing should be smaller than typical ligand van der Waals diameter (~2-3 Å)
        
        buffer (hard-coded 5.0 Å): Distance to extend bounding box beyond protein atoms
            • Purpose: Capture shallow cavities and clefts at protein surface
            • Typical value: 5.0 Å (covers protein hydration shell)
            • Larger values: Include more surrounding space (slower, uses more RAM)
            • Smaller values: Focus on internal pockets (faster but may miss surface pockets)
    
    Args:
        protein_atoms (list): List of Bio.PDB.Atom objects from get_protein_structure().
                             These represent all heavy atoms in the protein structure.
        spacing (float, optional): Spacing between grid points in Ångströms.
                                  Default: 1.0 Ångström
    
    Returns:
        np.ndarray: Shape (N, 3) array of float32 coordinates (x, y, z).
                   Each row is one grid point to be tested for pocket characteristics.
    
    Computational Complexity:
        • Grid points generated: ~(box_volume / spacing³)
        • For typical protein (50×50×50 Å box, 1.0 Å spacing): ~125,000 points
        • For small pocket (10×10×10 Å, 1.0 Å spacing): ~1,000 points
        • Larger spacing ⟹ exponentially fewer points (e.g., 2.0 Å gives 1/8 the points)
    
    Memory Usage:
        Each point uses 3 × 8 bytes (3 float64 coordinates) = 24 bytes
        • 100,000 points: ~2.4 MB
        • 1,000,000 points: ~24 MB
        • 10,000,000 points: ~240 MB
    
    Example:
        # Create grid for cavity detection
        protein_atoms = [atom1, atom2, ...]  # From get_protein_structure()
        grid = create_search_grid(protein_atoms, spacing=1.0)
        print(f"Grid contains {len(grid)} points")  # e.g., "Grid contains 125432 points"
    """
    
    # Print confirmation message for package initialization tracking
    print(" Generating 3D Search Grid ")
    
    # Buffer distance: extends the search space beyond protein atoms to capture surface cavities
    # 5.0 Ångströms is empirically chosen to cover:
    #   - Protein atomic van der Waals radii (1-2 Å from surface)
    #   - Hydration shell around protein (2-3 Å)
    #   - Shallow binding clefts and grooves
    BUFFER = 5.0
    
    # Extract all atomic coordinates from protein atoms into NumPy array
    # Shape: (num_atoms, 3) where each row is [x, y, z]
    coords = np.array([atom.get_coord() for atom in protein_atoms])
    
    # Compute the axis-aligned bounding box (AABB) of the protein
    # This is the smallest rectangular box that completely contains all atoms
    min_coords = coords.min(axis=0) - BUFFER  # Subtract buffer to expand box outward
    max_coords = coords.max(axis=0) + BUFFER  # Add buffer to expand box outward
    
    # Generate discrete grid points along each axis with specified spacing
    # arange(start, stop, step) produces array [start, start+step, start+2*step, ..., < stop]
    # Example: arange(0, 10, 1) → [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    x = np.arange(min_coords[0], max_coords[0], spacing)
    y = np.arange(min_coords[1], max_coords[1], spacing)
    z = np.arange(min_coords[2], max_coords[2], spacing)
    
    # Create 3D Cartesian product of x, y, z coordinates
    # meshgrid() generates all combinations of x, y, z values
    # Example: x=[0,1], y=[0,1], z=[0,1] → 8 combinations (corner points of unit cube)
    # np.stack(..., axis=-1) combines the three grids into a single array
    # Result shape: (len(x), len(y), len(z), 3)
    #
    # reshape(-1, 3) flattens the 4D array into 2D: (N, 3) where N = len(x)*len(y)*len(z)
    # This produces all grid points as rows in the output array
    grid = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
    
    return grid