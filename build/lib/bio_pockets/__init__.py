"""
Bio Pockets Package
---------------------
A specialized toolkit for identifying, clustering, and ranking 
protein binding pockets using structural geometry and 
evolutionary conservation (MSA) data.

This package provides high-level access to PDB processing, 3D grid 
generation, and conservation-based pocket analysis.
"""


# --- Package Initialization ---
# This print statement confirms the package is correctly mapped in the environment
print("INITIALIZING BIO POCKETS PACKAGE...") 

# --- Data I/O and Structure Handling ---
from .data import (
    get_protein_structure, # Load and parse PDB files
    save_clean_protein, # Remove water/ligands for analysis
    save_points_to_pdb, # Export pocket coordinates
    save_protein_with_colored_pockets, # Generate B-factor mapped PDBs
    extract_sequence_from_pdb # Convert PDB coordinates to FASTA
)

# --- Geometry and Grid Utilities ---
from .utils import create_search_grid # Define the 3D bounding box for scanning

# --- Core Analysis and Algorithms ---
from .analysis import (
    find_pocket_points, # Detect empty cavities in the structure
    cluster_pocket_points, # Group nearby points into discrete pockets
    calculate_conservation_scores, # Analyze MSA for evolutionary signals
    run_jackhmmer_alignment, # Execute HMMER searches against UniProt
    rank_pockets_master_score, # Combine geometry and conservation for ranking
    save_pocket_ranking_to_file # Generate a final summary report
)


__version__ = "1.0"
__author__ = "Cristina Torredemer & Ramona Walch"
