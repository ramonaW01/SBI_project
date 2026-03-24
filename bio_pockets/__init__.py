"""
Bio Pockets Package
===================
A specialized toolkit for identifying, clustering, and ranking 
protein binding pockets using structural geometry and 
evolutionary conservation (MSA) data.

This package provides high-level access to PDB processing, 3D grid 
generation, and conservation-based pocket analysis.

Core Modules:
    • data: PDB file I/O and structure manipulation
    • utils: Geometric utilities and grid generation
    • analysis: Core algorithms for pocket detection and ranking

Main Use Case:
    Identify potential ligand binding sites in protein structures
    by analyzing geometric cavities and evolutionary conservation signals.
"""

# --- Package Initialization ---
# This print statement confirms the package is correctly mapped in the environment.
# Useful for debugging import issues in complex setups.
print("INITIALIZING BIO POCKETS PACKAGE...") 

# --- Data I/O and Structure Handling ---
# Functions for reading/writing PDB files and manipulating protein structures
from .data import (
    get_protein_structure,              # Load and parse PDB files into BioPython structure objects
    save_clean_protein,                 # Remove water/ligands/heteroatoms for clean analysis
    save_points_to_pdb,                 # Export 3D coordinates as pseudo-atoms in PDB format
    save_protein_with_colored_pockets,  # Generate PDB with B-factors colored by pocket ID for visualization
    extract_sequence_from_pdb,          # Convert protein coordinates to FASTA sequence format
    generate_visualization_scripts,     # Create PyMOL and Chimera command scripts for pocket display
    open_in_chimera                     # Launch Chimera automatically with visualization
)

# --- Geometry and Grid Utilities ---
# Geometric operations for 3D scanning and spatial analysis
from .utils import create_search_grid  # Generate 3D lattice points around protein structure for cavity scanning

# --- Core Analysis and Algorithms ---
# Main computational functions for pocket detection and ranking
from .analysis import (
    find_pocket_points,                 # Detect empty cavity points using geometric criteria (distance, density, enclosure)
    cluster_pocket_points,              # Group nearby cavity points into discrete pockets using DBSCAN clustering
    calculate_conservation_scores,      # Compute residue conservation scores from Multiple Sequence Alignment (MSA)
    run_jackhmmer_alignment,            # Execute HMMER/Jackhmmer sequence search against UniProt database
    rank_pockets_master_score,          # Combine geometry, size, and conservation into integrated ranking score
    save_pocket_ranking_to_file         # Write human-readable summary report with pocket rankings
)

__version__ = "1.0"
__author__ = "Cristina Torredemer & Ramona Walch"