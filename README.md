# SBI_project
SBI and PYT project
Structure-Based Ligand Binding Site Predictor

This project implements a geometric and density-based approach to predict potential ligand binding sites (pockets) on protein structures using PDB files.
Overview

The algorithm identifies cavities on a protein surface by generating a 3D grid around the structure and filtering points based on their proximity to protein atoms. Finally, it groups these points into distinct clusters representing individual binding pockets.
Features

    Clean PDB Parsing: Automatically filters out water molecules and existing ligands to focus solely on the protein structure.

    Efficient Spatial Search: Utilizes KD-Trees for fast distance calculations between grid points and atoms.

    Density-Based Clustering: Uses the DBSCAN algorithm to automatically detect and separate multiple binding sites.

    Visualization Ready: Exports results as PDB files (HETATM records) for immediate analysis in Chimera, PyMOL, or other molecular graphics software.

Prerequisites

The following Python libraries are required:

    numpy

    biopython

    scipy

    scikit-learn

You can install them via pip:
Bash

pip install numpy biopython scipy scikit-learn

How It Works
1. Parsing (get_protein_structure)

The script reads a .pdb file and extracts only the coordinates of standard amino acid residues. This ensures that the "empty" spaces found later are truly available for new ligands.
2. Grid Generation (create_search_grid)

A 3D bounding box is created around the protein. A grid of points (default spacing 2.0Å for speed, 1.0Å for quality) is laid over the structure.
3. Geometric Filtering (find_pocket_points)

Each grid point is evaluated:

    Clash Filter: Points too close to any protein atom (< 2.5Å) are removed.

    Surface Filter: Points too far from the protein surface (> 5.0Å) are removed to avoid including the open solvent space.

    Remaining points represent the "negative volume" of the protein's cavities.

4. Clustering (cluster_pocket_points)

Individual points are grouped into clusters using DBSCAN. This step separates the global "cloud" of points into distinct, numbered pockets.
5. Export & Visualization (export_all_steps)

The script generates several files to document the process:

    step1_full_grid.pdb: The initial search box.

    step2_pocket_candidates.pdb: All points found in any surface indentation.

    step3_pocket_X.pdb: Individual files for each detected binding site.

Usage

    Place your target PDB file (e.g., 1H8D.pdb) in the project directory.

    Run the script in a Jupyter Notebook or Python environment.

    Load the resulting .pdb files into UCSF Chimera:

        Use Actions -> Atoms/Bonds -> sphere to visualize the grid points as spheres.

Testing

The script includes a test strategy that outputs the number of atoms found, the performance of the filtering step, and the total count of identified pockets.