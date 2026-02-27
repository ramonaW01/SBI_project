Protein Binding Pocket Finder (PocketFinder)

A command-line tool written in Python that identifies, scores, and visualizes potential ligand-binding pockets in protein structures.

Instead of relying solely on computationally expensive energy functions, this tool combines 3D geometry-based ray-casting, density-based clustering (DBSCAN), and evolutionary conservation profiling (HMMER) to accurately detect and rank functional cavities.
Features

    Intelligent Structure Cleaning: Automatically processes raw .pdb files. It removes water molecules, metal ions, and chemical ligands to expose the true apo-protein surface.

    Geometric Grid & Enclosure Scanning: Generates a 3D grid around the protein and filters points based on distance thresholds. It utilizes an advanced 26-directional ray-casting algorithm (enclosure check) to distinguish true, deep binding pockets from shallow surface grooves.

    DBSCAN Clustering: Groups the remaining valid pocket points into distinct clusters using density-based spatial clustering.

    Evolutionary Conservation (Jackhmmer): Automatically extracts the protein sequence, runs a local HMMER/Jackhmmer search against a provided FASTA database, and maps evolutionary conservation scores directly to the 3D structure.

    Biochemical Profiling & Master Scoring: Ranks identified pockets using a Master Score out of 100 (evaluating volume, local hydrophobicity, and evolutionary conservation). It also predicts the chemical preference of the pocket (e.g., Anionic, Lipophilic, Mixed/Polar).

    Automated Visualization: Exports a modified .pdb file with dummy atoms, generates ready-to-use scripts for PyMOL (.pml) and UCSF Chimera (.cmd), and automatically launches Chimera for immediate visual inspection.

Project Structure

The project is modularized for maintainability and easy extension:

    pocket_finder.py: The main command-line interface (CLI) pipeline script.

    bio_pockets/: The core Python package containing the logic.

        __init__.py: Package initialization and API routing.

        data.py: Handles I/O, loading/cleaning PDB files (via Biopython), extracting FASTA, and generating visualization scripts.

        utils.py: Contains constants (e.g., Kyte-Doolittle hydrophobicity scales) and the 3D grid generation logic.

        analysis.py: Contains the core algorithms for filtering geometry, clustering, Jackhmmer execution, and scoring.

Installation

This project is packaged using standard Python build tools. To install the tool and all required dependencies, navigate to the root directory of the project in your terminal and run:
Bash

pip install -e .

System Requirement: For the evolutionary conservation features to work, the HMMER suite (specifically jackhmmer) must be installed on your system and accessible via the system PATH.
Usage

You can run the tool directly from your terminal. It requires a .pdb file as the main argument.

Standard Analysis (with default database):
Bash

python pocket_finder.py <target_file.pdb>

Analysis with a Custom FASTA Database:
To include evolutionary conservation profiling, provide a local FASTA database (e.g., UniProt) as the second argument:
Bash

python pocket_finder.py <target_file.pdb> custom_database.fasta

(Note: If the database is not provided or not found, the tool elegantly falls back to geometry and chemistry-only scoring).
Output Files

The pipeline automatically generates several files using your input filename as a base (e.g., <target>_...):

    Terminal Summary: A quick overview of the top-ranked pockets and execution status.

    <target>_cleaned.pdb: The protein structure with all noise (water, heteroatoms) removed.

    <target>_query.fasta & <target>_alignment.sto: Sequence and alignment files generated during the HMMER search.

    <target>_ranking.txt: A detailed technical report listing pocket sizes, Master Scores, chemical preferences, and all interacting residues.

    <target>_results.pdb: The final structure file containing the protein and the predicted pockets (represented as dummy atoms, ranked by B-factor).

    Visualization Scripts: Automated commands to render the results perfectly in PyMOL or Chimera.

Dependencies

    Python >= 3.8

    Biopython

    NumPy

    Pandas

    SciPy

    Scikit-Learn
