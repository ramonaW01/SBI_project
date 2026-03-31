# PocketFinder

A Python pipeline for structure-based ligand binding site prediction.

**Authors:** Cristina Torredemer & Ramona Walch · **Version:** 1.0

------------------------------------------------------------------------

## What is PocketFinder?

PocketFinder detects potential ligand-binding sites in protein structures. Given a PDB file, it scans the protein surface for cavities, clusters them into discrete pockets, and ranks them by druggability based on volume, hydrophobicity, and evolutionary conservation.

------------------------------------------------------------------------

## Quick Installation

``` bash
mamba create -n pocketfinder python=3.11
mamba activate pocketfinder
mamba install -c bioconda hmmer   # optional, for conservation scoring
cd path/to/project_directory/
pip install -e .
```

------------------------------------------------------------------------

## Usage

``` bash
# Basic run — automatically uses the university server database if available
predict_pockets protein.pdb

# With a custom database for conservation scoring
predict_pockets protein.pdb /path/to/uniprot_sprot.fasta
```

Results are saved to `result/{PROTEIN_ID}/` and include a ranked pocket list, a combined PDB structure, and ready-to-use visualization scripts for UCSF Chimera and PyMOL.

------------------------------------------------------------------------

## Documentation

For a full walkthrough — including how to interpret results, advanced configuration, and visualization — see the **Tutorial**.

------------------------------------------------------------------------

## Contact

| Name | Email |
|----|----|
| Cristina Torredemer | [cristina.torredemer01\@estudiant.upf.edu](mailto:cristina.torredemer01@estudiant.upf.edu){.email} |
| Ramona Walch | [ramona.walch\@estudiant.upf.edu](mailto:ramona.walch@estudiant.upf.edu){.email} |
