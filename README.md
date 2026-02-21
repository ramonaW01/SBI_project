## SBI_project

# Pocket Finder
**Structure-Based Ligand Binding Site Predictor**  
*Final Project — Introduction to Python & Structural Bioinformatics (UPF)*

---

## Overview

Pocket Finder implements a geometry-based approach to predict potential ligand binding sites on protein structures. Given a protein in PDB format, the program identifies surface cavities that are geometrically and chemically suitable for small molecule binding — without requiring prior knowledge of known binding sites.

The algorithm places a 3D grid around the protein, filters points by proximity to protein atoms, clusters the remaining candidates into discrete pockets, and characterizes each pocket chemically based on its lining residues.

---

## Features

- **Clean PDB Parsing** — Automatically filters water molecules, ligands, ions, and co-factors using BioPython's `is_aa()` function
- **Efficient Spatial Search** — KD-Trees for fast distance calculations between grid points and protein atoms
- **Density-Based Clustering** — DBSCAN automatically detects and separates multiple binding sites without requiring a predefined pocket count
- **Chemical Characterization** — Classifies each pocket as hydrophobic, polar, charged, or mixed based on lining residue composition
- **Visualization Ready** — Exports results as PDB files (HETATM records) for direct use in Chimera or PyMOL

---

## Requirements

```bash
pip install numpy biopython scipy scikit-learn
```

| Package | Purpose |
|---|---|
| `numpy` | Array operations and coordinate math |
| `biopython` | PDB parsing and structure traversal |
| `scipy` | KDTree for spatial queries |
| `scikit-learn` | DBSCAN clustering |

---

## Usage

1. Place your PDB file (e.g. `1H8D.pdb`) in the project directory
2. Open `pocket_finder.ipynb` in Jupyter
3. Run all cells, or call the pipeline directly:

```python
structure, atoms = get_protein_structure("1H8D.pdb")
final_rankings = run_full_prediction(atoms, spacing=1.0)
```

4. Optionally export PDB files for visualization:

```python
export_all_steps(atoms, output_dir="pocket_output")
```

> Use `spacing=2.0` for fast testing, `spacing=1.0` for final predictions.



## How It Works

| Step | Function | Description |
|---|---|---|
| 1 | `get_protein_structure` | Parse PDB, keep only standard amino acids |
| 2 | `create_search_grid` | Generate 3D grid around the protein (default 1.0 Å spacing) |
| 3 | `find_pocket_points` | Remove interior points (< 2.5 Å) and bulk solvent (> 5.0 Å) |
| 4 | `cluster_pocket_points` | Group candidates into pockets via DBSCAN |
| 5 | `get_pocket_residues` | Identify residues lining each pocket (threshold 4.5 Å) |
| 6 | `analyze_and_rank_pockets` | Score and rank pockets by size and chemical composition |

---

## Output

**Console** — Ranked summary of the top 5 predicted binding sites including size, chemical nature, and key residues.

**PDB files** (via `export_all_steps`):
- `step1_full_grid.pdb` — Complete search grid
- `step2_pocket_candidates.pdb` — Filtered surface candidates
- `step3_pocket_0.pdb`, `step3_pocket_1.pdb`, … — Individual pocket clusters

---

## Visualization

**UCSF Chimera:** `File → Open → step3_pocket_0.pdb` → `Actions → Atoms/Bonds → sphere`

**PyMOL:**
```
load 1H8D.pdb, protein
load step3_pocket_0.pdb, pocket1
show spheres, pocket1
color blue, pocket1
```

---

## Documentation

For a full description of the algorithm, parameters, and worked example, see the included **User Tutorial** (`pocket_finder_tutorial.docx`).
