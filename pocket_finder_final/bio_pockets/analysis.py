"""
Core Analysis Module: Pocket Detection, Conservation, and Ranking
----------------------------------------------------------------
Unified version combining automated MSA searching (Jackhmmer), 
geometric pocket detection (DBSCAN), and detailed chemical profiling.
"""
import os
import subprocess
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from collections import Counter
from Bio import AlignIO
from .utils import KD_SCALE, AA_3_TO_1

print("IMPORTING ANALYSIS...")

# =================================================================
# 1. GEOMETRIC POCKET DETECTION
# =================================================================

def find_pocket_points(grid_points: np.ndarray, protein_atoms: list, 
                       min_dist=2.0, max_dist=4.0, surface_threshold=6.5, 
                       min_neighbors=30, max_neighbors=80,
                       enclosure_radii=(4.0, 6.0, 8.0),  # NEW: Enclosure check
                       min_enclosure=0.4) -> np.ndarray:
    """
    Identifies empty 3D space near the protein surface with high curvature.
    """
    if len(grid_points) == 0: return np.empty((0, 3))
    
    atom_coords = np.array([atom.get_coord() for atom in protein_atoms])
    tree = KDTree(atom_coords)
    
    # --- Step 1: Clash check (no atoms too close) ---
    clash_neighbors = tree.query_ball_point(grid_points, min_dist)
    candidates = grid_points[np.array([len(n) == 0 for n in clash_neighbors])]
    
    print(f"  [DEBUG] After clash check:   {len(candidates)} candidates")
    
    # --- Step 2: Surface proximity (must be near protein) ---
    surface_neighbors = tree.query_ball_point(candidates, max_dist)
    candidates = candidates[np.array([len(n) > 0 for n in surface_neighbors])]
    
    print(f"  [DEBUG] After surface check: {len(candidates)} candidates")
    
    # --- Step 3: Density check (deep cleft vs flat surface) ---
    surface_access = tree.query_ball_point(candidates, surface_threshold)
    candidates = candidates[np.array([min_neighbors <= len(n) <= max_neighbors for n in surface_access])]
    
    print(f"  [DEBUG] After density check: {len(candidates)} candidates")
    
    # --- Step 4: NEW - Enclosure check (true pocket vs flat surface) ---
    # Check if the point is surrounded by atoms from multiple directions
    # 26 directions = all combinations of -1, 0, +1 in 3D (cube neighbors)
    directions = np.array([
        [dx, dy, dz]
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dz in [-1, 0, 1]
        if not (dx == 0 and dy == 0 and dz == 0)  # Exclude the center point
    ], dtype=float)
    
    # Normalize to unit vectors
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    
    enclosed_mask = []
    enclosure_scores = []

    for point in candidates:
        directions_with_atoms = 0
        for direction in directions:
            for radius in enclosure_radii:
                probe = point + direction * radius
                neighbors = tree.query_ball_point(probe, 2.0)
                if len(neighbors) > 0:
                    directions_with_atoms += 1
                    break # This direction is covered, check the next one
        
        # At least min_enclosure% of the directions must have atoms
        enclosure_score = directions_with_atoms / len(directions)
        enclosure_scores.append(enclosure_score)
        enclosed_mask.append(enclosure_score >= min_enclosure)
    
    if enclosure_scores:
        print(f"  [DEBUG] Enclosure scores - min: {min(enclosure_scores):.2f}, max: {max(enclosure_scores):.2f}, mean: {np.mean(enclosure_scores):.2f}")
    print(f"  [DEBUG] After enclosure check: {sum(enclosed_mask)} candidates")
    
    pocket_points = candidates[np.array(enclosed_mask)]
    return pocket_points
   

def cluster_pocket_points(pocket_points: np.ndarray, eps=1.5, min_samples=15, 
                          min_points=100, max_points=1000) -> dict:
    """
    Groups detected points into discrete pockets.
    """
    if len(pocket_points) == 0:
        return {}

    # --- Step 1: DBSCAN initial clustering ---
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(pocket_points)

    unique_labels = [lbl for lbl in set(labels) if lbl != -1]
    raw_pockets = {lbl: pocket_points[labels == lbl] for lbl in unique_labels}

    # --- Step 2: Size filtering ---
    filtered = [pts for pts in raw_pockets.values()
                if min_points <= len(pts) <= max_points]

    sorted_pockets = sorted(filtered, key=len, reverse=True)

    if not sorted_pockets:
        return {}

    # --- FINAL RETURN (no exclusive reassignment) ---
    return {i + 1: pts for i, pts in enumerate(sorted_pockets)}


# =================================================================
# 2. EVOLUTIONARY CONSERVATION (JACKHMMER)
# =================================================================

def run_jackhmmer_alignment(sequence_fasta: str, database_path: str, output_aln: str):
    """Executes a local Jackhmmer search against the provided database."""
    print(f"--- Running Jackhmmer Evolutionary Analysis ---")
    command = ["jackhmmer", "--cpu", "2", "-A", output_aln, sequence_fasta, database_path]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_aln
    except Exception as e:
        print(f"Error running Jackhmmer: {e}")
        return None

def calculate_conservation_scores(alignment_file: str, protein_atoms: list) -> dict:
    """Calculates residue conservation from the Jackhmmer Stockholm output."""
    alignment = AlignIO.read(alignment_file, "stockholm")
    num_seqs = len(alignment)
    
    col_scores = []
    for i in range(alignment.get_alignment_length()):
        residues = [r.upper() for r in alignment[:, i] if r not in ('-', '.')]
        score = Counter(residues).most_common(1)[0][1] / num_seqs if residues else 0.0
        col_scores.append(score)
    
    # Map back to PDB residues
    pdb_res_keys = []
    for atom in protein_atoms:
        res = atom.get_parent()
        key = f"{res.get_parent().id}_{res.id[1]}"
        if not pdb_res_keys or pdb_res_keys[-1] != key:
            pdb_res_keys.append(key)
            
    return {pdb_res_keys[i]: score for i, score in enumerate(col_scores) if i < len(pdb_res_keys)}

# =================================================================
# 3. MASTER RANKING AND PROFILING
# =================================================================

def rank_pockets_master_score(pockets_dict: dict, protein_atoms: list, conservation_dict: dict = None) -> list:
    """
    Unified Ranking: Geometry + Conservation + Chemical Profiling.
    """
    atom_coords = np.array([atom.get_coord() for atom in protein_atoms])
    tree = KDTree(atom_coords)
    results = []
    
    # Chemical Group Definitions
    HYDROPHOBIC = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP', 'PRO'}
    POLAR, POS, NEG = {'SER', 'THR', 'ASN', 'GLN', 'CYS'}, {'ARG', 'LYS', 'HIS'}, {'ASP', 'GLU'}
    
    for p_id, points in pockets_dict.items():
        neighbor_indices = tree.query_ball_point(points, 4.5)
        flat_indices = set(idx for sublist in neighbor_indices for idx in sublist)
        residues = set(protein_atoms[idx].get_parent() for idx in flat_indices)
        
        # Calculate scores
        avg_hydro = np.mean([KD_SCALE.get(r.get_resname(), 0.0) for r in residues]) if residues else 0.0
        avg_cons = np.mean([conservation_dict.get(f"{r.get_parent().id}_{r.id[1]}", 0.0) for r in residues]) if conservation_dict and residues else 0.0
        
        # Composition and Ligand Preference
        comp = {'hydrophobic': 0, 'polar': 0, 'positive': 0, 'negative': 0, 'special': 0}
        res_strings = []
        for r in residues:
            n = r.get_resname()
            res_strings.append(f"{n}{r.id[1]}")
            if n in HYDROPHOBIC: comp['hydrophobic'] += 1
            elif n in POLAR: comp['polar'] += 1
            elif n in POS: comp['positive'] += 1
            elif n in NEG: comp['negative'] += 1
            else: comp['special'] += 1
            
        # Determine Preference
        if comp['hydrophobic'] > len(residues) * 0.5: preference = "Lipophilic/Hydrophobic"
        elif comp['positive'] > comp['negative']: preference = "Anionic (Acidic) Ligands"
        else: preference = "Mixed/Polar"

        # Master Score Calculation (Weighting: Cons 50%, Hydro 30%, Size 20%)
        final_score = (min(20, len(points)/50)) + (min(30, (avg_hydro + 4.5)*3)) + (avg_cons * 50)
        
        results.append({
            'id': p_id, 'score': round(final_score, 1), 'size': len(points),
            'preference': preference, 'composition': comp, 'residues': sorted(res_strings)
        })
        
    return sorted(results, key=lambda x: x['score'], reverse=True)

def save_pocket_ranking_to_file(ranked_pockets: list, filename="pocket_ranking.txt"):
    """Saves detailed human-readable report."""
    with open(filename, "w") as f:
        f.write("POCKET FINDER: FINAL BINDING SITE RANKING\n==========================================\n")
        for i, p in enumerate(ranked_pockets):
            f.write(f"\nRANK {i+1} | POCKET {p['id']} | MASTER SCORE: {p['score']}\n")
            f.write(f"Size: {p['size']} points | Preference: {p['preference']}\n")
            f.write(f"Residues: {', '.join(p['residues'])}\n")