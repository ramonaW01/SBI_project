import numpy as np

print("Importing Analysis")

a = np.random.random(2)

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from collections import Counter
from Bio import AlignIO
from .utils import KD_SCALE, AA_3_TO_1  # Relative import from utils.py

def find_pocket_points(grid_points: np.ndarray, protein_atoms: list, min_dist=2.0, max_dist=5.0, surface_threshold=6.5, min_neighbors=25, max_neighbors=50) -> np.ndarray:
    if len(grid_points) == 0: return np.empty((0, 3))
    
    atom_coords = np.array([atom.get_coord() for atom in protein_atoms])
    tree = KDTree(atom_coords)
    
    clash_neighbors = tree.query_ball_point(grid_points, min_dist)
    candidates = grid_points[np.array([len(n) == 0 for n in clash_neighbors])]
    if len(candidates) == 0: return np.empty((0, 3))
    
    surface_neighbors = tree.query_ball_point(candidates, max_dist)
    candidates = candidates[np.array([len(n) > 0 for n in surface_neighbors])]
    if len(candidates) == 0: return np.empty((0, 3))
    
    surface_access = tree.query_ball_point(candidates, surface_threshold)
    pocket_points = candidates[np.array([min_neighbors <= len(n) <= max_neighbors for n in surface_access])]
    
    return pocket_points

def cluster_pocket_points(pocket_points: np.ndarray, eps=1.5, min_samples=15, min_points=100, max_points=1000) -> dict:
    if len(pocket_points) == 0: return {}
    
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(pocket_points)
    
    unique_labels = [lbl for lbl in set(labels) if lbl != -1]
    raw_pockets = {lbl: pocket_points[labels == lbl] for lbl in unique_labels}
    sorted_pockets = dict(sorted(raw_pockets.items(), key=lambda item: len(item[1]), reverse=True))
    
    pockets = {new_id: pts for new_id, (_, pts) in enumerate(((lbl, pts) for lbl, pts in sorted_pockets.items() if min_points <= len(pts) <= max_points), start=1)}
    return pockets

def extract_and_save_residues(pockets_dict: dict, protein_atoms: list, output_file="pocket_residues.txt", threshold=4.5):
    atom_coords = np.array([atom.get_coord() for atom in protein_atoms])
    tree = KDTree(atom_coords)
    
    with open(output_file, 'w') as f:
        f.write("LIGAND BINDING SITE PREDICTIONS\n\n")
        for p_id, points in pockets_dict.items():
            neighbor_indices = tree.query_ball_point(points, threshold)
            flat_indices = set(idx for sublist in neighbor_indices for idx in sublist)
            
            found_residues = set()
            for idx in flat_indices:
                res = protein_atoms[idx].get_parent()
                found_residues.add(f"{res.get_resname()}-{res.get_parent().id}{res.id[1]}")
                
            f.write(f"Pocket {p_id} ({len(points)} points):\nLining Residues: {', '.join(sorted(found_residues))}\n\n")

def calculate_conservation_scores(alignment_file: str, protein_atoms: list, aln_format="fasta") -> dict:
    alignment = AlignIO.read(alignment_file, aln_format)
    target_seq = str(alignment[0].seq)
    num_seqs = len(alignment)
    
    col_scores = []
    for i in range(alignment.get_alignment_length()):
        residues = [r.upper() for r in alignment[:, i] if r not in ('-', '.')]
        if not residues:
            col_scores.append(0.0)
            continue
        col_scores.append(Counter(residues).most_common(1)[0][1] / num_seqs)
        
    pdb_residues = []
    full_pdb_seq = ""
    for atom in protein_atoms:
        res = atom.get_parent()
        res_key = f"{res.get_parent().id}_{res.id[1]}"
        if not pdb_residues or pdb_residues[-1] != res_key:
            pdb_residues.append(res_key)
            full_pdb_seq += AA_3_TO_1.get(res.get_resname(), "X")
            
    aligned_seq_no_gaps = target_seq.replace('-', '').replace('.', '').upper()
    offset = max(0, full_pdb_seq.find(aligned_seq_no_gaps))
    
    conservation_dict = {}
    pdb_idx = offset
    for i, char in enumerate(target_seq):
        if char not in ('-', '.') and pdb_idx < len(pdb_residues):
            conservation_dict[pdb_residues[pdb_idx]] = col_scores[i]
            pdb_idx += 1
            
    return conservation_dict

def filter_pockets_by_conservation(pockets_dict: dict, protein_atoms: list, conservation_dict: dict, threshold=4.5, min_score=0.45) -> dict:
    atom_coords = np.array([atom.get_coord() for atom in protein_atoms])
    tree = KDTree(atom_coords)
    conserved_pockets = {}
    
    for p_id, points in pockets_dict.items():
        neighbor_indices = tree.query_ball_point(points, threshold)
        flat_indices = set(idx for sublist in neighbor_indices for idx in sublist)
        
        pocket_res_keys = set(f"{protein_atoms[idx].get_parent().get_parent().id}_{protein_atoms[idx].get_parent().id[1]}" for idx in flat_indices)
        scores = [conservation_dict.get(k, 0.0) for k in pocket_res_keys]
        
        if scores and sum(scores) / len(scores) >= min_score:
            conserved_pockets[p_id] = points
            
    return conserved_pockets

def rank_pockets_master_score(pockets_dict, protein_atoms, threshold=4.5):
    atom_coords = np.array([atom.get_coord() for atom in protein_atoms])
    tree = KDTree(atom_coords)
    results = []
    
    for p_id, points in pockets_dict.items():
        neighbor_indices = tree.query_ball_point(points, threshold)
        flat_indices = set(idx for sublist in neighbor_indices for idx in sublist)
        
        residues = set(protein_atoms[idx].get_parent() for idx in flat_indices)
        hydro_scores = [KD_SCALE.get(res.get_resname(), 0.0) for res in residues]
        avg_hydro = sum(hydro_scores) / len(hydro_scores) if hydro_scores else 0.0
        
        size = len(points)
        size_score = min(40, max(0, (size - 50) / 250 * 40))
        hydro_points = min(40, max(0, (avg_hydro + 2.0) / 4.0 * 40))
        cons_points = 10  # Placeholder for conservation bonus
        
        results.append({
            'Pocket_ID': p_id,
            'Master_Score': round(size_score + hydro_points + cons_points, 1),
            'Hydrophobicity (KD)': round(avg_hydro, 2),
            'Size (Points)': size,
            'Residues': len(residues)
        })
            
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by='Master_Score', ascending=False).reset_index(drop=True)
        df.index += 1
        df.index.name = 'Rank'
        
    return df
