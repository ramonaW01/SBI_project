"""
Core Analysis Module: Pocket Detection, Conservation, and Ranking
 
Unified implementation combining:
    • Geometric pocket detection (DBSCAN clustering of cavity points)
    • Evolutionary conservation analysis (Jackhmmer MSA searching)
    • Chemical profiling (hydrophobicity, residue types, ligand preference)
    • Integrated master scoring system

Pipeline Flow:
    1. find_pocket_points(): Identify empty 3D space using 4-step geometric validation
    2. cluster_pocket_points(): Group cavity points into discrete pockets via DBSCAN
    3. run_jackhmmer_alignment(): Search sequence database for homologous proteins
    4. calculate_conservation_scores(): Extract conservation from MSA alignment
    5. rank_pockets_master_score(): Integrate geometry, conservation, and chemistry

Key Innovations:
    • Multi-criteria pocket validation (clash, surface, density, enclosure)
    • Enclosure check: evaluates if cavity is truly protein-enclosed (not just surface groove)
    • Master scoring: weighted combination of multiple independent signals
    • Chemical profiling: predicts ligand type from pocket composition

Dependencies:
    • NumPy: Array operations
    • SciPy: KDTree spatial indexing, DBSCAN clustering
    • scikit-learn: DBSCAN implementation
    • BioPython: Alignment file parsing (Stockholm format)
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


# 1. GEOMETRIC POCKET DETECTION

def find_pocket_points(grid_points: np.ndarray, protein_atoms: list, 
                       min_dist=2.0, max_dist=4.0, surface_threshold=6.5, 
                       min_neighbors=30, max_neighbors=80,
                       enclosure_radii=(4.0, 6.0, 8.0),
                       min_enclosure=0.4) -> np.ndarray:
    """
    Identify empty 3D space near the protein surface with high curvature.
    
    This is the core geometric pocket detection algorithm. It uses a multi-step
    validation pipeline to identify true binding pockets while filtering out
    false positives (surface grooves, flat interfaces, noise).
    
    VALIDATION PIPELINE (4 Steps):
     
    Each grid point must pass ALL 4 criteria to be considered a pocket point:
    
    Step 1: CLASH CHECK
        Criterion: No protein atoms within min_dist (typically 2.0 Å)
        Purpose: Ensure point is in empty space (doesn't overlap protein)
        False Positive Prevention: Rejects points too close to protein body
        Example: If min_dist=2.0, reject any point with atoms <2.0 Å away
    
    Step 2: SURFACE PROXIMITY
        Criterion: At least one protein atom within max_dist (typically 4.0 Å)
        Purpose: Ensure point is close to protein surface (not too far away)
        False Positive Prevention: Rejects floating points far from protein
        Example: If max_dist=4.0, require at least 1 atom within 4.0 Å
    
    Step 3: DENSITY CHECK
        Criterion: min_neighbors ≤ atom_count ≤ max_neighbors within surface_threshold
        Purpose: Distinguish clefts (many atoms) from flat surfaces (few atoms)
        False Positive Prevention: 
            • Too few atoms (<min_neighbors): point on flat surface
            • Too many atoms (>max_neighbors): point is buried/crowded
        Example: If range is 30-80 atoms, point is in a good-sized cleft
    
    Step 4: ENCLOSURE CHECK (NEW)
        Criterion: Point surrounded by atoms in ≥min_enclosure fraction of directions
        Purpose: Ensure point is truly enclosed pocket (not just surface groove)
        Strategy: Check 26 spatial directions (3D cube neighbors), multiple radii
        False Positive Prevention: Surface grooves typically enclosed from only 1-2 sides
        Example: If min_enclosure=0.4, point must be enclosed in ≥40% of directions
    
    ENCLOSURE LOGIC DETAILED:
     
    Surrounding "enclosed" means: in at least 40% of the 26 directions around the point,
    there is a protein atom. This is checked at multiple radii (4.0, 6.0, 8.0 Å).
    
    Why 26 directions?
        - 3D cube has 26 neighbors (excluding center): all combinations of -1,0,+1
        - Covers all spatial quadrants evenly
        - Robust to local noise
    
    Why multiple radii?
        - Detect enclosure at different scales
        - Flexible pocket shapes (some wider, some narrower)
        - Only need ONE radius to have atoms for each direction
    
    Configuration Parameters:
     
    min_dist (float): Minimum distance to nearest protein atom [Å]
        • 1.5 Å: Strict, only very empty points (detects shallow cavities)
        • 2.0 Å: Standard, good balance
        • 2.5 Å: Loose, allows points closer to surface
        Default: 2.0 Å
    
    max_dist (float): Maximum distance from nearest protein atom [Å]
        • 3.0 Å: Only very close to surface (surface pockets only)
        • 4.0 Å: Standard, includes subsurface regions
        • 5.0 Å: Loose, includes deeper cavities
        Default: 4.0 Å
    
    surface_threshold (float): Radius for counting neighbor atoms [Å]
        • Smaller (4.0 Å): Tight neighborhoods, detects more discrete pockets
        • Standard (6.5 Å): Good balance for protein-scale features
        • Larger (8.0 Å): Loose, merges nearby features
        Default: 6.5 Å
    
    min_neighbors (int): Minimum atoms in surface_threshold neighborhood
        • Smaller (10): Detects shallow/exposed cavities
        • Standard (30): Balanced for typical pockets
        • Larger (50): Only deep, buried cavities
        Default: 30
    
    max_neighbors (int): Maximum atoms in surface_threshold neighborhood
        • Smaller (60): Filters out crowded regions, smaller pockets only
        • Standard (80): Allows moderate crowding
        • Larger (100): Tolerates dense regions
        Default: 80
    
    enclosure_radii (tuple): Distances to check enclosure at [Å]
        • (4.0, 6.0, 8.0): Multi-scale check, detects various pocket shapes
        • Recommendation: Always use 3+ radii for robustness
        Default: (4.0, 6.0, 8.0)
    
    min_enclosure (float): Minimum enclosure fraction (0-1)
        • 0.3: Loose, allows partial enclosure (surface grooves accepted)
        • 0.4: Standard, requires good enclosure (pockets only)
        • 0.5: Strict, requires >50% enclosure (very conservative)
        Default: 0.4
    
    Args:
        grid_points (np.ndarray): Array of shape (N, 3) containing grid candidate coordinates
        protein_atoms (list): List of Bio.PDB.Atom objects representing protein structure
        min_dist (float, optional): Min distance to atoms [Å]. Default: 2.0
        max_dist (float, optional): Max distance to atoms [Å]. Default: 4.0
        surface_threshold (float, optional): Radius for neighbor count [Å]. Default: 6.5
        min_neighbors (int, optional): Minimum atoms in neighborhood. Default: 30
        max_neighbors (int, optional): Maximum atoms in neighborhood. Default: 80
        enclosure_radii (tuple, optional): Radii for enclosure check [Å]. Default: (4.0, 6.0, 8.0)
        min_enclosure (float, optional): Min enclosure fraction (0-1). Default: 0.4
    
    Returns:
        np.ndarray: Array of shape (M, 3) with M ≤ N (filtered pocket point candidates)
                   Only points passing all 4 validation steps are returned
     """
    if len(grid_points) == 0:
        return np.empty((0, 3))
    
    # Extract protein atomic coordinates and build spatial index
    atom_coords = np.array([atom.get_coord() for atom in protein_atoms])
    tree = KDTree(atom_coords)
    
    #   STEP 1: CLASH CHECK  
    # Find all grid points that have NO atoms within min_dist
    # query_ball_point returns list of atom indices within radius for each grid point
    clash_neighbors = tree.query_ball_point(grid_points, min_dist)
    # Filter: keep only points with empty list (no atoms nearby)
    candidates = grid_points[np.array([len(n) == 0 for n in clash_neighbors])]
    
    print(f"  [DEBUG] After clash check:   {len(candidates)} candidates")
    
    #   STEP 2: SURFACE PROXIMITY  
    # Find candidates that ARE near the protein surface (within max_dist)
    # Keep only points with at least one neighbor
    surface_neighbors = tree.query_ball_point(candidates, max_dist)
    candidates = candidates[np.array([len(n) > 0 for n in surface_neighbors])]
    
    print(f"  [DEBUG] After surface check: {len(candidates)} candidates")
    
    #   STEP 3: DENSITY CHECK  
    # Count atoms within surface_threshold sphere around each candidate
    # Only keep points with "good" atom density (not too sparse, not too crowded)
    surface_access = tree.query_ball_point(candidates, surface_threshold)
    candidates = candidates[np.array([min_neighbors <= len(n) <= max_neighbors for n in surface_access])]
    
    print(f"  [DEBUG] After density check: {len(candidates)} candidates")
    
    #   STEP 4: ENCLOSURE CHECK  
    # Check if point is truly enclosed (surrounded by atoms from multiple directions)
    # 26 directions = all 3D neighbors of a cube: {-1,0,1} × {-1,0,1} × {-1,0,1} excluding center
    
    # Generate unit vectors pointing in all 26 spatial directions
    directions = np.array([
        [dx, dy, dz]
        for dx in [-1, 0, 1]      # x-axis: left, center, right
        for dy in [-1, 0, 1]      # y-axis: back, center, front
        for dz in [-1, 0, 1]      # z-axis: down, center, up
        if not (dx == 0 and dy == 0 and dz == 0)  # Exclude center point itself
    ], dtype=float)
    
    # Normalize to unit vectors (length = 1) so direction matters, not magnitude
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    
    enclosed_mask = []
    enclosure_scores = []

    # For each candidate point, check enclosure in all directions
    for point in candidates:
        directions_with_atoms = 0
        
        # Check each of 26 directions
        for direction in directions:
            # Check if this direction has atoms at ANY of the specified radii
            atom_found_in_direction = False
            
            for radius in enclosure_radii:
                # Probe point: move along this direction at this radius distance
                probe = point + direction * radius
                # Query atoms near probe point (tolerance: 2.0 Å around probe)
                neighbors = tree.query_ball_point(probe, 2.0)
                
                # If we found atoms in this direction, count it and stop checking other radii
                if len(neighbors) > 0:
                    directions_with_atoms += 1
                    atom_found_in_direction = True
                    break  # This direction is covered, move to next direction
        
        # Calculate enclosure score: fraction of directions with atoms
        # Example: if 15 out of 26 directions have atoms → score = 0.577
        enclosure_score = directions_with_atoms / len(directions)
        enclosure_scores.append(enclosure_score)
        
        # Accept point if it meets minimum enclosure threshold
        # Example: if min_enclosure=0.4, need ≥10.4 directions with atoms → 11+ directions
        enclosed_mask.append(enclosure_score >= min_enclosure)
    
    # Print debug statistics
    if enclosure_scores:
        print(f"  [DEBUG] Enclosure scores - min: {min(enclosure_scores):.2f}, max: {max(enclosure_scores):.2f}, mean: {np.mean(enclosure_scores):.2f}")
    print(f"  [DEBUG] After enclosure check: {sum(enclosed_mask)} candidates")
    
    # Return points that passed ALL validation steps
    pocket_points = candidates[np.array(enclosed_mask)]
    return pocket_points
   

def cluster_pocket_points(pocket_points: np.ndarray, eps=1.5, min_samples=15, 
                          min_points=100, max_points=1000) -> dict:
    """
    Group detected cavity points into discrete, distinct pockets via DBSCAN clustering.
    
    CLUSTERING STRATEGY:
    ===================
    DBSCAN (Density-Based Spatial Clustering) groups nearby points without requiring
    a pre-specified number of clusters. This is ideal for pockets because:
    • Unknown number of pockets in advance
    • Pockets vary in size and shape
    • Handles noise automatically (points far from others marked as noise)
    
    TWO-STAGE FILTERING:
    ===================
    Stage 1: DBSCAN Clustering
        Algorithm: DBSCAN
        Input: Candidate pocket points (output from find_pocket_points)
        Output: Cluster labels (each point assigned to a cluster or marked as noise)
        Process:
            • Points within eps (1.5 Å) of min_samples (15) other points form clusters
            • Points not in any cluster marked as noise (label = -1)
            • Each cluster represents a potential pocket
    
    Stage 2: Size Filtering
        Filter 1: Remove tiny clusters (< min_points)
                 Reason: Too small to be real pockets (probably noise)
        Filter 2: Remove huge clusters (> max_points)
                 Reason: Likely merged multiple pockets or artifact
    
    Configuration Parameters:
    ========================
    eps (float): Neighborhood radius for DBSCAN [Å]
        • Smaller (1.0 Å): Separates nearby pockets, detects finer structures
        • Standard (1.5 Å): Good balance for most proteins
        • Larger (2.0 Å): Merges nearby pockets, creates larger clusters
        Impact: Smaller eps → more clusters; Larger eps → fewer clusters
        Default: 1.5 Å
    
    min_samples (int): Minimum points in eps-neighborhood to form cluster
        • Smaller (10): Loose, easier to form clusters
        • Standard (15): Balanced
        • Larger (20): Strict, only dense clusters formed
        Impact: Lower values → more small clusters; Higher values → only large clusters
        Default: 15
    
    min_points (int): Absolute minimum size for a valid pocket [# grid points]
        • Smaller (50): Detects tiny pockets (may be noise)
        • Standard (100): Balanced for typical pockets
        • Larger (200): Only substantial pockets kept
        Typical mapping: ~100 points → 5-10 Ångströms³ (depends on grid spacing)
        Default: 100
    
    max_points (int): Absolute maximum size for a valid pocket [# grid points]
        • Smaller (500): Conservative, splits large pockets
        • Standard (1000): Good for most proteins
        • Larger (2000): Tolerates very large pockets or merged regions
        Default: 1000
    
    Args:
        pocket_points (np.ndarray): Array of shape (N, 3) from find_pocket_points()
        eps (float, optional): DBSCAN neighborhood radius [Å]. Default: 1.5
        min_samples (int, optional): DBSCAN min points per cluster. Default: 15
        min_points (int, optional): Filter: minimum cluster size. Default: 100
        max_points (int, optional): Filter: maximum cluster size. Default: 1000
    
    Returns:
        dict: Maps pocket_id (int, starting from 1) → point coordinates (np.ndarray)
              Pockets sorted by size (largest first)
              Returns {} if no valid clusters found
    """
    if len(pocket_points) == 0:
        return {}

    #  STAGE 1: DBSCAN CLUSTERING  
    # Apply DBSCAN algorithm to group nearby points
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(pocket_points)
    
    # Extract clusters (ignore noise points labeled as -1)
    # Create dictionary: cluster_label → array of points in that cluster
    unique_labels = [lbl for lbl in set(labels) if lbl != -1]
    raw_pockets = {lbl: pocket_points[labels == lbl] for lbl in unique_labels}

    #   STAGE 2: SIZE FILTERING  
    # Filter out clusters that are too small or too large
    # min_points: reject tiny noise clusters
    # max_points: reject merged clusters or artifacts that are too big
    filtered = [pts for pts in raw_pockets.values()
                if min_points <= len(pts) <= max_points]

    # Sort by size (largest first) for more important pockets listed first
    sorted_pockets = sorted(filtered, key=len, reverse=True)

    # Return empty dict if no valid pockets after filtering
    if not sorted_pockets:
        return {}

    # Create output dictionary with sequential IDs starting from 1
    # pocket_id 1, 2, 3, ... maps to sorted point arrays
    return {i + 1: pts for i, pts in enumerate(sorted_pockets)}


 # 2. EVOLUTIONARY CONSERVATION (JACKHMMER)
 
def run_jackhmmer_alignment(sequence_fasta: str, database_path: str, output_aln: str):
    """
    Execute a local Jackhmmer search against provided sequence database.
    
    Jackhmmer (part of HMMER suite) performs iterative homology search:
    1. Convert input sequence to profile HMM
    2. Search database for homologous sequences
    3. Align matching sequences
    4. Iterate to find more distant homologs
    5. Output: multiple sequence alignment (MSA) in Stockholm format
    
    The MSA provides evolutionary signal: conserved residues appear in most homologs.
    This conservation information correlates with functional importance.
    
    Configuration:
        --cpu 2: Use 2 CPU threads (configurable via external parameter)
        -A: Output alignment in Stockholm format
    
    Args:
        sequence_fasta (str): Path to FASTA file with query sequence
        database_path (str): Path to sequence database (e.g., UniProt/Swiss-Prot)
        output_aln (str): Path where Stockholm alignment output will be saved
    
    Returns:
        str: Path to output alignment file if successful
        None: If Jackhmmer execution fails
    
    Dependencies:
        • Jackhmmer must be installed and in PATH
        • Database file must be HMMER-formatted (indexed)
    
    """
    print(f"--- Running Jackhmmer Evolutionary Analysis ---")
    # Command: jackhmmer [options] <seqfile> <seqdb>
    # -A: output alignment to file
    # --cpu: number of parallel threads
    command = ["jackhmmer", "--cpu", "2", "-A", output_aln, sequence_fasta, database_path]
    try:
        # Execute command, suppress stdout/stderr (avoid cluttering output)
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_aln
    except Exception as e:
        print(f"Error running Jackhmmer: {e}")
        return None


def calculate_conservation_scores(alignment_file: str, protein_atoms: list) -> dict:
    """
    Calculate residue-level conservation scores from Jackhmmer MSA.
    
    Conservation Score Definition:
        For each position in the MSA:
            conservation = (frequency of most common residue) / (total sequences)
        
        Example: At position 42, alignment shows:
            Seq1: A, Seq2: A, Seq3: A, Seq4: G (3 match out of 4)
            Score = 3/4 = 0.75 (75% conserved)
    
    High Conservation Meaning:
        • Residue type preserved across evolution
        • Suggests functional importance
        • Likely involved in catalysis or binding
        • False positives rare (evolutionary filtering is stringent)
    
    Mapping to PDB Residues:
        MSA columns correspond to sequence positions.
        Sequence positions map to PDB residue numbers (via atom order).
        Output: dict {(chain_id, residue_number): conservation_score}
    
    Args:
        alignment_file (str): Path to Stockholm format alignment (from Jackhmmer)
        protein_atoms (list): List of Bio.PDB.Atom objects (defines sequence order)
    
    Returns:
        dict: Maps residue identifier → conservation score (0.0-1.0)
              Key format: f"{chain_id}_{residue_seq_number}"
              Value: conservation score (0.0 = completely variable, 1.0 = completely conserved)
    
    Notes:
        • Gaps/missing residues in alignment handled gracefully
        • Positions at sequence boundaries may have fewer sequences (less reliable)
        • Conservation correlates but doesn't perfectly predict binding site residues
     """
    # Read Stockholm format alignment generated by Jackhmmer
    alignment = AlignIO.read(alignment_file, "stockholm")
    num_seqs = len(alignment)  # Total number of sequences in MSA
    
    # Calculate conservation score for each column (position) in alignment
    col_scores = []
    for i in range(alignment.get_alignment_length()):
        # Get residues at this position, excluding gaps and unknown characters
        residues = [r.upper() for r in alignment[:, i] if r not in ('-', '.')]
        
        # Conservation = most common residue frequency / total sequences
        # If no residues at position (all gaps), score = 0.0
        score = Counter(residues).most_common(1)[0][1] / num_seqs if residues else 0.0
        col_scores.append(score)
    
    # Map MSA positions back to PDB residue identifiers
    # Extract unique (chain, residue_number) pairs from atoms in order
    pdb_res_keys = []
    for atom in protein_atoms:
        res = atom.get_parent()
        # Unique identifier: chain_id + residue number
        key = f"{res.get_parent().id}_{res.id[1]}"
        # Avoid duplicates (each residue has multiple atoms, only count once)
        if not pdb_res_keys or pdb_res_keys[-1] != key:
            pdb_res_keys.append(key)
    
    # Create mapping: PDB residue → conservation score from MSA column
    # Handle case where MSA might be shorter than PDB sequence (rare)
    return {pdb_res_keys[i]: score for i, score in enumerate(col_scores) if i < len(pdb_res_keys)}

 
 # 3. MASTER RANKING AND PROFILING
 
def rank_pockets_master_score(pockets_dict: dict, protein_atoms: list, conservation_dict: dict = None) -> list:
    """
    Unified ranking system integrating geometry, conservation, and chemistry.
    
    SCORING FORMULA:
    
    final_score = Size_Score + Hydrophobicity_Score + Conservation_Score
    
    Where:
    • Size_Score: min(20, len(points)/50)
            - Maximum 20 points for large pockets
            - Scales: 50 points = 1 score, 500 points = 10 score, 1000+ points = 20 score
            - Rationale: Bigger pockets = more volume for ligand binding
    
    • Hydrophobicity_Score: min(30, (avg_hydro + 4.5)*3)
            - Average KD hydrophobicity scaled to 0-30 point range
            - KD range: -4.5 (most hydrophilic) to +4.5 (most hydrophobic)
            - Formula: (value + 4.5) shifts to 0-9 range, ×3 gives 0-27, capped at 30
            - Rationale: Hydrophobic pockets preferred for drug binding (most ligands are hydrophobic)
    
    • Conservation_Score: avg_cons * 50
            - Conservation ranges 0.0-1.0
            - Score ranges 0-50 points
            - Rationale: Evolutionarily conserved residues → functionally important
    
    Total Range: 0-100 scale (theoretically)
    Typical Values:
        • 80+: Excellent pocket (large, hydrophobic, conserved)
        • 60-80: Good pocket (2 out of 3 properties strong)
        • 40-60: Moderate pocket (average quality)
        • <40: Poor pocket (small or unfavorable chemistry)
    
    CHEMICAL PROFILING:
   
    Residue Classification:
        HYDROPHOBIC: ALA, VAL, ILE, LEU, MET, PHE, TYR, TRP, PRO
                    (prefer nonpolar, lipophilic ligands)
        POLAR: SER, THR, ASN, GLN, CYS
               (can form H-bonds, accept polar ligands)
        POSITIVE: ARG, LYS, HIS
                 (attract anionic/acidic ligands)
        NEGATIVE: ASP, GLU
                 (attract cationic/basic ligands)
    
    Ligand Preference Heuristics:
        • If >50% hydrophobic residues: "Lipophilic/Hydrophobic"
          → Likely binds fatty acids, steroids, aromatic drugs
        • Else if (positive > negative): "Anionic (Acidic) Ligands"
          → Likely binds acidic molecules, carboxyl-containing ligands
        • Else: "Mixed/Polar"
          → Diverse ligand types, peptides, cofactors, nucleotides
    
    Args:
        pockets_dict (dict): Maps pocket_id → point coordinates (from cluster_pocket_points)
        protein_atoms (list): List of Bio.PDB.Atom objects
        conservation_dict (dict, optional): Maps residue_key → conservation_score (from calculate_conservation_scores)
                                          If None, conservation weighting is skipped
    
    Returns:
        list: Sorted list of dicts (highest score first).
              Each dict contains:
              - 'id': pocket identifier (int)
              - 'score': master score 0-100 (float)
              - 'size': number of grid points (int)
              - 'preference': predicted ligand type (str)
              - 'composition': dict with counts {hydrophobic, polar, positive, negative, special}
              - 'residues': list of nearby residue identifiers (str list)
     """
    # Extract atomic coordinates and build spatial index for efficient queries
    atom_coords = np.array([atom.get_coord() for atom in protein_atoms])
    tree = KDTree(atom_coords)
    results = []
    
    # Define residue classes for chemical profiling
    # These groupings are empirically validated and widely used in biochemistry
    HYDROPHOBIC = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP', 'PRO'}
    POLAR = {'SER', 'THR', 'ASN', 'GLN', 'CYS'}
    POS = {'ARG', 'LYS', 'HIS'}  # Positively charged/ionizable
    NEG = {'ASP', 'GLU'}  # Negatively charged
    
    # Process each pocket
    for p_id, points in pockets_dict.items():
        # Find all protein residues within 4.5 Å of ANY pocket point
        # 4.5 Å is standard van der Waals distance + hydration shell
        neighbor_indices = tree.query_ball_point(points, 4.5)
        # Flatten: combine all neighbors from all pocket points, remove duplicates
        flat_indices = set(idx for sublist in neighbor_indices for idx in sublist)
        # Extract unique residues (multiple atoms per residue)
        residues = set(protein_atoms[idx].get_parent() for idx in flat_indices)
        
        # Calculate average hydrophobicity (KD scale)
        # Higher value = more hydrophobic = prefers lipophilic ligands
        avg_hydro = np.mean([KD_SCALE.get(r.get_resname(), 0.0) for r in residues]) if residues else 0.0
        
        # Calculate average conservation if conservation data provided
        # Higher value = more evolutionarily conserved = likely functionally important
        avg_cons = np.mean([conservation_dict.get(f"{r.get_parent().id}_{r.id[1]}", 0.0) for r in residues]) if conservation_dict and residues else 0.0
        
        # Count residues by chemical class (for profiling and preference determination)
        comp = {'hydrophobic': 0, 'polar': 0, 'positive': 0, 'negative': 0, 'special': 0}
        res_strings = []  # Human-readable residue list for reports
        
        for r in residues:
            n = r.get_resname()
            res_strings.append(f"{n}{r.id[1]}")
            
            # Classify residue (note: some residues could fit multiple categories,
            # but we assign each to only one for clear profiling)
            if n in HYDROPHOBIC:
                comp['hydrophobic'] += 1
            elif n in POLAR:
                comp['polar'] += 1
            elif n in POS:
                comp['positive'] += 1
            elif n in NEG:
                comp['negative'] += 1
            else:
                comp['special'] += 1
        
        # Determine ligand preference based on pocket composition
        # Heuristic rules derived from binding site analysis of known complexes
        if comp['hydrophobic'] > len(residues) * 0.5:
            # >50% hydrophobic → strong lipophilic character
            preference = "Lipophilic/Hydrophobic"
        elif comp['positive'] > comp['negative']:
            # More positive than negative → attracts acidic ligands
            preference = "Anionic (Acidic) Ligands"
        else:
            # Default for mixed/neutral pockets
            preference = "Mixed/Polar"

        # MASTER SCORE CALCULATION
        # Weighted sum of three independent quality metrics
        # Size: max 20 points (larger = better for ligand accommodation)
        # Hydrophobicity: max 30 points (hydrophobic = better for drugs)
        # Conservation: max 50 points (evolutionarily conserved = functionally important)
        final_score = (min(20, len(points)/50)) + (min(30, (avg_hydro + 4.5)*3)) + (avg_cons * 50)
        
        # Store result
        results.append({
            'id': p_id,
            'score': round(final_score, 1),
            'size': len(points),
            'preference': preference,
            'composition': comp,
            'residues': sorted(res_strings)
        })
    
    # Sort by score (highest first = best binding sites)
    return sorted(results, key=lambda x: x['score'], reverse=True)


def save_pocket_ranking_to_file(ranked_pockets: list, filename="pocket_ranking.txt"):
    """
    Export detailed human-readable report of pocket rankings with ligand type predictions.
    
    Report Format:
   
    For each pocket:
        • Rank and pocket ID (top 3 marked with ***)
        • Master score (0-100 scale)
        • Volume in grid points
        • Chemical profile (hydrophobic/polar/charged residues)
        • Ligand type prediction (what kind of molecule likely to bind)
        • List of nearby residues
    
    Ligand Type Predictions:
    
    Based on pocket composition:
    
    "Lipophilic/Hydrophobic":
        → Likely binds: fatty acids, retinoids, steroids, aromatic compounds,
                       lipid-like molecules, nonpolar drugs
        → Examples: estrogen, testosterone, warfarin, ibuprofen
    
    "Anionic (Acidic) Ligands":
        → Likely binds: carboxyl-containing molecules, phosphorylated compounds,
                       sulfated molecules, nucleotides, acidic amino acids
        → Examples: ATP, GTP, heparin, substrate analogs
    
    "Mixed/Polar":
        → Likely binds: diverse molecules - antibiotics, cofactors, nucleotides,
                       peptides, polar pharmaceuticals, water-soluble drugs
        → Examples: penicillin, NAD+, glucose, ethanol
    
    Args:
        ranked_pockets (list): Output from rank_pockets_master_score (sorted by score)
        filename (str, optional): Output text file path. Default: "pocket_ranking.txt"
    
    """
    
    # Map pocket preferences to predicted ligand types
    # Based on biochemical knowledge of binding site properties
    ligand_type_map = {
        "Lipophilic/Hydrophobic": (
            "Hydrophobic/lipophilic ligands (e.g., fatty acids, retinoids, "
            "steroids, aromatic compounds, lipid-like molecules)"
        ),
        "Anionic (Acidic) Ligands": (
            "Cationic/positively charged ligands (e.g., arginine-rich inhibitors, "
            "positively charged pharmaceuticals, basic amino acids)"
        ),
        "Mixed/Polar": (
            "Diverse ligands with mixed character: antibiotics, cofactors, "
            "nucleotides, peptides, polar pharmaceuticals, water-soluble drugs"
        )
    }
    
    with open(filename, "w") as f:
        f.write("="*80 + "\n")
        f.write("POCKET FINDER: FINAL BINDING SITE RANKING\n")
        f.write("="*80 + "\n\n")
        
        for i, p in enumerate(ranked_pockets):
            # Rank number with *** for top 3
            f.write(f"RANK {i+1} ({'*'*3 if i < 3 else '   '}) | ")
            f.write(f"POCKET {p['id']:<2d} | MASTER SCORE: {p['score']:<6.1f}\n")
            f.write("-"*80 + "\n")
            
            # Basic metrics
            f.write(f"Volume (size):     {p['size']:4d} grid points\n")
            f.write(f"Chemical Profile:  {p['preference']}\n")
            
            # Get detailed ligand type description from mapping
            ligand_type = ligand_type_map.get(
                p['preference'],
                "Unknown ligand type"
            )
            f.write(f"Expected Ligands:  {ligand_type}\n")
            
            # List nearby residues (useful for mutation studies, docking)
            f.write(f"Nearby Residues:   {', '.join(p['residues'])}\n")
            
            # Show detailed composition breakdown
            comp = p.get('composition', {})
            if comp:
                f.write(f"Composition:       {comp.get('hydrophobic', 0)} hydrophobic, "
                       f"{comp.get('polar', 0)} polar, "
                       f"{comp.get('positive', 0)} positive, "
                       f"{comp.get('negative', 0)} negative\n")
            
            f.write("\n")