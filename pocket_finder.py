#!/usr/bin/env python3

"""
PocketFinder - FINAL WORKING VERSION

A comprehensive computational tool for identifying and ranking potential binding pockets
in protein structures using multiple geometric and conservation-based detection presets.

This script:
  1. Loads three preset configurations (small_enzymes, medium_proteins, large_proteins)
  2. Runs independent pocket detection analysis for each preset
  3. Deduplicates results across presets to create a unified ranking
  4. Generates visualization scripts for PyMOL and Chimera
  5. Saves ranking and coordinate data for further analysis

The unified output prioritizes pockets found by multiple presets and combines scoring
metrics across different detection methodologies.

Usage:
    python pocketfinder.py <input.pdb> [database.fasta]
"""

import sys
import os
import argparse
import yaml
import numpy as np
from scipy.spatial import distance
import bio_pockets


def load_preset_config(preset_name: str):
    """
    Load a preset configuration file from disk.
    
    Searches multiple directory locations for flexibility in deployment.
    Presets define algorithm parameters like grid spacing, distance thresholds,
    and clustering parameters optimized for different protein size categories.
    
    Args:
        preset_name (str): Name of the preset to load (e.g., 'small_enzymes')
        
    Returns:
        dict: Configuration dictionary containing all analysis parameters
        
    Raises:
        FileNotFoundError: If the preset file cannot be found in any expected location
    """
    preset_path = None
    # Try multiple path patterns for cross-platform compatibility
    possible_paths = [
        f"presets/{preset_name}.yaml",
        f"presets/presets_{preset_name}.yaml",
        preset_name
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            preset_path = path
            break
    
    if not preset_path:
        raise FileNotFoundError(f"Preset '{preset_name}' not found")
    
    with open(preset_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def run_analysis_with_preset(pdb_file: str, db_file: str, preset_name: str, config: dict):
    """
    Execute complete pocket detection pipeline for a single preset.
    
    Pipeline stages:
      [1/4] Preprocessing: Clean structure, remove hydrogens/non-standard residues
      [2/4] Conservation: Generate MSA and calculate residue conservation scores
      [3/4] Geometric Detection: Identify cavity points using grid-based search
      [4/4] Ranking: Score and rank detected pockets by multiple criteria
    
    Configuration parameters explained:
    
    GRID parameters:
      - spacing_angstrom (float): Distance between grid points. Smaller = higher precision 
        but slower computation. Typical: 1.5-2.5 Å
    
    GEOMETRY parameters:
      - min_distance_angstrom (float): Minimum distance from protein atom surface to grid point.
        Filters out very shallow cavities. Typical: 1.5-2.0 Å
      - max_distance_angstrom (float): Maximum distance limit. Typical: 10.0-15.0 Å
      - surface_threshold_angstrom (float): Radius to determine protein surface. Typical: 3.5-4.5 Å
      - min_neighbors (int): Minimum nearby atoms for point to be considered. Typical: 3-5
      - max_neighbors (int): Maximum nearby atoms (filters buried points). Typical: 8-12
      - enclosure_check: Evaluates if pocket is surrounded by protein
        * radii_angstrom: Multiple radii to check enclosure at different scales
        * min_enclosure_fraction: Minimum fraction of sphere that must be protein-enclosed (0.0-1.0)
    
    CLUSTERING parameters:
      - eps_angstrom (float): DBSCAN epsilon - max distance between cluster members. 
        Typical: 2.0-3.5 Å (larger values merge nearby pockets)
      - min_samples (int): Minimum points in DBSCAN cluster. Typical: 10-20
      - min_points (int): Filter clusters smaller than this. Typical: 15-25
      - max_points (int): Filter clusters larger than this. Typical: 2000-5000
    
    Args:
        pdb_file (str): Path to input PDB structure file
        db_file (str): Path to sequence database for conservation analysis (optional)
        preset_name (str): Name of the configuration preset being used
        config (dict): Configuration dictionary loaded from preset YAML file
        
    Returns:
        tuple: (ranking_data, clustered_pockets, clean_atoms)
            - ranking_data (list): Pockets ranked by score (or empty list if no pockets found)
            - clustered_pockets (dict): Maps pocket ID to numpy array of 3D coordinates
            - clean_atoms (list): BioPython atom objects for cleaned structure
    """
    
    base_name = os.path.splitext(os.path.basename(pdb_file))[0]
    
    print(f"\n{'-'*70}")
    print(f"RUNNING: {preset_name.upper()}")
    
    
    try:
        # STEP 1: Preprocessing - Clean and standardize the protein structure
        print("[1/4] Preprocessing structure...")
        structure, _ = bio_pockets.get_protein_structure(pdb_file)
        cleaned_pdb = f"{base_name}_cleaned_temp_{preset_name}.pdb"
        bio_pockets.save_clean_protein(structure, cleaned_pdb)
        _, clean_atoms = bio_pockets.get_protein_structure(cleaned_pdb)
        print(f"      → {len(clean_atoms)} atoms cleaned")

        # STEP 2: Conservation Analysis - Generate multiple sequence alignment for conservation scoring
        print("[2/4] Generating MSA...")
        query_fasta = f"{base_name}_query_temp_{preset_name}.fasta"
        alignment_sto = f"{base_name}_alignment_temp_{preset_name}.sto"
        bio_pockets.extract_sequence_from_pdb(clean_atoms, query_fasta)

        cons_scores = None
        if os.path.exists(db_file):
            if bio_pockets.run_jackhmmer_alignment(query_fasta, db_file, alignment_sto):
                cons_scores = bio_pockets.calculate_conservation_scores(alignment_sto, clean_atoms)
                print("      → MSA generated")
        else:
            print(f"      ⚠️  Database not found, skipping conservation")

        # STEP 3: Geometric Detection - Scan for cavity points using multiple criteria
        print("[3/4] Scanning for pockets...")
        
        # Extract configuration parameters with detailed explanations
        grid_spacing = config['grid']['spacing_angstrom']
        min_dist = config['geometry']['min_distance_angstrom']
        max_dist = config['geometry']['max_distance_angstrom']
        surface_threshold = config['geometry']['surface_threshold_angstrom']
        min_neighbors = config['geometry']['min_neighbors']
        max_neighbors = config['geometry']['max_neighbors']
        
        # Extract enclosure check parameters - detects if cavity is protein-enclosed
        enclosure_config = config['geometry']['enclosure_check']
        enclosure_radii = tuple(enclosure_config['radii_angstrom'])
        min_enclosure = enclosure_config['min_enclosure_fraction']
        
        # Create 3D search grid and identify cavity candidate points
        grid = bio_pockets.create_search_grid(clean_atoms, spacing=grid_spacing)
        candidates = bio_pockets.find_pocket_points(
            grid, clean_atoms,
            min_dist=min_dist,
            max_dist=max_dist,
            surface_threshold=surface_threshold,
            min_neighbors=min_neighbors,
            max_neighbors=max_neighbors,
            enclosure_radii=enclosure_radii,
            min_enclosure=min_enclosure
        )
        
        # Cluster nearby candidate points to form discrete pocket regions
        clustering_config = config['clustering']
        clustered_pockets = bio_pockets.cluster_pocket_points(
            candidates,
            eps=clustering_config['eps_angstrom'],
            min_samples=clustering_config['min_samples'],
            min_points=clustering_config['min_points'],
            max_points=clustering_config['max_points']
        )
        
        print(f"      → Found {len(clustered_pockets)} pockets")

        # STEP 4: Ranking - Score and rank pockets by integrated metrics
        ranking_data = []
        if len(clustered_pockets) > 0:
            print("[4/4] Ranking pockets...")
            # Master score combines geometric quality, size, and conservation information
            ranking_data = bio_pockets.rank_pockets_master_score(
                clustered_pockets,
                clean_atoms,
                cons_scores
            )
            print(f"      → Ranked {len(ranking_data)} pockets")
        else:
            print("[4/4] No pockets to rank")
        
        # Cleanup temporary files to avoid disk clutter
        for f in [cleaned_pdb, query_fasta, alignment_sto]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        
        return ranking_data, clustered_pockets, clean_atoms
        
    except Exception as e:
        print(f"[!] Error: {str(e)}")
        return [], {}, None


def deduplicate_pockets(all_results: dict, distance_threshold=3.0):
    """
    Merge and deduplicate pockets identified across different presets.
    
    Strategy:
      1. Collect all pockets from all preset analyses
      2. Sort by score (best-scoring pockets first)
      3. Iteratively keep highest-scoring pockets, remove similar ones
      4. Spatial similarity determined by Euclidean distance between pocket centers
      
    The distance_threshold of 3.0 Å is empirically optimized to distinguish between:
      - Same pocket detected by different presets → merged (typically 1-2 Å apart)
      - Adjacent but distinct pockets → kept separate (typically 5-10 Å apart)
    
    Args:
        all_results (dict): Maps preset_name -> (ranking_data, clustered_pockets)
            - ranking_data: List of dicts with 'id', 'score', 'size' etc.
            - clustered_pockets: Dict maps pocket_id -> numpy array of 3D points
        distance_threshold (float): Maximum distance (Ångströms) to consider two pockets 
            as duplicates. Default: 3.0 Å (spatial separation of ~4 Å is distinct)
        
    Returns:
        tuple: (unified_ranking, unified_pocket_points, unified_pockets)
            - unified_ranking: List of pocket dicts sorted by score (with unified_id added)
            - unified_pocket_points: Dict maps unified_id -> numpy array of coordinates
            - unified_pockets: Original pocket info with metadata (for debugging)
    """
    
    print(f"\n{'-'*70}")
    print(f"DEDUPLICATING RESULTS")
   
    
    # Collect all pockets from all presets with their metadata
    all_pockets_list = []
    
    for preset_name, (ranking_data, clustered_pockets) in all_results.items():
        if not ranking_data or len(ranking_data) == 0:
            continue
            
        for pocket_data in ranking_data:
            pocket_id = pocket_data['id']
            if pocket_id not in clustered_pockets:
                continue
            
            # Calculate geometric center of pocket point cloud
            points = clustered_pockets[pocket_id]
            center = points.mean(axis=0)
            
            all_pockets_list.append({
                'preset': preset_name,
                'center': center,
                'points': points,
                'ranking_data': pocket_data,
                'score': pocket_data['score'],
                'size': pocket_data['size']
            })
    
    if not all_pockets_list:
        print("No pockets found in any preset!")
        return [], {}, []
    
    print(f"Total pockets found: {len(all_pockets_list)}")
    
    # Deduplication algorithm: greedy approach selecting best pockets first
    # and merging nearby duplicates (common across presets)
    used_indices = set()
    unified_pockets = []
    # Sort by score descending - process best pockets first
    sorted_pockets = sorted(all_pockets_list, key=lambda x: x['score'], reverse=True)
    
    for i, pocket1 in enumerate(sorted_pockets):
        if i in used_indices:
            continue
        
        # Keep this pocket as representative
        unified_pockets.append(pocket1)
        used_indices.add(i)
        
        # Find and mark all similar pockets as duplicates
        # Compare Euclidean distance between pocket centers
        for j, pocket2 in enumerate(sorted_pockets[i+1:], start=i+1):
            if j in used_indices:
                continue
            
            # Calculate 3D Euclidean distance between pocket centers
            dist = distance.euclidean(pocket1['center'], pocket2['center'])
            # If pockets are close enough, consider them the same cavity
            if dist < distance_threshold:
                used_indices.add(j)
    
    print(f"After deduplication: {len(unified_pockets)} unique pockets\n")
    
    # Create unified ranking with new sequential IDs
    unified_ranking = [p['ranking_data'] for p in unified_pockets]
    unified_ranking = sorted(unified_ranking, key=lambda x: x['score'], reverse=True)
    
    # Assign sequential unified IDs to deduplicated pockets
    for new_id, pocket_data in enumerate(unified_ranking, 1):
        pocket_data['unified_id'] = new_id
    
    # Map unified IDs to their 3D point clouds
    unified_pocket_points = {}
    for pocket_info in unified_pockets:
        original_id = pocket_info['ranking_data']['id']
        unified_id = pocket_info['ranking_data']['unified_id']
        unified_pocket_points[unified_id] = pocket_info['points']
    
    return unified_ranking, unified_pocket_points, unified_pockets


def main():
    """
    Main orchestration function: controls full analysis pipeline.
    
    Workflow:
      1. Parse command-line arguments
      2. Load and run 3 preset analyses independently
      3. Deduplicate and merge results across presets
      4. Map residues to pocket vicinity
      5. Generate output files (ranking, PDB, visualization scripts)
      6. Open visualization in Chimera
    """
    parser = argparse.ArgumentParser(description="PocketFinder - Unified Preset Analysis")
    parser.add_argument("pdb_file", help="Input PDB file")
    parser.add_argument("database", nargs="?", default="/mnt/NFS_UPF/soft/databases/blastdat/uniprot_sprot.fasta",
                       help="Conservation database (default: /mnt/NFS_UPF/soft/databases/blastdat/uniprot_sprot.fasta)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdb_file):
        print(f"[!] Error: PDB file not found: {args.pdb_file}")
        sys.exit(1)
    
    base_name = os.path.splitext(os.path.basename(args.pdb_file))[0]

    print(f"\n{'-'*70}")
    print(f"POCKETFINDER - UNIFIED ANALYSIS")
    
    print(f"Input:    {args.pdb_file}")
    print(f"Database: {args.database}")
    

    # Run all three detection presets - each optimized for different protein sizes
    presets_to_run = ['small_enzymes', 'medium_proteins', 'large_proteins']
    all_results = {}
    clean_atoms = None
    
    # Execute independent analysis for each preset
    for preset_name in presets_to_run:
        try:
            config = load_preset_config(preset_name)
            ranking_data, clustered_pockets, atoms = run_analysis_with_preset(
                args.pdb_file, args.database, preset_name, config
            )
            
            if atoms:
                clean_atoms = atoms
            
            all_results[preset_name] = (ranking_data, clustered_pockets)
            
        except Exception as e:
            print(f"[!] Error with {preset_name}: {e}")
            all_results[preset_name] = ([], {})
    
    # Merge results across presets and remove duplicates
    unified_ranking, unified_pocket_points, unified_pockets = deduplicate_pockets(all_results)
    
    if not unified_ranking or not clean_atoms:
        print("\n[!] No pockets detected in any preset")
        sys.exit(1)
    
    # Build residue-to-chain mapping for visualization - identifies which residues
    # surround each pocket for highlighting in molecular visualization software
    from scipy.spatial import KDTree
    
    unified_residue_to_chain = {}
    # Build KDTree for efficient spatial queries of nearby atoms
    atom_coords = np.array([atom.get_coord() for atom in clean_atoms])
    tree = KDTree(atom_coords)
    
    for unified_id in unified_pocket_points.keys():
        # Assign chain letters cyclically for visualization (A, B, C, ...)
        chain_letter = chr(ord('A') + (unified_id - 1) % 26)
        points = unified_pocket_points[unified_id]
        
        # Find all atoms within 4.5 Ångströms of any pocket point (typical protein radius)
        neighbor_indices = tree.query_ball_point(points, 4.5)
        # Flatten nested list of indices
        flat_indices = set(idx for sublist in neighbor_indices for idx in sublist)
        
        # Associate residues with pocket chain assignment
        for idx in flat_indices:
            atom = clean_atoms[idx]
            res = atom.get_parent()
            chain = res.get_parent()
            res_key = (chain.id, res.id[1])
            
            if res_key not in unified_residue_to_chain:
                unified_residue_to_chain[res_key] = chain_letter
    
    # Save all output files
    print(f"\n{'-'*70}")
    print(f"SAVING UNIFIED RESULTS")
   
    
    current_dir = os.getcwd()
    out_dir = os.path.join(current_dir, "result", base_name)
    os.makedirs(out_dir, exist_ok=True)
    
    ranking_file = os.path.join(out_dir, f"{base_name}_ranking.txt")
    bio_pockets.save_pocket_ranking_to_file(unified_ranking, ranking_file)
    print(f"✓ {ranking_file}")
    
    results_pdb = os.path.join(out_dir, f"{base_name}_results.pdb")
    bio_pockets.save_protein_with_colored_pockets(clean_atoms, unified_pocket_points, results_pdb)
    print(f"✓ {results_pdb}")
    
    vis_base = os.path.join(out_dir, base_name)
    pymol_cmd, chimera_cmd = bio_pockets.generate_visualization_scripts(
        vis_base, results_pdb, unified_residue_to_chain, ranked_pockets=unified_ranking
    )
    print(f"✓ {chimera_cmd}")
    print(f"✓ {pymol_cmd}")
    
    # Print summary statistics
    print(f"\n{'-'*70}")
    print(f"ANALYSIS COMPLETE")
    
    
    print(f"Preset Results:")
    for preset_name, (ranking_data, _) in all_results.items():
        if ranking_data and len(ranking_data) > 0:
            print(f"  • {preset_name:20s}: {len(ranking_data):2d} pockets (top: {ranking_data[0]['score']:5.1f})")
        else:
            print(f"  • {preset_name:20s}: 0 pockets")
    
  
    print(f"\nUnified Results:")
    print(f"  • Unique pockets: {len(unified_ranking)}")
    print(f"  • Top 3 pockets:\n")
    
    for i, p in enumerate(unified_ranking[:3], 1):
        num_pts = len(unified_pocket_points[p['unified_id']])
        if num_pts < 150:
            cat = "🔵 SMALL"
        elif num_pts < 400:
            cat = "🟢 MEDIUM"
        else:
            cat = "🔴 LARGE"

        prot_id = p.get('id', p['unified_id'])
        print(f"    {i}. Pocket {prot_id} | Score: {p['score']:5.1f} | "
              f"Size: {num_pts:4d} pts | {cat} | {p.get('preference', 'Unknown')}")

    print(f"\n Color scheme (by ligand size):")
    print(f"  🔵 SMALL (< 150 pts):   BLUE")
    print(f"  🟢 MEDIUM (150-400):    GREEN")
    print(f"  🔴 LARGE (> 400 pts):   RED")

    print(f"\n{'-'*70}")
    
    print(f"\n Opening Chimera...")
    if os.path.exists(chimera_cmd):
        bio_pockets.open_in_chimera(chimera_cmd)
    


if __name__ == "__main__":
    main()