#!/usr/bin/env python3

"""
PocketFinder Pipeline
---------------------
An automated bioinformatics tool for the structural and evolutionary 
analysis of protein binding sites. 

The pipeline performs:
1. Structural preprocessing and noise reduction.
2. Evolutionary conservation profiling via local HMMER/Jackhmmer.
3. Density-based geometric pocket detection (DBSCAN).
4. Biochemical ranking and ligand-preference prediction.
"""

import sys
import os
import bio_pockets  # The core package containing data, utils, and analysis logic

def main():
    # --- 1. Environment Setup & Argument Parsing ---
    if len(sys.argv) < 2:
        print("\n[!] Error: Input PDB file is required.")
        print("Usage: python pocket_finder.py <structure.pdb> [database.fasta]")
        sys.exit(1)

    # Core Input Parameters
    FILE_PATH = sys.argv[1]
    # Default database path for evolutionary search if not specified
    DB_PATH = sys.argv[2] if len(sys.argv) > 2 else "uniprot_sprot.fasta"

    if not os.path.exists(FILE_PATH):
        print(f"[!] Error: Path '{FILE_PATH}' does not exist.")
        sys.exit(1)

    # File Naming Convention for Pipeline Output
    base_name = os.path.splitext(os.path.basename(FILE_PATH))[0]
    clean_pdb = f"{base_name}_cleaned.pdb"
    query_fasta = f"{base_name}_query.fasta"
    aln_sto = f"{base_name}_alignment.sto"
    output_pdb = f"{base_name}_results.pdb"
    ranking_report = f"{base_name}_ranking.txt"

    print(f"\n" + "="*60)
    print(f"POCKETFINDER ANALYSIS: {base_name.upper()}")
    print(f"DATABASE SOURCE: {DB_PATH}")
    print("="*60 + "\n")

    # --- 2. Structural Preprocessing ---
    print("[1/4] Preprocessing structure (cleaning water/heteroatoms)...")
    structure, _ = bio_pockets.get_protein_structure(FILE_PATH)
    bio_pockets.save_clean_protein(structure, clean_pdb)
    
    # Reload coordinates to ensure all downstream math uses the cleaned model
    _, clean_atoms = bio_pockets.get_protein_structure(clean_pdb)

    # --- 3. Evolutionary Conservation Profiling ---
    print("[2/4] Generating Multiple Sequence Alignment (MSA)...")
    bio_pockets.extract_sequence_from_pdb(clean_atoms, query_fasta)
    
    cons_scores = None
    if os.path.exists(DB_PATH):
        # run_jackhmmer_alignment triggers the local HMMER suite
        if bio_pockets.run_jackhmmer_alignment(query_fasta, DB_PATH, aln_sto):
            cons_scores = bio_pockets.calculate_conservation_scores(aln_sto, clean_atoms)
            print("  -> MSA successfully generated. Conservation scores mapped.")
    else:
        print(f"  [!] Warning: '{DB_PATH}' not found. Skipping conservation analysis.")

    # --- 4. Geometric Cavity Detection ---
    print("[3/4] Scanning 3D space for geometric cavities...")
    grid = bio_pockets.create_search_grid(clean_atoms, spacing=1.0)
    candidates = bio_pockets.find_pocket_points(grid, clean_atoms)
    clustered_pockets = bio_pockets.cluster_pocket_points(candidates)
    
    # --- 5. Integrated Ranking & Biochemical Profiling ---
    print("[4/4] Calculating master scores and chemical preferences...")
    # Combines structural volume, hydrophobicity, and evolutionary signal
    ranking_data = bio_pockets.rank_pockets_master_score(clustered_pockets, clean_atoms, cons_scores)

    # --- 6. Final Reporting & Data Export ---
    print("\n" + "-"*40)
    print(f"{'Rank':<5} | {'Pocket ID':<10} | {'Score':<8}")
    print("-"*40)

    if ranking_data:
        for i, p in enumerate(ranking_data):
            # Print only the essential data to the terminal
            print(f"{i+1:<5} | {p['id']:<10} | {p['score']:<8}")
        
        # --- THE EXPORTS STILL CONTAIN EVERYTHING ---
        # This function still writes the full details (Preference/Residues) to the text file
        bio_pockets.save_pocket_ranking_to_file(ranking_data, ranking_report)
        
        # This still saves the colored chains for PyMOL/Chimera
        bio_pockets.save_protein_with_colored_pockets(clean_atoms, clustered_pockets, output_pdb)

        print("-"*40)
        print(f"\n[SUMMARY] Analysis finalized.")
        print(f"  > Structural Visualization: {output_pdb}")
        print(f"  > Detailed Technical Report: {ranking_report}")
    else:
        print("No significant pockets detected.")
    
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
