import sys
import os
import myproject  

def main():
    # 1. Check if the user provided a file in the terminal
    if len(sys.argv) < 2:
        print("Error: Please provide a PDB file.")
        print("Usage: python pocket_finder.py <your_file.pdb>")
        sys.exit(1)

    # The first argument after the script name is the PDB file
    FILE_PATH = sys.argv[1]

    # Check if the file actually exists
    if not os.path.exists(FILE_PATH):
        print(f"Error: The file '{FILE_PATH}' was not found.")
        sys.exit(1)

    # 2. Automatic name generation
    base_name = os.path.splitext(FILE_PATH)[0]
    clean_file = f"{base_name}_cleaned.pdb"
    output_file = f"{base_name}_with_pockets.pdb"

    print(f"\nStarting analysis for: {FILE_PATH} ...")

    # 3. Load and Clean
    structure, atoms = myproject.get_protein_structure(FILE_PATH)
    myproject.save_clean_protein(structure, clean_file, protein_chains=None)
    clean_structure, clean_atoms = myproject.get_protein_structure(clean_file)

    # 4. Geometry & Clustering
    print("Searching for pockets and calculating clusters...")
    grid = myproject.create_search_grid(clean_atoms, spacing=1.0)
    candidates = myproject.find_pocket_points(grid, clean_atoms)
    all_pockets = myproject.cluster_pocket_points(candidates)
    
    # 5. Score & Export
    print("Calculating Master Scores...\n")
    # This now returns a LIST of dictionaries
    ranking_data = myproject.rank_pockets_master_score(all_pockets, clean_atoms)

    # Output directly in the terminal
    print("RESULTS")
    print(f"{'Rank':<5} | {'Pocket ID':<10} | {'Score':<8} | {'Size':<6} | {'Preference'}")
    print("-" * 75)
    
    # FIX: Check if list is not empty
    if ranking_data:
        # Loop through the list to print a clean table
        for i, p in enumerate(ranking_data):
            print(f"{i+1:<5} | {p['id']:<10} | {p['score']:<8} | {p['size']:<6} | {p['preference']}")
    else:
        print("No pockets found.")
    print("-" * 75)

    # 6. Save results
    # Save the PDB file for Chimera/PyMOL
    myproject.save_protein_with_colored_pockets(clean_atoms, all_pockets, output_file)
    
    # FIX: Save the detailed text ranking report
    myproject.save_pocket_ranking_to_file(ranking_data, "pocket_ranking.txt")
    
    print(f"\nDone! Results were saved as '{output_file}' and 'pocket_ranking.txt'.\n")

if __name__ == "__main__":
    main()