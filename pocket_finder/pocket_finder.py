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
    ranking_df = myproject.rank_pockets_master_score(all_pockets, clean_atoms)

    # Output as a nicely formatted text table directly in the terminal
    print("RESULTS")
    print("-" * 75)
    if not ranking_df.empty:
        # to_string() prints a Pandas DataFrame cleanly in the terminal
        print(ranking_df.to_string())
    else:
        print("No pockets found.")
    print("-" * 75)

    # Save results
    myproject.save_protein_with_colored_pockets(clean_atoms, all_pockets, output_file)
    print(f"\nDone! Results were saved as '{output_file}'.\n")


if __name__ == "__main__":
    main()
    
    
