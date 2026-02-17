import numpy as np # used for linear algebra - we need to handle the X,Y, Z coordinates of thousands of atoms as matrices 
from Bio import PDB # it knows that a PDB file isn't just text, but chains, residues and atoms
from scipy.spatial import KDTree # it organizes coordinates in a tree structure so we can ask "What is near this point?" in milliseconds instead of minutes 
from sklearn.cluster import DBSCAN # it finds clusters of density (pockets), without needing us to tell it how many pockets to look for
import sys
import argparse  # NEW: For command line arguments (Allows you to run: python predictor.py -i 1H8D.pdb)

# --- CLASS DEFINITION (Better PYT Score for Structure) ---

class LigandSitePredictor: 
    # You can create one object of this class for each protein and they won't interfere with each other
    def __init__(self, pdb_file):
        """Initialize with PDB file path."""
        self.pdb_file = pdb_file
        self.parser = PDB.PDBParser(QUIET=True)
        self.structure = None
        self.atoms = []
        self.atom_coords = None
        self.grid = None
        self.pockets = {}

    def load_structure(self): 
        """Parses PDB and filters for protein atoms (removes HETATM/Water)."""
        try:
            self.structure = self.parser.get_structure('protein', self.pdb_file) 
            
            # --- FIXED SECTION START ---
            self.atoms = []
            for residue in self.structure.get_residues():
                # Check if the RESIDUE is a standard amino acid
                if PDB.is_aa(residue, standard=True):
                    # If yes, keep all its atoms
                    for atom in residue:
                        self.atoms.append(atom)
            # --- FIXED SECTION END ---

            self.atom_coords = np.array([a.get_coord() for a in self.atoms])
            print(f"[+] Structure loaded: {len(self.atoms)} atoms.")
        except Exception as e:
            print(f"[!] Error loading PDB: {e}")
            sys.exit(1)

    def generate_grid(self, spacing=2.0, buffer=5.0): 
        """Defines the search space (Bounding Box).""" 
        # Bounding Box: we find the extreme Left, Right, Top, and Bottom atoms. 
        # We add buffer (5A) to ensure we check the space just outside the protein surface too.
        min_c = self.atom_coords.min(axis=0) - buffer
        max_c = self.atom_coords.max(axis=0) + buffer
        
        x = np.arange(min_c[0], max_c[0], spacing)
        y = np.arange(min_c[1], max_c[1], spacing)
        z = np.arange(min_c[2], max_c[2], spacing)
        
        # Meshgrid generates all 3D combinations (creates a 3D lattice of points)
        self.grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3) 
        print(f"[+] Grid generated: {len(self.grid)} points.")

    def scan_pockets(self):
        """Geometric filtering with STRICT Burial Check."""
        tree = KDTree(self.atom_coords)
        
        # 1. TIGHTER Distance Filter
        dists, _ = tree.query(self.grid)
        # Change max_dist from 5.0 to 4.5 to cut off the "outer skin"
        mask = (dists > 2.5) & (dists < 4.5) 
        candidates = self.grid[mask]
        
        print(f"[+] Distance filter: {len(candidates)} points found (still too many).")

        # 2. AGGRESSIVE Burial Filter
        # We look for protein atoms within 6.0 Angstroms
        neighbor_indices = tree.query_ball_point(candidates, r=6.0)
        
        # Count neighbors
        neighbor_counts = np.array([len(indices) for indices in neighbor_indices])
        
        # NEW THRESHOLD: Must have at least 28 neighbors to be considered "inside" a pocket
        # (Surface points usually have ~15-20 neighbors. Deep pockets have >28)
        concavity_mask = neighbor_counts >= 28
        final_points = candidates[concavity_mask]
        
        print(f"[+] Burial filter: Kept {len(final_points)} deep points (removed {len(candidates) - len(final_points)} surface points).")
        return final_points

    def cluster_points(self, points, eps=4.0, min_samples=5):
        """DBSCAN Clustering to group points into pockets."""
        if len(points) == 0: 
            print("No candidate points found.")
            return
        
        # The Problem: The previous step gave us a list of "hollow" points, but the computer sees them as a random list.
        # The Solution (DBSCAN): This algorithm looks at the points. If point A is close to point B, they are friends.
        
        # eps=4.0: The "reach" of a point. Points within 4A are connected.
        # min_samples=5: A group must have at least 5 points to be a real pocket. This filters out random noise.
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = db.labels_
        
        # Group points by cluster label
        unique_labels = set(labels)
        for label in unique_labels:
            if label != -1:  # -1 is noise
                self.pockets[label] = points[labels == label]
        
        print(f"[+] Clustering complete. Found {len(self.pockets)} pockets.")

    def identify_residues(self, pocket_points, contact_cutoff=4.0): # Biological Mapping
        """Identifies residues lining the pocket."""
        tree = KDTree(self.atom_coords)
        
        # Now we work backwards. We take the Pocket Points (empty space) and ask: 
        # "Which protein atoms are touching this empty space?" (within 4.0A).
        indices = tree.query_ball_point(pocket_points, r=contact_cutoff) 
        
        # Flatten indices and map to unique residues
        unique_atom_indices = {idx for sublist in indices for idx in sublist}
        
        # get_parent(): In Biopython, an Atom belongs to a Residue. 
        # We want the list of Residues (e.g., Histidine 57), not the list of 500 carbon atoms.
        residues = {self.atoms[idx].get_parent() for idx in unique_atom_indices}
        
        # Format: "HIS 57 A"
        res_list = sorted(
            list(residues), 
            key=lambda r: r.id[1]
        )
        return [f"{r.get_resname()} {r.id[1]} {r.get_full_id()[2]}" for r in res_list]

    def save_pdb(self, points, filename): # Visualization output
        """Exports grid points as HETATM for visualization."""
        # We fake a PDB file. We pretend our grid points are atoms (HETATM) named PTS. 
        # This tricks programs like Chimera into displaying them as dots so you can see the pocket shape.
        with open(filename, 'w') as f:
            for i, p in enumerate(points):
                f.write(f"HETATM{i+1:>5}  P   PTS A   1    {p[0]:>8.3f}{p[1]:>8.3f}{p[2]:>8.3f}  1.00  0.00           N\n")
            f.write("END\n")

    # --- NEW FEATURE: AUTOMATIC CHIMERA SCRIPT ---
    def create_chimera_script(self):
        """Generates a view_results.cmd file to instantly view pockets in Chimera."""
        with open("view_results.cmd", "w") as f:
            f.write(f"open {self.pdb_file}\n")  # Open the protein
            f.write("preset apply interactive 1\n") # Make it look nice (white surface)
            f.write("surface\n")
            f.write("color white,a\n") # Color protein white
            
            # Open each pocket file and color it
            for p_id in self.pockets:
                filename = f"pocket_{p_id}.pdb"
                f.write(f"open {filename}\n")
                # Assign a different color ID to each pocket
                f.write(f"color red #{p_id + 1}\n") # Pockets will appear Red
        print("[+] Visualization script 'view_results.cmd' created.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Setup Arguments (Improved "Applicability" Score)
    # This allows you to run: python predictor.py --input 1H8D.pdb
    parser = argparse.ArgumentParser(description="Ligand Binding Site Predictor")
    parser.add_argument("-i", "--input", required=True, help="Path to PDB file")
    args = parser.parse_args()

    # 2. Run Analysis
    predictor = LigandSitePredictor(args.input)
    predictor.load_structure()
    
    # Grid settings
    SPACING = 2.0
    predictor.generate_grid(spacing=SPACING)
    
    candidates = predictor.scan_pockets()
    predictor.cluster_points(candidates)
    
    # 3. Analysis & Output
    print("\n" + "="*40)
    print(f"RESULTS FOR {args.input}")
    print("="*40)

    # Sort pockets by size (Largest is usually the active site)
    sorted_pockets = sorted(predictor.pockets.items(), key=lambda x: len(x[1]), reverse=True)

    for p_id, points in sorted_pockets:
        # --- NEW FEATURE: VOLUME CALCULATION ---
        # Each point represents a 2x2x2 cube = 8 cubic Angstroms
        volume = len(points) * (SPACING ** 3)
        
        res_list = predictor.identify_residues(points)
        
        print(f"\n> Pocket {p_id}")
        print(f"  Approx Volume: {volume} A^3")
        print(f"  Residues involved: {', '.join(res_list[:10])} ...") # Show first 10
        
        predictor.save_pdb(points, f"pocket_{p_id}.pdb")

    # 4. Create the Visualization Script (Improved "Tutorial" Score)
    predictor.create_chimera_script()

    # To run it: python predictor.py --input 1H8D.pdb