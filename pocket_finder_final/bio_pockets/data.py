"""
Data Processing and I/O Module
------------------------------
Handles the parsing, cleaning, and conversion of PDB structural data. 
Includes utilities for sequence extraction and exporting results for 
visualization in tools like Chimera or PyMOL.
"""

print("IMPORTING DATA...")

import subprocess
import platform

import os
import numpy as np
from Bio import PDB, SeqIO
from Bio.PDB import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Polypeptide import is_aa
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .utils import AA_3_TO_1

def get_protein_structure(pdb_file: str):
    """
    Parses a PDB file and filters for standard amino acid atoms.

    Args:
        pdb_file (str): Path to the input .pdb file.

    Returns:
        tuple: (Bio.PDB.Structure object, list of Bio.PDB.Atom objects)
    """
    parser = PDB.PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure('protein_obj', pdb_file)
    protein_atoms = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Exclude HETATM records (water, ligands, ions)
                if not is_aa(residue, standard=True):
                    continue
                for atom in residue:
                    protein_atoms.append(atom)
    return structure, protein_atoms

def extract_sequence_from_pdb(protein_atoms: list, output_fasta: str = "query.fasta") -> str:
    """
    Converts 3D atomic data into a 1D primary amino acid sequence.

    Args:
        protein_atoms (list): List of Bio.PDB.Atom objects.
        output_fasta (str): Filename to save the resulting FASTA file.

    Returns:
        str: Path to the generated FASTA file.
    """
    residues_seen = set()
    sequence_chars = []
    
    for atom in protein_atoms:
        res = atom.get_parent()
        # Unique key (Chain, ResSeq) prevents double-counting residues by their atoms
        res_key = (res.get_parent().id, res.id[1])
        
        if res_key not in residues_seen:
            res_name = res.get_resname()
            # Defaults to 'X' for unknown residues to maintain sequence length
            sequence_chars.append(AA_3_TO_1.get(res_name, 'X'))
            residues_seen.add(res_key)
    
    sequence = "".join(sequence_chars)
    record = SeqRecord(Seq(sequence), id="query", description="Extracted_from_PDB")
    
    with open(output_fasta, "w") as f:
        SeqIO.write(record, f, "fasta")
    
    return output_fasta

def save_clean_protein(structure, output_file: str, protein_chains: list = None, min_chain_length: int = 30):
    """
    Removes non-protein components and filters out short peptides.

    This is critical for pocket detection to avoid identifying 'pockets' created 
    by artifacts or small crystallized inhibitors.

    Args:
        structure (Structure): The Biopython structure object.
        output_file (str): Path to save the cleaned PDB.
        protein_chains (list, optional): List of chain IDs to keep.
        min_chain_length (int): Minimum residues required to keep a chain.
    """
    clean_structure = Structure("clean")
    model_new = Model(0)
    clean_structure.add(model_new)
    
    for model in structure:
        for chain in model:
            if protein_chains is not None and chain.id not in protein_chains:
                continue
            
            # Keep only standard amino acids

            valid_residues = [res for res in chain if is_aa(res, standard=True)]
            
            # Filter out chains that are too short (noise/inhibitors)
            if len(valid_residues) < min_chain_length:
                print(f"  -> Automated cleanup: chain '{chain.id}' removed (too short)")
                continue
                
            new_chain = Chain(chain.id)
            for residue in valid_residues:
                new_chain.add(residue.copy())
                
            if len(new_chain) > 0:
                model_new.add(new_chain)
                
    io = PDBIO()
    io.set_structure(clean_structure)
    io.save(output_file)

def save_points_to_pdb(points: np.ndarray, output_file: str) -> None:
    """
    Exports 3D grid points as HETATM records for visual inspection.
    """
    with open(output_file, 'w') as f:
        for i, (x, y, z) in enumerate(points):
            serial = (i % 99999) + 1
            # Standard PDB fixed-column formatting
            f.write(f"HETATM{serial:5d}  P   PTS A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P\n")
        f.write("END\n")


def save_protein_with_colored_pockets(protein_atoms: list, pockets_dict: dict, output_file: str):
    """
    Creates a PDB file mapping each pocket to a unique chain ID.
    Each residue is assigned to exactly ONE pocket (nearest pocket center wins).
    """
    from scipy.spatial import KDTree

    POCKET_CHAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pocket_items = list(pockets_dict.items())
    pocket_centers = np.array([pts.mean(axis=0) for _, pts in pocket_items])
    pocket_trees   = [KDTree(pts) for _, pts in pocket_items]
    center_tree    = KDTree(pocket_centers)

    # --- Assign each residue exclusively to one pocket ---
    seen_residues    = set()
    residue_to_chain = {}

    for atom in protein_atoms:
        res   = atom.get_parent()
        chain = res.get_parent()
        res_key = (chain.id, res.id[1])
        if res_key in seen_residues:
            continue
        seen_residues.add(res_key)

        # Prefer CA atom, otherwise first atom
        coord = np.array(res["CA"].get_coord() if "CA" in res
                         else list(res.get_atoms())[0].get_coord())

        # Assign residue if within 6 Å of ANY pocket point
        assigned = False
        for idx, tree in enumerate(pocket_trees):
            min_dist, _ = tree.query(coord)
            if min_dist <= 6.0:
                residue_to_chain[res_key] = POCKET_CHAIN_LETTERS[idx % 26]
                assigned = True
                break
        
    # --- Write PDB ---
    with open(output_file, 'w') as f:
        atom_id = 1

        for atom in protein_atoms:
            res    = atom.get_parent()
            chain  = res.get_parent()
            x, y, z = atom.get_coord()
            serial = (atom_id % 99999) + 1
            f.write(f"ATOM  {serial:5d} {atom.get_name():^4s} {res.get_resname():3s} "
                    f"{chain.id}{res.id[1]:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00           {atom.element:1s}\n")
            atom_id += 1

        for rank, (pocket_id, points) in enumerate(pocket_items):
            chain_id = POCKET_CHAIN_LETTERS[rank % 26]
            for point in points:
                x, y, z = point
                serial  = (atom_id % 99999) + 1
                f.write(f"HETATM{serial:5d}  P   PKT {chain_id}{rank+1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P\n")
                atom_id += 1
        f.write("END\n")

    return residue_to_chain



def generate_visualization_scripts(base_name: str, output_pdb: str, residue_to_chain: dict):
    """
    Generates visualization scripts using EXACT residue assignments.
    No distance-based selection = no overlap possible.
    residue_to_chain: {(chain_id, res_seq): pocket_chain_letter}
    """
    top3   = [("A", "firebrick"), ("B", "red"), ("C", "salmon")]
    others = [("D", "marine"), ("E", "forest"), ("F", "purple"), ("G", "cyan"),
              ("H", "yellow"), ("I", "magenta"), ("J", "teal"), ("K", "olive"), ("L", "slate")]

    # Group residues by pocket chain
    # Format: {"A": [(chain, resi), ...], "B": [...], ...}
    pocket_residues = {}
    for (orig_chain, resi), pocket_chain in residue_to_chain.items():
        if pocket_chain not in pocket_residues:
            pocket_residues[pocket_chain] = []
        pocket_residues[pocket_chain].append((orig_chain, resi))

   # =========================================================
    # 1. PyMOL Script
    # =========================================================
    pymol_script = f"{base_name}_pymol.pml"
    with open(pymol_script, "w") as f:
        f.write(f"load {output_pdb}\n")
        f.write("hide all\n")
        
        # Show surface ONLY for the protein (ignore pocket points)
        f.write("show surface, all and not resn PKT\n")
        f.write("color gray80, all and not resn PKT\n")

        # Others first, top 3 last (highest priority for overlaps)
        # IMPORTANT HERE: The order must be "others + top3" as in Chimera!
        all_colors = others + top3 
        
        for pocket_chain, color in all_colors:
            # We continue to check if the pocket is valid
            if pocket_chain not in pocket_residues:
                continue  # This pocket does not exist

            f.write(f"# Pocket {pocket_chain} -> {color}\n")
            
            # --- THE PYMOL DISTANCE TRICK (Equivalent to z<3.5 in Chimera) ---
            # Selects all protein atoms (not resn PKT) that are a maximum of 3.5 Ångströms 
            # away from the grid points of the current chain.
            selection = f"(all and not resn PKT) within 3.5 of (resn PKT and chain {pocket_chain})"
            
            f.write(f"select pocket{pocket_chain}_area, {selection}\n")
            f.write(f"color {color}, pocket{pocket_chain}_area\n")

        # (Optional) If you want to make the gray surface slightly transparent:
        # f.write("set transparency, 0.2, all and not resn PKT\n")
        
        # Hide the grid points completely
        f.write("hide everything, resn PKT\n")
        
        # Clear the selection markers (pink squares) and zoom
        f.write("deselect\n")
        f.write("zoom all\n")

        
    # =========================================================
    # 2. Chimera Script
    # =========================================================
    chimera_script = f"{base_name}_chimera.cmd"
    with open(chimera_script, "w") as f:
        f.write(f"open {output_pdb}\n")
        f.write("~display\n")
        f.write("surface\n")
        f.write("color light gray\n")

        # Others first, top 3 last (highest priority)
        all_colors_chimera = others + top3
        chimera_color_map = {
            "firebrick": "dark red", "red": "red", "salmon": "salmon",
            "marine": "cornflower blue", "forest": "forest green", "purple": "purple",
            "cyan": "cyan", "yellow": "yellow", "magenta": "magenta",
            "teal": "teal", "olive": "olive drab", "slate": "slate gray"
        }

        for pocket_chain, color in all_colors_chimera:
            if pocket_chain not in pocket_residues:
                continue

            chimera_color = chimera_color_map.get(color, color)
            
            # --- HERE IS THE ONLY CHANGE ---
            # Instead of passing jagged residues, we use "z<3.5" (distance to the grid) 
            # for a continuous, smooth surface around the respective pocket.
            f.write(f"color {chimera_color} :PKT.{pocket_chain} z<3.5\n")

        f.write("~display :PKT\n")
        f.write("focus\n")

    return pymol_script, chimera_script


def open_in_chimera(chimera_script: str):
    """
    Automatically opens UCSF Chimera with the generated visualization script,
    so the protein surface with colored pockets is displayed immediately.
    """

    # --- Default installation paths for Chimera on each operating system ---
    chimera_paths = {
        "linux":   "/user/bin/chimera",
        "mac":     "/Applications/Chimera.app/Contents/MacOS/chimera",
        "windows": r"C:\Program Files\Chimera\bin\chimera.exe"
    }

    # --- Detect the current operating system ---
    # platform.system() returns 'Linux', 'Darwin' (macOS), or 'Windows'
    system = platform.system().lower()
    if system == "darwin":
        system = "mac"  # Rename Darwin -> mac to match our dictionary keys

    # --- Look up the expected Chimera executable path for this OS ---
    chimera_exe = chimera_paths.get(system)

    # --- Try to launch Chimera using the default path ---
    if chimera_exe and os.path.exists(chimera_exe):
        # Launch Chimera as a separate process and pass the script as an argument
        # Popen does NOT block the rest of your program while Chimera is open
        subprocess.Popen([chimera_exe, chimera_script])
        print(f"Chimera opened with script: {chimera_script}")

    else:
        # --- Fallback: try if 'chimera' is available anywhere in the system PATH ---
        # This handles cases where Chimera was installed to a non-default location
        try:
            subprocess.Popen(["chimera", chimera_script])
            print(f"Chimera opened with script: {chimera_script}")

        except FileNotFoundError:
            # If Chimera cannot be found at all, print a manual instruction
            print("Chimera executable not found. Please open it manually:")
            print(f"  chimera {chimera_script}")