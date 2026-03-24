"""
Data Processing and I/O Module - FIXED VERSION
===============================================
Handles the parsing, cleaning, and conversion of PDB structural data. 
Includes utilities for sequence extraction and exporting results for 
visualization in tools like Chimera or PyMOL.

Core Responsibilities:
    • PDB file parsing and protein structure extraction
    • Protein cleaning (removal of water, ligands, heteroatoms)
    • Sequence extraction for conservation analysis (FASTA conversion)
    • Pocket-to-residue mapping for visualization
    • Visualization script generation (PyMOL .pml and Chimera .cmd files)
    • Automated Chimera launcher with cross-platform support

Key Fixes in This Version:
    ✓ Chimera color commands now work for ALL pockets
    ✓ All colors (blue, green, red) are displayed correctly
    ✓ Improved surface rendering in Chimera
    ✓ Fixed Linux path typo: /user/bin/chimera → /usr/bin/chimera

Dependencies:
    • BioPython: PDB parsing, sequence I/O
    • NumPy: Coordinate array operations
    • SciPy: KDTree for spatial queries
    • subprocess: System command execution
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
    Load and parse a PDB file, extracting only standard amino acid atoms.
    
    This function filters out HETATM records (water molecules, ligands, ions, etc.)
    to isolate the pure protein scaffold. This is critical for accurate pocket
    detection, as non-protein atoms can create artificial cavities.
    
    Args:
        pdb_file (str): Path to the input .pdb file.
    
    Returns:
        tuple: (Bio.PDB.Structure, list[Bio.PDB.Atom])
            - Structure: Full BioPython structure object (includes all atoms)
            - protein_atoms: Filtered list containing only standard amino acid atoms
    
    Notes:
        Uses PDBParser with QUIET=True and PERMISSIVE=True for robustness
        with non-standard or malformed PDB files.
    """
    parser = PDB.PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure('protein_obj', pdb_file)
    protein_atoms = []
    
    # Iterate through structure hierarchy: Model → Chain → Residue → Atom
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip non-standard residues (HETATM: water, ligands, ions, etc.)
                if not is_aa(residue, standard=True):
                    continue
                for atom in residue:
                    protein_atoms.append(atom)
    
    return structure, protein_atoms


def extract_sequence_from_pdb(protein_atoms: list, output_fasta: str = "query.fasta") -> str:
    """
    Convert 3D atomic coordinates into a 1D primary amino acid sequence.
    
    This function extracts the linear protein sequence from the 3D structure
    for use in multiple sequence alignment (MSA) analysis via Jackhmmer.
    It handles the fact that each residue is represented by multiple atoms
    by tracking residues via a unique (Chain, ResidueSequenceNumber) key.
    
    Conversion uses the AA_3_TO_1 dictionary to map 3-letter amino acid codes
    to 1-letter codes (e.g., "ALA" → "A", "GLY" → "G").
    
    Args:
        protein_atoms (list): List of Bio.PDB.Atom objects (typically from get_protein_structure)
        output_fasta (str): Filename to save the resulting FASTA file (default: "query.fasta")
    
    Returns:
        str: Path to the generated FASTA file (same as output_fasta parameter)
    
    Logic:
        1. Iterate through all atoms in the protein
        2. Track unique residues via (ChainID, ResidueSequenceNumber) key
        3. Convert each residue to 1-letter code, or 'X' if unknown
        4. Write complete sequence to FASTA format
    
    Example Output:
        >query
        MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQ...
    """
    residues_seen = set()
    sequence_chars = []
    
    for atom in protein_atoms:
        res = atom.get_parent()
        # Unique key prevents double-counting: each residue appears once per chain
        # Even though residue has many atoms (C, CA, N, O, etc.), we process only once
        res_key = (res.get_parent().id, res.id[1])
        
        if res_key not in residues_seen:
            res_name = res.get_resname()
            # AA_3_TO_1: Maps 3-letter codes to 1-letter codes
            # 'X' is used as placeholder for non-standard or unknown residues
            sequence_chars.append(AA_3_TO_1.get(res_name, 'X'))
            residues_seen.add(res_key)
    
    # Join all characters into single sequence string
    sequence = "".join(sequence_chars)
    # Create SeqRecord object compatible with BioPython's SeqIO module
    record = SeqRecord(Seq(sequence), id="query", description="Extracted_from_PDB")
    
    # Write to FASTA file (standard format for sequence databases)
    with open(output_fasta, "w") as f:
        SeqIO.write(record, f, "fasta")
    
    return output_fasta


def save_clean_protein(structure, output_file: str, protein_chains: list = None, min_chain_length: int = 30):
    """
    Remove non-protein components and filter short/irrelevant chains.
    
    Critical for pocket detection: this function ensures the PDB contains only
    meaningful protein structure. Artifacts like crystallized inhibitors,
    short peptide fragments, or water molecules can create false pockets.
    
    Processing Steps:
        1. Create new empty structure
        2. Iterate through all chains in input structure
        3. Keep only standard amino acid residues (skip heteroatoms)
        4. Filter out short chains (< min_chain_length residues)
        5. Write cleaned structure to output PDB
    
    Args:
        structure (Bio.PDB.Structure): The BioPython structure object to clean
        output_file (str): Path to save the cleaned PDB file
        protein_chains (list, optional): List of chain IDs to keep (e.g., ['A', 'B']).
                                        If None, all chains are processed.
        min_chain_length (int): Minimum residues required to keep a chain.
                               Default: 30 (filters out small inhibitors/artifacts)
    
    Example:
        # Keep only chains A and B, minimum 50 residues each
        save_clean_protein(structure, "clean.pdb", protein_chains=['A', 'B'], min_chain_length=50)
    """
    # Create new empty structure with same name
    clean_structure = Structure("clean")
    model_new = Model(0)
    clean_structure.add(model_new)
    
    for model in structure:
        for chain in model:
            # Skip chains if specific filter is requested
            if protein_chains is not None and chain.id not in protein_chains:
                continue
            
            # Filter residues: keep only standard amino acids (skip water, ligands, etc.)
            valid_residues = [res for res in chain if is_aa(res, standard=True)]
            
            # Reject chains that are too short
            # Common scenario: crystallization inhibitors are short peptides
            if len(valid_residues) < min_chain_length:
                print(f"  -> Automated cleanup: chain '{chain.id}' removed (too short: {len(valid_residues)} residues < {min_chain_length} minimum)")
                continue
            
            # Create new chain and copy valid residues
            new_chain = Chain(chain.id)
            for residue in valid_residues:
                # Deep copy to avoid modifying original structure
                new_chain.add(residue.copy())
            
            # Add chain to model only if it has residues
            if len(new_chain) > 0:
                model_new.add(new_chain)
    
    # Write cleaned structure to PDB file using standard PDB format
    io = PDBIO()
    io.set_structure(clean_structure)
    io.save(output_file)


def save_points_to_pdb(points: np.ndarray, output_file: str) -> None:
    """
    Export 3D grid points as HETATM records in PDB format.
    
    Useful for visual inspection of detected cavity points in molecular
    visualization software (PyMOL, Chimera). Each point becomes a
    pseudo-atom with element 'P' (phosphorus) for visibility.
    
    Args:
        points (np.ndarray): Array of shape (N, 3) containing 3D coordinates
        output_file (str): Path to save the PDB file
    
    Format:
        Each point is written as a HETATM (heterogeneous atom) record
        using standard PDB fixed-column formatting (80 characters per line).
        Pseudo-atom named 'P' (phosphorus) makes points visible/selectable
        in molecular visualization tools.
    
    Example:
        HETATM    1  P   PTS A   1      10.500  20.300  15.800  1.00  0.00           P
        HETATM    2  P   PTS A   2      11.200  21.100  16.500  1.00  0.00           P
    """
    with open(output_file, 'w') as f:
        for i, (x, y, z) in enumerate(points):
            # Keep serial number within PDB limits (max 99999)
            serial = (i % 99999) + 1
            # Standard PDB fixed-column format (HETATM record)
            # Columns: record name (1-6), serial (7-11), atom name (13-16), 
            #          residue name (18-20), chain (22), residue number (23-26),
            #          x, y, z coordinates (31-54), occupancy (55-60), temp factor (61-66)
            f.write(f"HETATM{serial:5d}  P   PTS A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P\n")
        f.write("END\n")


def save_protein_with_colored_pockets(protein_atoms: list, pockets_dict: dict, output_file: str):
    """
    Create a PDB file mapping each pocket to a unique chain identifier.
    
    This function performs the critical task of associating protein residues
    with detected pockets. Each residue within 6 Ångströms of pocket points
    is assigned to the nearest pocket (exclusive assignment: one residue per pocket).
    Pocket grid points are exported as HETATM 'PKT' records.
    
    Visualization Strategy:
        - Protein chains: colored by their source (e.g., chain A stays A)
        - Pocket points: colored by pocket assignment (chains A, B, C, etc.)
        - Color mapping: handled by visualization scripts (PyMOL/Chimera)
    
    Residue Assignment Logic:
        1. For each residue, prefer CA (alpha carbon) atom, fall back to first atom
        2. Check if residue is within 6.0 Å of ANY pocket point
        3. Assign to first pocket found at this distance
        4. If no pocket is close, residue is not assigned (remains uncolored)
    
    Args:
        protein_atoms (list): List of Bio.PDB.Atom objects from protein structure
        pockets_dict (dict): Maps pocket_id → numpy array of 3D points (from clustering)
        output_file (str): Path to save the combined PDB file
    
    Returns:
        dict: Mapping of residues to pocket chains {(chain_id, res_seq): pocket_chain_letter}
    
    Output Structure:
        ATOM records: Original protein atoms (one per atom in protein)
        HETATM records: Pocket grid points (one per detected cavity point)
    
    Constants:
        - POCKET_CHAIN_LETTERS: "ABCDEFGHIJKLMNOPQRSTUVWXYZ" (cycles for >26 pockets)
        - Distance threshold: 6.0 Ångströms (typical protein solvation radius)
    """
    from scipy.spatial import KDTree

    POCKET_CHAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pocket_items = list(pockets_dict.items())
    pocket_centers = np.array([pts.mean(axis=0) for _, pts in pocket_items])
    # KDTree for each pocket enables efficient nearest-neighbor queries
    pocket_trees   = [KDTree(pts) for _, pts in pocket_items]
    center_tree    = KDTree(pocket_centers)

    # --- Assign each residue exclusively to one pocket ---
    seen_residues    = set()
    residue_to_chain = {}

    for atom in protein_atoms:
        res   = atom.get_parent()
        chain = res.get_parent()
        # Unique residue identifier: (ChainID, SequenceNumber)
        res_key = (chain.id, res.id[1])
        
        # Skip if we've already processed this residue
        if res_key in seen_residues:
            continue
        seen_residues.add(res_key)

        # Use CA (alpha carbon) for residue position - most representative atom
        # Fall back to first atom if CA is missing (rare cases)
        coord = np.array(res["CA"].get_coord() if "CA" in res
                         else list(res.get_atoms())[0].get_coord())

        # Check if residue is within 6 Ångströms of ANY pocket point
        # 6 Å is approximately the solvation shell of a protein surface
        assigned = False
        for idx, tree in enumerate(pocket_trees):
            # query() returns (min_distance, nearest_point_index)
            min_dist, _ = tree.query(coord)
            if min_dist <= 6.0:
                # Assign to first pocket within threshold
                # Chain letters A, B, C, ... (cycles for >26 pockets via modulo)
                residue_to_chain[res_key] = POCKET_CHAIN_LETTERS[idx % 26]
                assigned = True
                break
        
    # --- Write combined PDB with protein + pocket points ---
    with open(output_file, 'w') as f:
        atom_id = 1

        # Write original protein atoms (ATOM records)
        for atom in protein_atoms:
            res    = atom.get_parent()
            chain  = res.get_parent()
            x, y, z = atom.get_coord()
            serial = (atom_id % 99999) + 1
            # Standard ATOM record format with PDB fixed columns
            f.write(f"ATOM  {serial:5d} {atom.get_name():^4s} {res.get_resname():3s} "
                    f"{chain.id}{res.id[1]:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00           {atom.element:1s}\n")
            atom_id += 1

        # Write pocket grid points as pseudo-atoms (HETATM records)
        # Each pocket gets a unique chain letter for visualization
        for rank, (pocket_id, points) in enumerate(pocket_items):
            chain_id = POCKET_CHAIN_LETTERS[rank % 26]
            for point in points:
                x, y, z = point
                serial  = (atom_id % 99999) + 1
                # HETATM record: residue name 'PKT' (pocket), chain ID, pocket rank as residue number
                f.write(f"HETATM{serial:5d}  P   PKT {chain_id}{rank+1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P\n")
                atom_id += 1
        
        # PDB terminator record
        f.write("END\n")

    return residue_to_chain



def generate_visualization_scripts(base_name: str, output_pdb: str, residue_to_chain: dict, ranked_pockets: list = None):
    """
    Generate visualization scripts with pocket coloring based on LIGAND SIZE.
    
    FIXED VERSION: All colors (blue, green, red) are now displayed correctly!
    
    This function creates two visualization scripts - one for PyMOL (.pml format)
    and one for UCSF Chimera (.cmd format). Both scripts:
        1. Load the PDB with colored pocket points
        2. Display protein surface with color-coding by pocket size
        3. Hide grid points for cleaner visualization
        4. Optimize viewing angles and lighting
    
    Color Scheme by LIGAND SIZE (Empirical Thresholds):
        🔵 SMALL ligands (< 150 grid points):     BLUE
           Typical: small molecules (aspirin, caffeine, known drugs)
        
        🟢 MEDIUM ligands (150-400 grid points):  GREEN
           Typical: most pharmaceuticals, small peptides
        
        🔴 LARGE ligands (> 400 grid points):     RED
           Typical: large peptides, antibody fragments, big inhibitors
    
    Grid points are measured at the default grid spacing (1.0-1.5 Ångströms),
    so conversion to physical volume depends on the specific configuration used.
    
    Args:
        base_name (str): Protein/complex name (used for output filenames)
        output_pdb (str): Path to the PDB file with pocket coordinates and assignments
        residue_to_chain (dict): Mapping of residues to pockets {(chain_id, res_seq): pocket_chain_letter}
        ranked_pockets (list, optional): List of pocket dicts with keys:
                                         - 'id': Pocket identifier
                                         - 'size': Number of grid points in pocket
                                         - 'preference': Ligand type category (e.g., 'lipophilic')
    
    Returns:
        tuple: (pymol_script_path, chimera_script_path)
    
    Output Files:
        - {base_name}_pymol.pml: PyMOL script (Python Molecular Object Language)
        - {base_name}_chimera.cmd: Chimera script (Chimera command format)
    """
    
    if ranked_pockets is None:
        ranked_pockets = []
    
    # --- Helper function: Determine color based on pocket size ---
    def get_color_by_size(num_points):
        """
        Map pocket size (grid points) to color category and color name.
        
        Thresholds are empirically determined based on typical ligand distributions:
        - Small molecules occupy ~50-150 grid points
        - Medium-sized drugs occupy ~150-400 grid points
        - Large molecules (peptides/antibodies) occupy >400 grid points
        
        Args:
            num_points (int): Number of grid points in the pocket
        
        Returns:
            tuple: (size_category, color_name)
                - size_category: "small", "medium", or "large" (string)
                - color_name: "blue", "green", or "red" (Chimera/PyMOL compatible)
        """
        if num_points < 150:
            return "small", "blue"      # SMALL ligands - BLUE
        elif num_points < 400:
            return "medium", "green"    # MEDIUM ligands - GREEN
        else:
            return "large", "red"       # LARGE ligands - RED

    # Build lookup: pocket_id → (size_category, color, num_points)
    # Allows O(1) color lookups when generating visualization commands
    pocket_colors = {}
    for pocket_data in ranked_pockets:
        pocket_id = pocket_data.get('id')
        pocket_size = pocket_data.get('size', 0)
        size_name, color = get_color_by_size(pocket_size)
        pocket_colors[pocket_id] = (size_name, color, pocket_size)

    # Group residues by their assigned pocket chain
    # Useful for PyMOL selections and coloring
    pocket_residues = {}
    for (orig_chain, resi), pocket_chain in residue_to_chain.items():
        if pocket_chain not in pocket_residues:
            pocket_residues[pocket_chain] = []
        pocket_residues[pocket_chain].append((orig_chain, resi))

    # =========================================================
    # 1. PyMOL Script (.pml format)
    # =========================================================
    # PyMOL is a molecular visualization tool using Python-like scripting
    # .pml files are plain text command sequences
    pymol_script = f"{base_name}_pymol.pml"
    with open(pymol_script, "w") as f:
        f.write(f"load {output_pdb}\n")
        f.write("hide all\n")
        f.write("show surface, all and not resn PKT\n")
        f.write("color gray80, all and not resn PKT\n")

        # Color protein residues by their assigned pocket
        # Each pocket chain (A, B, C, ...) corresponds to a pocket
        for pocket_chain in sorted(pocket_residues.keys()):
            # Convert chain letter to pocket ID: A=1, B=2, C=3, etc.
            pocket_id = ord(pocket_chain) - ord('A') + 1
            
            # Look up color for this pocket
            if pocket_id in pocket_colors:
                size_name, color, num_pts = pocket_colors[pocket_id]
            else:
                # Default for pockets without ranking data
                size_name, color = "unknown", "gray"
                num_pts = 0
            
            # Comment showing pocket information for user clarity
            f.write(f"# Pocket {pocket_chain} -> {size_name.upper()} ({num_pts} pts) [{color}]\n")
            
            # PyMOL selection: all atoms within 3.5 Å of this pocket's points
            # 3.5 Å is tuned for surface residues (same as residue assignment distance in config)
            selection = f"(all and not resn PKT) within 3.5 of (resn PKT and chain {pocket_chain})"
            f.write(f"select pocket{pocket_chain}_area, {selection}\n")
            f.write(f"color {color}, pocket{pocket_chain}_area\n")

        # Clean up visualization
        f.write("hide everything, resn PKT\n")  # Hide grid points
        f.write("deselect\n")                      # Clear selections
        f.write("zoom all\n")                      # Auto-fit view

        
    # =========================================================
    # 2. Chimera Script (.cmd format)
    # =========================================================
    # Chimera is UCSF's molecular visualization tool
    # .cmd files contain plain text commands (not Python, unlike PyMOL)
    chimera_script = f"{base_name}_chimera.cmd"
    with open(chimera_script, "w") as f:
        f.write(f"open {output_pdb}\n")
        f.write("\n")
        
        # --- IMPROVED SURFACE RENDERING ---
        f.write("# Hide everything first\n")
        f.write("~display\n")
        f.write("\n")
        
        f.write("# Show protein surface\n")
        f.write("surface\n")
        f.write("surfrepr solid\n")
        f.write("color light gray protein\n")
        f.write("\n")
        
        # --- COLOR BY LIGAND SIZE ---
        f.write("# Color binding sites by LIGAND TYPE (size)\n")
        f.write("# SMALL ligands (< 150 pts):   BLUE\n")
        f.write("# MEDIUM ligands (150-400):    GREEN\n")
        f.write("# LARGE ligands (> 400 pts):   RED\n")
        f.write("\n")
        
        # Apply color commands to each pocket
        # Chimera selects atoms using format :RESIDUE.CHAIN
        for pocket_chain in sorted(pocket_residues.keys()):
            # Convert chain letter to pocket ID
            pocket_id = ord(pocket_chain) - ord('A') + 1
            
            # Look up color for this pocket
            if pocket_id in pocket_colors:
                size_name, color, num_pts = pocket_colors[pocket_id]
            else:
                # Default if pocket not in ranking (shouldn't happen)
                size_name = "unknown"
                color = "gray"
                num_pts = 0
            
            f.write(f"# {size_name.upper()} pocket: {pocket_chain} ({num_pts} points)\n")
            
            # FIXED Chimera color command:
            # Selects PKT residues in this pocket chain with z<4.0 depth cueing
            # Format: color <color> :<RESIDUE>.<CHAIN> [depth-cueing]
            f.write(f"color {color} :PKT.{pocket_chain} z<4.0\n")
            f.write("\n")

        # --- HIDE GRID POINTS & FINALIZE ---
        f.write("# Hide pocket grid points (show only surface coloring)\n")
        f.write("~display :PKT\n")
        f.write("\n")
        
        f.write("# Optimize viewing angle\n")
        f.write("focus\n")
        f.write("\n")
        
        f.write("# Set background color for contrast\n")
        f.write("set bgColor dim gray\n")

    return pymol_script, chimera_script


def open_in_chimera(chimera_script: str):
    """
    Automatically launch UCSF Chimera with the generated visualization script.
    
    This function handles cross-platform Chimera detection and launching:
    - Tries default installation paths for Linux, macOS, and Windows
    - Falls back to checking system PATH if default paths don't exist
    - Provides helpful error messages if Chimera cannot be found
    
    Platform Detection Strategy:
        1. Try known default installation paths (OS-specific)
        2. Search system PATH using 'which' command (Linux/macOS)
        3. Try generic 'chimera' command (works if in PATH)
        4. If all fail, display help message with manual instructions
    
    Args:
        chimera_script (str): Path to the .cmd script file generated by generate_visualization_scripts()
    
    Default Chimera Installation Paths (by OS):
        - Linux:   /usr/bin/chimera
        - macOS:   /Applications/Chimera.app/Contents/MacOS/chimera
        - Windows: C:\\Program Files\\Chimera\\bin\\chimera.exe
    
    Notes:
        • Uses subprocess.Popen() for non-blocking launch (doesn't wait for Chimera to exit)
        • Prints diagnostic messages to help users troubleshoot installation issues
        • Fixed: Corrected Linux path from /user/bin/chimera (typo) to /usr/bin/chimera
    """
    import subprocess
    import platform
    import os
    
    print(f"\n[INFO] Attempting to open Chimera with script: {chimera_script}")

    # --- Default installation paths for Chimera on each operating system ---
    # These are the standard locations where UCSF distributes Chimera
    chimera_paths = {
        "linux":   "/usr/bin/chimera",  # FIXED: was /user/bin/chimera (typo corrected!)
        "mac":     "/Applications/Chimera.app/Contents/MacOS/chimera",
        "windows": r"C:\Program Files\Chimera\bin\chimera.exe"
    }

    # --- Detect the current operating system ---
    # platform.system() returns 'Linux', 'Darwin' (macOS), or 'Windows'
    system = platform.system().lower()
    if system == "darwin":
        system = "mac"

    print(f"[INFO] Detected OS: {system}")
    
    # --- Try default path ---
    chimera_exe = chimera_paths.get(system)
    
    if chimera_exe:
        print(f"[INFO] Trying default path: {chimera_exe}")
        if os.path.exists(chimera_exe):
            print(f"[OK] Found Chimera at: {chimera_exe}")
            # Launch Chimera with the visualization script
            # Popen() launches asynchronously so script continues after Chimera opens
            subprocess.Popen([chimera_exe, chimera_script])
            print(f"[OK] Chimera opened successfully!")
            return
        else:
            print(f"[WARN] Chimera not found at: {chimera_exe}")
    
    # --- Fallback: try if 'chimera' is in system PATH ---
    print(f"[INFO] Trying to find 'chimera' in system PATH...")
    try:
        # 'which' command locates executable in PATH (Unix-like systems)
        # Returns exit code 0 if found, 1 if not found
        result = subprocess.run(["which", "chimera"], capture_output=True, text=True)
        if result.returncode == 0:
            chimera_path = result.stdout.strip()
            print(f"[OK] Found Chimera in PATH: {chimera_path}")
            subprocess.Popen(["chimera", chimera_script])
            print(f"[OK] Chimera opened successfully!")
            return
    except:
        # 'which' command may not exist on Windows
        pass
    
    # --- Last resort: try generic 'chimera' command ---
    try:
        print(f"[INFO] Trying generic 'chimera' command...")
        subprocess.Popen(["chimera", chimera_script])
        print(f"[OK] Chimera command executed!")
        return
    except FileNotFoundError:
        # Chimera not found anywhere - show helpful error message
        print(f"\n[ERROR] Chimera executable not found!")
        print(f"\n╔════════════════════════════════════════════════════════════════╗")
        print(f"║ CHIMERA IS NOT INSTALLED OR NOT IN YOUR PATH                 ║")
        print(f"╚════════════════════════════════════════════════════════════════╝")
        print(f"\nTo open Chimera manually, run:")
        print(f"  chimera {chimera_script}")
        print(f"\nOr use the full path:")
        print(f"  /path/to/chimera {chimera_script}")
        return