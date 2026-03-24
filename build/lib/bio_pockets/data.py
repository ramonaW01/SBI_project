"""
Data Processing and I/O Module
------------------------------
Handles the parsing, cleaning, and conversion of PDB structural data. 
Includes utilities for sequence extraction and exporting results for 
visualization in tools like Chimera or PyMOL.
"""

print("IMPORTING DATA...")

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

    This allows researchers to use 'color by chain' in PyMOL to instantly 
    distinguish between different predicted binding sites.
    """
    POCKET_CHAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    with open(output_file, 'w') as f:
        atom_id = 1

        # Section: Write Protein Atoms
        for atom in protein_atoms:
            res = atom.get_parent()
            chain = res.get_parent()
            x, y, z = atom.get_coord()
            resseq = res.id[1]
            serial = (atom_id % 99999) + 1
            f.write(f"ATOM  {serial:5d} {atom.get_name():^4s} {res.get_resname():3s} {chain.id}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom.element:1s}\n")
            atom_id += 1
        
        # Section: Write Pocket Points (HETATM)
        for rank, (pocket_id, points) in enumerate(pockets_dict.items()):
            # Cycle through alphabet if there are many pockets
            chain_id = POCKET_CHAIN_LETTERS[rank % len(POCKET_CHAIN_LETTERS)]
            for point in points:
                x, y, z = point
                serial = (atom_id % 99999) + 1
                f.write(f"HETATM{serial:5d}  P   PKT {chain_id}{rank + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P\n")
                atom_id += 1
        f.write("END\n")