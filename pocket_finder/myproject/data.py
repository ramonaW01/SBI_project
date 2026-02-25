print("Importing data")

import os
import numpy as np
from Bio import PDB
from Bio.PDB import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Polypeptide import is_aa

def get_protein_structure(pdb_file: str):
    """ Reads a PDB file and extracts only standard amino acid atoms. """
    parser = PDB.PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure('protein_obj', pdb_file)
    protein_atoms = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_aa(residue, standard=True):
                    continue
                for atom in residue:
                    protein_atoms.append(atom)
    return structure, protein_atoms

def save_clean_protein(structure, output_file, protein_chains=None, min_chain_length=30):
    """ Saves a cleaned PDB keeping specific chains and removing water/ligands/short peptides. """
    clean_structure = Structure("clean")
    model_new = Model(0)
    clean_structure.add(model_new)
    
    for model in structure:
        for chain in model:
            # 1. Wenn der Benutzer eine Kette vorgibt (wie "H"), halte dich daran
            if protein_chains is not None and chain.id not in protein_chains:
                continue
            
            # 2. Sammle alle echten Aminosäuren der Kette
            valid_residues = [res for res in chain if is_aa(res, standard=True)]
            
            # 3. DIE NEUE AUTOMATIK: Ist die Kette zu kurz? -> Raus damit!
            if len(valid_residues) < min_chain_length:
                print(f"  -> Automaticly deleted: chain '{chain.id}' (only {len(valid_residues)} Aminosäuren, vermutlich Peptid/Inhibitor)")
                continue
                
            # 4. Wenn die Kette lang genug ist, füge sie zum sauberen Protein hinzu
            new_chain = Chain(chain.id)
            for residue in valid_residues:
                new_chain.add(residue.copy())
                
            if len(new_chain) > 0:
                model_new.add(new_chain)
                
    io = PDBIO()
    io.set_structure(clean_structure)
    io.save(output_file)

def save_points_to_pdb(points: np.ndarray, output_file: str) -> None:
    """ Saves grid coordinates as dummy HETATM PDB records. """
    with open(output_file, 'w') as f:
        for i, (x, y, z) in enumerate(points):
            serial = (i % 99999) + 1
            f.write(f"HETATM{serial:5d}  P   PTS A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P\n")
        f.write("END\n")

def save_protein_with_colored_pockets(protein_atoms, pockets_dict, output_file):
    """ Exports protein and predicted pockets as separate chains for easy coloring. """
    POCKET_CHAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    with open(output_file, 'w') as f:
        atom_id = 1
        for atom in protein_atoms:
            res = atom.get_parent()
            chain = res.get_parent()
            x, y, z = atom.get_coord()
            resseq = res.id[1]
            serial = (atom_id % 99999) + 1
            f.write(f"ATOM  {serial:5d} {atom.get_name():^4s} {res.get_resname():3s} {chain.id}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom.element:1s}\n")
            atom_id += 1
            
        for rank, (pocket_id, points) in enumerate(pockets_dict.items()):
            chain_id = POCKET_CHAIN_LETTERS[rank % len(POCKET_CHAIN_LETTERS)]
            for point in points:
                x, y, z = point
                serial = (atom_id % 99999) + 1
                f.write(f"HETATM{serial:5d}  P   PKT {chain_id}{rank + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P\n")
                atom_id += 1
        f.write("END\n")
