"""
Extracts and saves one chain from a given homomultimer assembly

Needs to check if they are all the same!
"""
import os
import argparse
from pathlib import Path
from Bio.PDB.MMCIFParser import MMCIFParser

def extract_monomer(input_path : Path, chain_id : str, output_dir : Path):
    """
    Code from https://github.com/BioinfoMachineLearning/CAF3/blob/main/monomer/utils/extract_cif_chains.py
    """
    rcsb_id = str(input_path.stem).split("-")[0]
    assembly_nr = str(input_path.stem).split("-")[1]
    output_path = output_dir / Path(rcsb_id) / Path(assembly_nr) 
    os.makedirs(output_path, exist_ok=True)
    output_path = output_path / Path(rcsb_id + "_" + chain_id + ".cif")

    atom_loop_started = False
    atom_headers = []
    atom_data_start = False

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            # Skip HETATM lines
            if line.startswith("HETATM"):
                continue

            # Detect beginning of atom_site loop
            if line.strip() == "loop_":
                atom_loop_started = True
                atom_headers = []
                atom_data_start = False
                outfile.write(line)
                continue

            # If in atom loop, collect headers
            if atom_loop_started and line.strip().startswith("_atom_site."):
                atom_headers.append(line.strip())
                outfile.write(line)
                continue

            # End of header block = start of data
            if atom_loop_started and not line.strip().startswith("_") and not atom_data_start:
                atom_data_start = True

            # If in atom_site data
            if atom_data_start:
                tokens = line.strip().split()
                if len(tokens) != len(atom_headers):
                    outfile.write(line)  # malformed or non-atom line
                    continue

                # Find chain ID column dynamically
                try:
                    # chain_col = [h for h in atom_headers if "_atom_site.label_asym_id" in h][0]
                    chain_col = [h for h in atom_headers if "_atom_site.auth_asym_id" in h][0]
                    chain_idx = atom_headers.index(chain_col)
                    if tokens[chain_idx] == chain_id:
                        outfile.write(line)
                except IndexError:
                    continue  # Skip if chain index is out of range
                continue

            # Default: write all non-ATOM lines
            outfile.write(line)

def extract_all_monomers(input_path : Path, output_dir : Path):
    """
    """
    # read all chains
    parser = MMCIFParser()
    structure = parser.get_structure("structure", input_path)
    model = structure[0]
    for chain in model:
        extract_monomer(input_path=input_path, chain_id=chain.id, output_dir=output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_assembly", type=Path)
    parser.add_argument("-c", "--chain", type=str, default=None)
    parser.add_argument("-o", "--output", type=Path, default=".")
    args = parser.parse_args()

    if args.chain != None:
        extract_monomer(args.input_assembly, args.chain, args.output)
    else:
        extract_all_monomers(args.input_assembly, args.output)