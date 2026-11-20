"""
Extracts and saves one chain from a given homomultimer assembly

Needs to check if they are all the same!
"""
import os
import argparse
from pathlib import Path
import gzip
from Bio.PDB.MMCIFParser import MMCIFParser
import gemmi

def extract_monomer(input_path : Path, chain_id : str, output_dir : Path, pdb : bool = False):
    """
    Code from https://github.com/BioinfoMachineLearning/CAF3/blob/main/monomer/utils/extract_cif_chains.py
    """
    rcsb_id = str(input_path.stem).split("-")[0]
    assembly_nr = str(input_path.stem).split("-")[1].split(".")[0]
    output_path = output_dir / Path(rcsb_id) / Path(assembly_nr) 
    os.makedirs(output_path, exist_ok=True)
    output_path = output_path / Path(rcsb_id + "_" + chain_id + ".cif")


    atom_loop_started = False
    atom_headers = []
    atom_data_start = False

    if str(input_path).endswith(".gz"):

        with gzip.open(input_path,'rt') as infile, open(output_path, 'w') as outfile:
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
    else:
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

    if pdb:
        structure = gemmi.read_structure(str(output_path))
        pdb_path = str(output_path).replace(".cif", ".pdb")
        structure.write_pdb(pdb_path)


def extract_all_monomers(input_path : Path, output_dir : Path, pdb : bool = False):
    """
    """
    # read all chains
    parser = MMCIFParser()

    if str(input_path).endswith(".gz"):
        with gzip.open(input_path, "rt") as f:
            structure = parser.get_structure("structure", f)
    elif str(input_path).endswith(".cif"):
        structure = parser.get_structure("structure", input_path)
    else:
        exit(0)

    model = structure[0]
    for chain in model:
        extract_monomer(input_path=input_path, chain_id=chain.id, output_dir=output_dir, pdb=pdb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_assembly", type=Path)
    parser.add_argument("-c", "--chain", type=str, default=None)
    parser.add_argument("-o", "--output", type=Path, default=".")
    parser.add_argument("--pdb", action="store_true")
    args = parser.parse_args()

    if args.chain != None:
        extract_monomer(args.input_assembly, args.chain, args.output, args.pdb)
    else:
        extract_all_monomers(args.input_assembly, args.output, args.pdb)