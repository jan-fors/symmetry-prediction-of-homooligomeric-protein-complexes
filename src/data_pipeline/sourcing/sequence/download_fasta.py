"""
"""
import argparse
import requests
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import json
import os
from pathlib import Path

def get_sequence(rcsb_id : str, chain : str) -> str:
    """
    """
    for i in range(1,10):
        # get entity type
        response = requests.get(f"https://data.rcsb.org/rest/v1/core/polymer_entity/{rcsb_id}/{i}")

        if response.status_code == 200:
            raw = response.text
            data = json.loads(raw)

     

            # read type
            rcsb_entity_polymer_type = data["entity_poly"]["rcsb_entity_polymer_type"]
            asym_ids = data["rcsb_polymer_entity_container_identifiers"]["asym_ids"]

            if rcsb_entity_polymer_type == "Protein" and chain in asym_ids:
                sequence = data["entity_poly"]["pdbx_seq_one_letter_code_can"]

                if sequence == None:
                    sequence = data["entity_poly"]["pdbx_seq_one_letter_code"]

                if sequence == None:
                    continue

                return sequence
            else:
                print("Non protein entity or wrong chain")
                continue

        else:
            print("Request did not work.")
            continue

def download_fasta(rcsb_id : str, chain : str, output_dir : Path):
    """
    """
    sequence = get_sequence(rcsb_id, chain)

    if sequence == None:
        print(f"No Sequence found for {rcsb_id}, {chain}")
    else:
        output_dir = output_dir / Path(rcsb_id)
        os.makedirs(output_dir, exist_ok = True)
        output_path = output_dir / Path(rcsb_id+"_"+chain+".fasta")

        # Create a record
        record = SeqRecord(Seq(sequence), id=rcsb_id+"_"+chain, description="")

        # Save to file
        SeqIO.write(record, output_path, "fasta")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rcsb_id", type=str)
    parser.add_argument("chain", type=str)
    parser.add_argument("-o", "--output", type=Path, default=".")
    args = parser.parse_args()

    download_fasta(args.rcsb_id, args.chain, args.output)


