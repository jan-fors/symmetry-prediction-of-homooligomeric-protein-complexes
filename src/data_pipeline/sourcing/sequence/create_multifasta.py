import argparse
from pathlib import Path
import os
from Bio import SeqIO

def create_multifasta_from_dir(sequence_dir : Path, output_dir : Path):
    """
    """
    seqrecords = []
    for rcsb_dir in os.listdir(sequence_dir):
        for fasta in os.listdir(sequence_dir/Path(rcsb_dir)):
            sequence_name = fasta.split(".")[0]
            fasta_path = sequence_dir/Path(rcsb_dir)/Path(fasta)

            for seq_record in SeqIO.parse(fasta_path, "fasta"):
                seqrecords.append(seq_record)

    output_path = output_dir / Path(sequence_dir.stem + ".fasta")
    SeqIO.write(seqrecords, output_path, "fasta")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_dir", type=Path)
    parser.add_argument("-o", "--output_dir", type=Path, default=".")
    args = parser.parse_args()

    create_multifasta_from_dir(args.sequence_dir, args.output_dir)