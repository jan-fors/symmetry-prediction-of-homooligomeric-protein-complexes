"""
"""
import argparse
from pathlib import Path
import pandas as pd
from src.data_pipeline.sourcing.sequence.download_fasta import download_fasta

def batch_download(input_csv : Path, output_dir : Path):
    """
    """
    df = pd.read_csv(input_csv)
    ids = df["CHAINID"].to_list()

    for id in ids:
        print(id)
        rcsb_id = id.split("_")[0]
        chain = id.split("_")[1]

        download_fasta(rcsb_id, chain, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type = Path)
    parser.add_argument("output_dir", type = Path)
    args = parser.parse_args()

    batch_download(input_csv=args.input_csv, output_dir=args.output_dir)

