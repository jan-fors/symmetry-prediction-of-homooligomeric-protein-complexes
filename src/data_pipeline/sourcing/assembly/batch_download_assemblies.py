import argparse
import os
from pathlib import Path
from src.io.readers.read_rcsb_ids import read_rcsb_ids
from src.data_pipeline.sourcing.assembly.download_assemblies import fetch_bioassemblies

def batch_download_assemblies(src_file : Path, output_dir : Path):
    """
    """
    # read ids
    rcsb_ids = read_rcsb_ids(src_file)

    # iterate
    for single_id in rcsb_ids:
        fetch_bioassemblies(single_id, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("src_file", type=Path, help="Path to the file containing the rcsb ids to download the assemblies for.")
    parser.add_argument("-o", "--output_dir", type=Path, help="Output directory")

    args = parser.parse_args()

    batch_download_assemblies(args.src_file, args.output_dir)