import argparse
from src.io.utils.unzip_gz import unzip_gz
from pathlib import Path
import os

def batch_unpack(directory : Path) -> None:
    """
    """
    for subdir in os.listdir(directory):
        for compressed_file in os.listdir(directory/Path(subdir)):
            file_path = directory/Path(subdir)/Path(compressed_file)

            unzip_gz(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("directory", type=Path)

    args =  parser.parse_args()

    batch_unpack(args.directory)