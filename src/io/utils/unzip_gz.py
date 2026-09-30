import gzip
import shutil
from pathlib import Path

def unzip_gz(file_path : Path) -> None:
    """
    """
    with gzip.open(file_path, 'rb') as f_in:
        with open(str(file_path).split(".")[0], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)