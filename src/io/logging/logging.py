import logging
from pathlib import Path

def init_logging(exp_out_dir : Path):
    """
    """
    logging.basicConfig(
            filename=exp_out_dir/Path("run.log"),
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            encoding="utf-8"
        )