from src.utils.constants import (
    VERBOSE
)
from datetime import datetime

def v_print(content : str, verbose : bool = VERBOSE):
    """
    """
    if verbose:
        print(f"[{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}]\t{content}")