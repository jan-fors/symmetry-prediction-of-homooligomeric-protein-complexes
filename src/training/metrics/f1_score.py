
import numpy as np
from typing import Literal
from sklearn.metrics import f1_score as f1

def f1_score(y_true : np.array, y_pred : np.array, average : Literal["micro", "macro", "samples", "weighted", "binary"] = "binary") -> float:
    """
    """
    return f1(y_true, y_pred, average=average)