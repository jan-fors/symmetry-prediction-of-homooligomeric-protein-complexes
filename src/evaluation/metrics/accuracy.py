from sklearn.metrics import accuracy_score 
import numpy as np

def accuracy(y_pred : np.array, y_true : np.array) -> float:
    """
    """
    return accuracy_score(y_true, y_pred)