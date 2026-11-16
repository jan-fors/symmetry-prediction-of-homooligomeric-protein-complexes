"""
"""

def get_correct_suggest(trial, type_str: str):
    """
    Returns the correct suggestions function of optuna for a given data type.
    """
    if type_str == "int":
        return trial.suggest_int

    if type_str == "float":
        return trial.suggest_float

    if type_str == "categorical":
        return trial.suggest_categorical