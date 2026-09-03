import torch
from pathlib import Path

def save_model(model, output_dir):
    """
    """
    output_path = output_dir / Path("best_model.pkl")
    torch.save(model.state_dict(), output_path)

    return output_path