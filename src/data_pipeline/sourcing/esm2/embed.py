import torch
import esm
import argparse
from pathlib import Path
from Bio import SeqIO
import pickle
import os
from src.utils.constants import (
    ESM2_MODEL,
    EXPORTED_LAYERS
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def embed_sequence(sequence: str):
    print(f"Start to embed sequence:\n{sequence}")
    model, alphabet = ESM2_MODEL
    model = model.to(device)          # model on GPU
    model.eval()

    batch_converter = alphabet.get_batch_converter()
    data = [("my_protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    batch_tokens = batch_tokens.to(device)   # input on GPU

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[EXPORTED_LAYERS])

    token_embeddings = results["representations"][EXPORTED_LAYERS]
    sequence_embedding = token_embeddings[0, 1:-1]

    print("Embedding complete.")

    return sequence_embedding.detach().cpu()


def read_sequence_from_fasta(fasta: Path):
    """ """
    data = SeqIO.read(fasta, "fasta")

    return data.seq


def save_embedding(embedding, name, output_dir):
    """ """
    output_dir = output_dir / Path(name.split("_")[0]) # TODO make more beautyful
    
    os.makedirs(output_dir, exist_ok = True)
    with open(output_dir / Path(name + ".pkl"), "wb") as f:  # open a text file
        pickle.dump(embedding, f)  # serialize the list


def main(fasta: Path, output_dir: Path):
    """ """
    if not os.path.exists(fasta):
        print(f"Path {fasta} does not exist")
    else:
        sequence = read_sequence_from_fasta(fasta)

        embedding = embed_sequence(sequence)

        fasta_name = fasta.stem

        save_embedding(embedding, fasta_name, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output_dir", type=Path, default=".")
    args = parser.parse_args()

    main(args.input, args.output_dir)
