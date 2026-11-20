"""
Script to generate artificial data.
Artificial data should look like esm2 embeddings.

The script mimics the output of the official ESM ``extract.py`` script for the
model ``esm2_t33_650M_UR50D`` (33 layers, embedding dimension 1280). Every
``.pt`` file contains a dictionary of the form

    {
        "label": "1ABC_A",
        "representations": {33: tensor of shape (L, 1280)},   # per residue
        "mean_representations": {33: tensor of shape (1280,)},  # per chain
    }

where L is a random sequence length. The numbers are random and carry no
biological meaning; only the structure, shapes and dtype match real data.

Output layout:

    example/
    ├── metadata.csv              CHAINID, CLUSTER, SYMM
    ├── features/
    │   └── <RCSB_CODE>/<CHAINID>.pt
    └── split/
        ├── train.txt             one cluster number per line
        ├── val.txt
        └── test.txt

Usage:
    python generate_artificial_data.py --n-samples 100
    python generate_artificial_data.py --n-samples 500 --n-clusters 50 --seed 1
"""
# imports
import argparse
import csv
import random
import string
from pathlib import Path

import torch

# constants
METADATA_FILE = Path("example/metadata.csv")
SPLIT_DIR = Path("example/split")
FEATURE_DIR = Path("example/features")

ESM2_LAYER = 33          # last layer of esm2_t33_650M_UR50D
EMBEDDING_DIM = 1280     # embedding size of esm2_t33_650M_UR50D
MIN_SEQ_LEN = 50         # shortest artificial chain
MAX_SEQ_LEN = 500        # longest artificial chain
EMBEDDING_STD = 0.25     # rough scale of real ESM2 activations

MAX_CHAINS_PER_ENTRY = 4     # chains A, B, C, D per artificial PDB entry
MAX_SYMMETRIES_PER_SAMPLE = 3

# C1 ... C12 and D2, D4, D6, D8
SYMMETRIES = [f"C{i}" for i in range(1, 13)] + [f"D{i}" for i in range(2, 9, 2)]

SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SUFFIX = ".txt"


# functions
def generate_rcsb_codes(n_codes, rng):
    """
    Create ``n_codes`` unique fake RCSB/PDB codes.

    Real PDB codes have 4 characters: a digit from 1-9 followed by three
    alphanumeric characters (e.g. ``1ABC``). Upper case is used throughout.
    """
    alphabet = string.ascii_uppercase + string.digits
    codes = set()
    while len(codes) < n_codes:
        code = rng.choice("123456789") + "".join(rng.choices(alphabet, k=3))
        codes.add(code)
    return sorted(codes)


def create_sample(rcsb_code, chain, n_clusters, rng, generator):
    """
    Create one artificial sample (one protein chain).

    Args:
        rcsb_code: fake PDB code, used as the sub folder name.
        chain: chain letter, e.g. ``"A"``.
        n_clusters: number of available clusters; the sample gets a random one.
        rng: ``random.Random`` instance for metadata randomness.
        generator: ``torch.Generator`` for the embedding values.

    Returns:
        dict with the metadata (rcsb_code, chain_id, cluster, symm) and the
        embedding dictionary in ESM2 ``extract.py`` format.
    """
    chain_id = f"{rcsb_code}_{chain}"
    seq_len = rng.randint(MIN_SEQ_LEN, MAX_SEQ_LEN)

    # per residue embeddings of layer 33, shape (L, 1280), float32 like ESM2
    per_residue = torch.randn(seq_len, EMBEDDING_DIM, generator=generator) * EMBEDDING_STD

    embedding = {
        "label": chain_id,
        "representations": {ESM2_LAYER: per_residue},
        "mean_representations": {ESM2_LAYER: per_residue.mean(dim=0)},
    }

    n_symm = rng.randint(1, MAX_SYMMETRIES_PER_SAMPLE)
    symm = " ".join(rng.sample(SYMMETRIES, k=n_symm))

    return {
        "rcsb_code": rcsb_code,
        "chain_id": chain_id,
        "cluster": rng.randrange(n_clusters),
        "symm": symm,
        "embedding": embedding,
    }


def save_embedding(sample, feature_dir=FEATURE_DIR):
    """
    Save the embedding of one sample as a ``.pt`` file.

    The path follows the convention used when loading:
        embedding_file_path = esm2_dir / Path(rcsb_code) / Path(chain_id + ".pt")

    Returns:
        Path of the written file.
    """
    file_path = feature_dir / Path(sample["rcsb_code"]) / Path(sample["chain_id"] + ".pt")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sample["embedding"], file_path)
    return file_path


def create_split(samples, rng, split_dir=SPLIT_DIR, fractions=SPLIT_FRACTIONS):
    """
    Split the data by cluster into train, val and test.

    Whole clusters are assigned to one split, so chains of the same cluster
    never end up in different splits (no data leakage). Each output file
    contains one cluster number per line.

    Returns:
        dict mapping split name to the list of cluster numbers.
    """
    clusters = sorted({sample["cluster"] for sample in samples})
    rng.shuffle(clusters)

    n_total = len(clusters)
    n_val = int(round(n_total * fractions["val"]))
    n_test = int(round(n_total * fractions["test"]))
    # make sure val and test are not empty if there are enough clusters
    if n_total >= 3:
        n_val = max(n_val, 1)
        n_test = max(n_test, 1)

    splits = {
        "val": clusters[:n_val],
        "test": clusters[n_val:n_val + n_test],
        "train": clusters[n_val + n_test:],
    }

    split_dir.mkdir(parents=True, exist_ok=True)
    for name, split_clusters in splits.items():
        with open(split_dir / f"{name}{SPLIT_SUFFIX}", "w") as handle:
            handle.writelines(f"{cluster}\n" for cluster in sorted(split_clusters))
    return splits


def write_metadata_file(samples, metadata_file=METADATA_FILE):
    """
    Write the metadata CSV with the columns CHAINID, CLUSTER and SYMM.

    SYMM can contain several symmetries separated by a single whitespace,
    e.g. ``"C2 D4"``.
    """
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["CHAINID", "CLUSTER", "SYMM"])
        for sample in samples:
            writer.writerow([sample["chain_id"], sample["cluster"], sample["symm"]])


def parse_args():
    parser = argparse.ArgumentParser(description="Generate artificial ESM2-like embeddings.")
    parser.add_argument("--n-samples", type=int, required=True,
                        help="number of chains (= .pt files) to generate")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="number of clusters (default: n_samples // 5, at least 3)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_samples < 1:
        raise ValueError("--n-samples must be at least 1")
    n_clusters = args.n_clusters or max(3, args.n_samples // 5)

    rng = random.Random(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    # enough fake PDB entries so that every entry has 1 to MAX_CHAINS_PER_ENTRY chains
    rcsb_codes = generate_rcsb_codes(args.n_samples, rng)
    samples = []
    for rcsb_code in rcsb_codes:
        n_chains = rng.randint(1, MAX_CHAINS_PER_ENTRY)
        for chain in string.ascii_uppercase[:n_chains]:
            if len(samples) == args.n_samples:
                break
            sample = create_sample(rcsb_code, chain, n_clusters, rng, generator)
            save_embedding(sample)
            sample.pop("embedding")  # free memory, the tensor is on disk now
            samples.append(sample)
        if len(samples) == args.n_samples:
            break

    splits = create_split(samples, rng)
    write_metadata_file(samples)

    print(f"Wrote {len(samples)} embeddings to {FEATURE_DIR}")
    print(f"Wrote metadata to {METADATA_FILE}")
    for name, split_clusters in splits.items():
        print(f"{name}: {len(split_clusters)} clusters")


if __name__ == "__main__":
    main()