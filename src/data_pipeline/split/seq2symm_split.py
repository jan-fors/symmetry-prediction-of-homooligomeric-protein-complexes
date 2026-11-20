"""
Reads the seq2symm splits and creates the needed lists.
"""
import pickle
from pathlib import Path
import os
from datetime import datetime

SEQ2SYMM_PATH = Path("dat/source/datasets/train_val_test_splits.pkl")

with open(SEQ2SYMM_PATH, "rb") as f:
    data = pickle.load(f)

print(data.keys())
train_cl = data["train_clusterids"]
train_cl = [str(x) for x in train_cl]

val_cl = data["validation_clusterids"]
val_cl = [str(x) for x in val_cl]

test_cl = data["test_clusterids"]
test_cl = [str(x) for x in test_cl]

all_cl_len = len(train_cl) + len(val_cl) + len(test_cl)

print(f"Clusters in train: {len(train_cl)}\nClusters in val: {len(val_cl)}\nClusters in test: {len(test_cl)}")

# save
output_dir = Path("dat/split/seq2symm_split")
os.makedirs(output_dir, exist_ok=True)

train_cl_out = "\n".join(train_cl)
train_path = output_dir / Path("train.txt")
with open(train_path, "a+") as f:
    f.write(train_cl_out)

val_cl_out = "\n".join(val_cl)
val_path = output_dir / Path("val.txt")
with open(val_path, "a+") as f:
    f.write(val_cl_out)

test_cl_out = "\n".join(test_cl)
test_path = output_dir / Path("test.txt")
with open(test_path, "a+") as f:
    f.write(test_cl_out)

info_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
info_str += f"Performing split: {len(train_cl)/all_cl_len*100}/{len(val_cl)/all_cl_len*100}/{len(test_cl)/all_cl_len*100}\n" 
info_str += f"Read {all_cl_len} cluster\n"
info_str += f"Clusters in train: {len(train_cl)}\nClusters in val: {len(val_cl)}\nClusters in test: {len(test_cl)}"
info_path = output_dir / Path("INFO")
with open(info_path, "a+") as f:
    f.write(info_str)