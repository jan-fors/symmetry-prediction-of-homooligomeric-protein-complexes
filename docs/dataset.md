# Dataset
## Base 
Original dataset from seq2symm (http://files.ipd.uw.edu/pub/seq2symm/datasets.zip)
```sh
curl http://files.ipd.uw.edu/pub/seq2symm/datasets.zip -o dataset.zip
unzip dataset.zip
```
Retrieve and download the `.zip` file. Intially use the `homomer_pdbids_hash_clusterid_labels_fullset.csv` file.
RCSB ids are extracted using:
```sh
./datasets/prep_batch_download.sh
```
which returns a `.txt` file (`homomer_pdbids_fullset.txt`).

## Structure Datasets
### Assembly Dataset
The protein assemblies are downloaded using
```sh
./assembly/batch_assembly_download.sh
```
which creates a folder named after the **rcsb id** that contains all the downloaded assemblies.
All of the assemblies are then unpacked using `unpack.sh`

### Monomer Dataset
The monomer dataset is created by extracting all monomer chains from the assemblies. Using the script 
```sh
./extract_from_assemblies.sh 
```
which results in a folder for each rcsb id and subfolders for each assembly.

## Sequence Dataset
The sequence dataset is created using
```sh
./sequence/retrieve.sh
```
with the original `homomer_pdbids_hash_clusterid_labels_fullset.csv` file as input. It sequentially gets the the polymer entity ID first and downloads the sequence. Each sequence is saved as `fasta`. Afterwards, sort the files to get the structure `<rcsb id>/<fasta>` using `sort.sh`.

Since some of the sequences are not retrieved correctly (some of them are nucleotide sequences) a check has to be performed. Therefore *a script* iterates over each folder and check whether or not there is a representative aminoacid sequence. If there is none the sequence is extracted from a monomer structure.

## ESM-2 Dataset
Using the `Sequence Dataset` each Aminoacid Sequence is embedded using the pretrained model `esm2_t33_650M_UR50D`.
```sh
./esm2/embed.sh
```

## Interface Residue Dataset
The Interfaces are calculated based on the `Assembly Dataset`. 
```sh
./interface/calculate_interfaces.sh
```
