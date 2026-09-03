#!/bin/bash

SRC_DIR=$1
DST_DIR=$2

for directory in $SRC_DIR/*; do 
    echo $directory
    for fasta in $directory/*.fasta; do
        echo $fasta
        python -m src.data_pipeline.sourcing.esm2.embed $fasta -o $DST_DIR
    done
done