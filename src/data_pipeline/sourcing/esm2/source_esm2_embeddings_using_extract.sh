#!/bin/bash

INPUT_FASTA=$1
OUTPUT_DIR=$2
esm-extract esm2_t33_650M_UR50D $INPUT_FASTA $OUTPUT_DIR --repr_layers 33