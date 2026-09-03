#!/bin/bash

SRC_DIR=$1 # has to be an assembly directory
DST_DIR=$2 

for assembly in $SRC_DIR/*; do

    for file in $SRC_DIR/$(basename $assembly)/*; do # file is input
        python -m src.data_pipeline.sourcing.monomer.extract_monomers $file -o $DST_DIR
    done

done