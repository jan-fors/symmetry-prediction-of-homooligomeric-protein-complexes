#!/bin/bash

SRC_DIR=$1 # Path to an assembly folder
DST_DIR=$2

for assembly in $SRC_DIR/*; do
    python -m  src.data_pipeline.sourcing.interface.get_interfaces_for_assemblies $assembly -o $DST_DIR 
done