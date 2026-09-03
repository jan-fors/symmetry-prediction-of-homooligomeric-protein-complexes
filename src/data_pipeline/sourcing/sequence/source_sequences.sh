#!/bin/bash

INPUT_CSV=$1
OUTPUT_DIR=$2

python -m src.data_pipeline.sourcing.sequence.batch_download_fastas $INPUT_CSV $OUTPUT_DIR