#!/bin/bash

id_file=$1
output_dir=$2

python -m src.data_pipeline.sourcing.assembly.batch_download_assemblies $id_file -o $output_dir