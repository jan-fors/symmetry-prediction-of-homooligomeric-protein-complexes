#!/bin/bash
#SBATCH --job-name=monomer_extraction
#SBATCH --output=/work2/mb92adej-homomultimerprediction/monomer_extraction.out
#SBATCH --error=/work2/mb92adej-homomultimerprediction/monomer_extraction.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --partition=paula
#SBATCH --mail-user=mb92adej@studserv.uni-leipzig.de
#SBATCH --mail-type=FAIL

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hmpred

# if base still shadows the env, prepend the env explicitly:
export PATH="$CONDA_PREFIX/bin:$PATH"
# which python    # must now be envs/hmpred

HOME_DIR=/home/sc.uni-leipzig.de/mb92adej/homomultimerprediction
WORK_DIR=/work2/mb92adej-homomultimerprediction

export TORCH_HOME=$WORK_DIR/torch_cache
mkdir -p "$TORCH_HOME"

$ASSEMBLIES=$WORK_DIR/dat/raw/assembly/fullset
$OUTPUT_DIR=$WORK_DIR/dat/raw/monomer

bash $HOME_DIR/src/data_pipeline/sourcing/monomer/source_monomers.sh $ASSEMBLIES $OUTPUT_DIR