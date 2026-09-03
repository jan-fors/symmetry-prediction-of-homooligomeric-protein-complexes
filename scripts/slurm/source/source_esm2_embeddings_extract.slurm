#!/bin/bash
#SBATCH --job-name=esm2_embeddings
#SBATCH --output=/work2/mb92adej-homomerprediction/esm2.out
#SBATCH --error=/work2/mb92adej-homomerprediction/esm2.err
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

# python -c "import sys; print(sys.executable)"
# python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# nvidia-smi

WORK_DIR=/work2/mb92adej-homomerprediction

export TORCH_HOME=$WORK_DIR/torch_cache
mkdir -p "$TORCH_HOME"

SEQUENCES=$WORK_DIR/homomultimerprediction/dat/raw/sequences/fullset.fasta
OUTPUT=$WORK_DIR/homomultimerprediction/dat/processed/esm2/esm2_t33_650M_UR50D/fullset

bash scripts/source/source_esm2_embeddings_using_extract.sh $SEQUENCES $OUTPUT


