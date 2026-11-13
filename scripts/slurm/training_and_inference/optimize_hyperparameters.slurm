#!/bin/bash
#SBATCH --job-name=hmpred_hyperparamemter
#SBATCH --output=/work2/mb92adej-homomultimerprediction/logs/hyperparameter/%x-%j.out
#SBATCH --error=/work2/mb92adej-homomultimerprediction/logs/hyperparameter/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --partition=clara
#SBATCH --mail-user=mb92adej@studserv.uni-leipzig.de
#SBATCH --mail-type=FAIL,END

# load env
module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hmpred

export PATH="$CONDA_PREFIX/bin:$PATH" # TODO can be removed?

CONFIG_DIR_PATH=/home/sc.uni-leipzig.de/mb92adej/homomultimerprediction/configs/hyperparameter/cluster
CONFIG_NAME=$1

python -m src.optimize_hyperparameters $CONFIG_DIR_PATH/$CONFIG_NAME