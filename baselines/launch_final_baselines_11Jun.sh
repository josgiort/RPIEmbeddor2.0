#!/bin/bash
#SBATCH --job-name=baselines_rpi
#SBATCH --time=32:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%A_%a.out
#SBATCH --error=logs/error_%A_%a.err
#SBATCH --array=0-35

SEEDS=(2727 5924 8137)

TASKS=(
"cnn_onehot_PROTxRNA/CNN_eCLIP2_protdisj.py"
"cnn_onehot_PROTxRNA/CNN_eCLIP2_protdistr.py"
"cnn_onehot_PROTxRNA/CNN_rnainter_random.py"
"cnn_onehot_PROTxRNA/CNN_rnainter_rnafamdisj.py"

"MLP/mlp_eCLIP2_protdisj.py"
"MLP/mlp_eCLIP2_protdistr.py"
"MLP/mlp_rnainter_random.py"
"MLP/mlp_rnainter_rnafamdisj.py"

"logreg/bseln_logreg_eCLIP2_protdisj.py"
"logreg/bseln_logreg_eCLIP2_protdistr.py"
"logreg/bseln_logreg_rnainter_randm.py"
"logreg/bseln_logreg_rnainter_rnafamdisj.py"
)

N_SEEDS=3

SCRIPT_INDEX=$((SLURM_ARRAY_TASK_ID / N_SEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % N_SEEDS))

SCRIPT=${TASKS[$SCRIPT_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "Running:"
echo "  Script = ${SCRIPT}"
echo "  Seed   = ${SEED}"

module purge
module load devel/miniforge
source activate rpi

cd /gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/baselines_final

python "${SCRIPT}" --seed "${SEED}"
