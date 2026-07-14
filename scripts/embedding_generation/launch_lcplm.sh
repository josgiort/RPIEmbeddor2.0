#!/bin/bash
#SBATCH --job-name=lcplm_rnainteract
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/lcplm_rnainteract_%j.log
#SBATCH --error=logs/lcplm_rnainteract_%j.err

echo "Start: $(date)"
module load devel/miniforge
source activate rpi
cd /gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/rpi-main_4/dataset/embeddings/embed_framework

python extract_embeddings.py \
    --model_type lcplm \
    --enable_cuda \
    --unique_seq_path data/annotations/unique_proteins_final.parquet \
    --emb_dir data/embeddings/eCLIP_2/RNAInterAct \
    --working_dir /gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/rpi-main_4

echo "Done: $(date)"