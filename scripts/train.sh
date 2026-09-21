#!/usr/bin/env bash
# Train one FoMo model with the paper's configuration and evaluate it on the
# four benchmarks after every epoch (final-epoch numbers are the ones to read).
#
#   bash scripts/train.sh <backbone> [exp_name]
#   backbone: lpips_alex | lpips_vgg | dists | dino | clip | mae | dreamsim
#
#   bash scripts/train.sh dino                    # → experiments/dino/
#   CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh dreamsim my_run
#
# Per-backbone settings:
#   lpips_alex / lpips_vgg / dists : lr 2e-4, fixed 256×256 crops
#   dino / clip / mae              : lr 1e-4, random 64–256 crops
#   dreamsim                       : lr 1e-4, fixed 224×224 crops (fixed-resolution ensemble)
#   shared: 3 epochs, batch 64, weight decay 1e-8, EMA 0.999, bf16
#
# Relaunching the same command resumes from experiments/<exp_name>/ckpt_latest.pt.
# Afterwards:  python evaluate.py --model <name> --checkpoint experiments/<exp_name>/ckpt_latest.pt

set -euo pipefail

backbone="${1:?usage: bash scripts/train.sh <backbone> [exp_name]}"
exp_name="${2:-$backbone}"
DATAROOT="${DATAROOT:-./data/fomo}"
TRAIN_INDEX="${TRAIN_INDEX:-$DATAROOT/index.json}"
BENCHMARK_DIR="${BENCHMARK_DIR:-./benchmarks}"

case "$backbone" in
    lpips_alex|lpips_vgg|dists) lr=2e-4; min_crop=256; max_crop=256 ;;
    dreamsim)                   lr=1e-4; min_crop=224; max_crop=224 ;;
    dino|clip|mae)              lr=1e-4; min_crop=64;  max_crop=256 ;;
    *) echo "unknown backbone: $backbone"; exit 1 ;;
esac

python train.py \
    --exp_name "$exp_name" --backbone "$backbone" \
    --dataroot "$DATAROOT" --train_index_file "$TRAIN_INDEX" \
    --epochs 3 --batch_size 64 --lr "$lr" --weight_decay 1e-8 \
    --ema_decay 0.999 --amp bf16 \
    --min_crop_size "$min_crop" --max_crop_size "$max_crop" \
    --val pipal tid2013 csiq live --benchmark_dir "$BENCHMARK_DIR"
