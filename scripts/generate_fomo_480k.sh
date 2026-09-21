#!/usr/bin/env bash
# Generate the full 480k-sample FoMo training set with FLUX.1-dev:
#
#   240,000 scratch samples   (tag FLUX — base image generated from a blank
#                              canvas at strength=1.0, then 2 img2img variants)
#   240,000 ImageNet samples  (tag IN1K — base image is an ImageNet-1k crop,
#                              then 2 img2img variants)
#
# Each sample directory holds base.png + <t1>.png + <t2>.png, where t1/t2 are
# the denoising timesteps (1–50) the variants were generated at. Both stages
# append to the same index.json, which is what train.py consumes.
#
# NOTE: this is a very large job (~2–4 s per pipeline call on an L40; each
# sample needs 2–3 calls → several GPU-months for the full 480k). Shard it
# across GPUs/machines by splitting the --idx-from/--idx-to range; every shard
# can safely append to the same index because samples are keyed by index.
# The pre-generated dataset is available on the HuggingFace Hub
# (scripts/download_fomo_480k.sh) — regenerating from scratch is only needed
# if you want a fresh dataset.
#
# Usage:
#   bash scripts/generate_fomo_480k.sh                    # both stages, full range
#   STAGE=scratch IDX_FROM=0 IDX_TO=59999 bash scripts/generate_fomo_480k.sh
#   STAGE=in1k    IDX_FROM=0 IDX_TO=59999 bash scripts/generate_fomo_480k.sh
#
# FLUX.1-dev is a gated model: accept the license on HuggingFace and run
# `huggingface-cli login` first. The IN1K stage needs the ImageNet-1k train
# split at $IMAGENET_ROOT/train/<class>/*.JPEG.

set -euo pipefail

OUTPUTS="${OUTPUTS:-./data/fomo}"
IMAGENET_ROOT="${IMAGENET_ROOT:-./imagenet-1k}"
STAGE="${STAGE:-both}"           # scratch | in1k | both
IDX_FROM="${IDX_FROM:-0}"
IDX_TO="${IDX_TO:-239999}"

RESOLUTION="512 512"
NUM_STEPS=50
N_PER_SAMPLE=2
GUIDANCE_SCALE=3.5

run_scratch() {
    python generate_fomo_dataset.py \
        --outputs        "$OUTPUTS" \
        --split          train \
        --tag            FLUX \
        --resolution     $RESOLUTION \
        --num-steps      $NUM_STEPS \
        --n-per-sample   $N_PER_SAMPLE \
        --guidance-scale $GUIDANCE_SCALE \
        --img-format     png \
        --idx-from       "$IDX_FROM" \
        --idx-to         "$IDX_TO" \
        --index-file     index.json
}

run_in1k() {
    python generate_fomo_dataset.py \
        --ref-images     "$IMAGENET_ROOT" \
        --outputs        "$OUTPUTS" \
        --split          train \
        --tag            IN1K \
        --resolution     $RESOLUTION \
        --num-steps      $NUM_STEPS \
        --n-per-sample   $N_PER_SAMPLE \
        --guidance-scale $GUIDANCE_SCALE \
        --img-format     png \
        --idx-from       "$IDX_FROM" \
        --idx-to         "$IDX_TO" \
        --index-file     index.json
}

case "$STAGE" in
    scratch) run_scratch ;;
    in1k)    run_in1k ;;
    both)    run_scratch; run_in1k ;;
    *) echo "Unknown STAGE=$STAGE (use scratch | in1k | both)"; exit 1 ;;
esac
