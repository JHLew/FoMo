#!/usr/bin/env bash
# Download the released 480k FoMo training dataset from the HuggingFace Hub
# (https://huggingface.co/datasets/JHLew/fomo_480k — 240 tar shards, ~550 GB)
# and unpack it into the folder tree + index.json that train.py consumes.
# The unpacker (unpack.py) and index ship with the dataset itself.
#
# Usage:
#   bash scripts/download_fomo_480k.sh
#
# Disk usage: ~550 GB for the shards + ~550 GB unpacked. The unpack step
# deletes each shard after extraction (--remove-shards), so peak usage stays
# around 1.1 TB and final usage ~550 GB. Remove that flag to keep the shards.
#
# Override the source repo with FOMO_DATA_REPO=<user/repo>.

set -euo pipefail

FOMO_DATA_REPO="${FOMO_DATA_REPO:-JHLew/fomo_480k}"
SHARD_DIR="${SHARD_DIR:-./data/fomo_shards}"
TARGET="${TARGET:-./data/fomo}"

mkdir -p "$SHARD_DIR"
huggingface-cli download "$FOMO_DATA_REPO" --repo-type dataset --local-dir "$SHARD_DIR"

python "$SHARD_DIR/unpack.py" \
    --shards "$SHARD_DIR" \
    --out    "$TARGET" \
    --remove-shards

echo "Done. Dataset at $TARGET (index: $TARGET/index.json)."
