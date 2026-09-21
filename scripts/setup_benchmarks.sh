#!/usr/bin/env bash
# Download and extract the evaluation benchmarks (PIPAL, TID2013, CSIQ, LIVE)
# into ./benchmarks/, from the chaofengc/IQA-PyTorch-Datasets mirror on the
# HuggingFace Hub.
#
# Usage:
#   bash scripts/setup_benchmarks.sh

set -euo pipefail

HF_DIR="${HF_DIR:-./hf_iqa}"
TARGET="${TARGET:-./benchmarks}"

huggingface-cli download chaofengc/IQA-PyTorch-Datasets --repo-type dataset \
    --include "pipal.tar" "tid2013.tgz" "csiq.tgz" "live.tgz" \
    --local-dir "$HF_DIR"

mkdir -p "$TARGET"
tar -xf  "$HF_DIR/pipal.tar"   -C "$TARGET"
tar -xzf "$HF_DIR/tid2013.tgz" -C "$TARGET"
tar -xzf "$HF_DIR/csiq.tgz"    -C "$TARGET"
tar -xzf "$HF_DIR/live.tgz"    -C "$TARGET"

echo "Done. Benchmarks extracted under $TARGET/:"
ls "$TARGET"
