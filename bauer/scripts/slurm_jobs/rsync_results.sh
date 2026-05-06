#!/bin/bash
# Rsync cluster fit outputs to local repo for plotting.
# Run this LOCALLY (not on cluster). Requires sciencecluster ssh alias.

set -e
LOCAL_DEST="${1:-$HOME/git/bauer/notebooks/results_cluster}"
mkdir -p "$LOCAL_DEST"

echo "rsync /shares/zne.uzh/gdehol/bauer_results/ -> $LOCAL_DEST"
rsync -avz --info=progress2 \
    --include='*/' \
    --include='*.nc' \
    --include='*.parquet' \
    --exclude='*' \
    sciencecluster:/shares/zne.uzh/gdehol/bauer_results/ \
    "$LOCAL_DEST/"

echo "done. files at $LOCAL_DEST"
