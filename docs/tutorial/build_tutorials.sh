#!/usr/bin/env bash
# build_tutorials.sh — regenerate, execute, and render all tutorial notebooks.
#
# Usage:
#   ./docs/tutorial/build_tutorials.sh           # default: generate + execute + build docs
#   ./docs/tutorial/build_tutorials.sh --no-exec # skip notebook execution (uses cached outputs)
#
# Run this script from the repo root before committing tutorial changes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TUTORIAL_DIR="$REPO_ROOT/docs/tutorial"
TIMEOUT=1200   # seconds per notebook (20 min — sampling is slow)

EXECUTE=true
for arg in "$@"; do
  [[ "$arg" == "--no-exec" ]] && EXECUTE=false
done

echo "=== Step 1: Regenerate .ipynb files from make_notebooks.py ==="
cd "$TUTORIAL_DIR"
python make_notebooks.py

if $EXECUTE; then
  echo ""
  echo "=== Step 2: Execute notebooks (timeout ${TIMEOUT}s each) ==="
  for nb in lesson1.ipynb lesson2.ipynb lesson3.ipynb lesson4.ipynb; do
    echo "  Executing $nb ..."
    jupyter nbconvert \
      --to notebook \
      --execute \
      --inplace \
      --ExecutePreprocessor.timeout=$TIMEOUT \
      "$nb"
    echo "  Done: $nb"
  done
else
  echo ""
  echo "=== Step 2: Skipped (--no-exec) — using existing outputs ==="
fi

echo ""
echo "=== Step 3: Build Sphinx HTML ==="
cd "$REPO_ROOT/docs"
make html

echo ""
echo "=== Done ==="
echo "HTML output: $REPO_ROOT/docs/_build/html"
echo ""
echo "To publish: git add docs/tutorial/*.ipynb && git commit && git push origin main"
