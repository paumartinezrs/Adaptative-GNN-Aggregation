#!/bin/bash
# ==============================================================================
# Grid-search wrapper for benchmark_training_data.py
#
# Runs the quick benchmark for every (feat_dim × out_dim) combination so that
# the classifier training data covers the full dimensionality space.
#
# Usage:
#   bash scripts/run_grid_search.sh /path/to/edgelist/dir [output.csv]
# ==============================================================================
set -e

GRAPHS_DIR="${1:?Usage: $0 <graphs_dir> [output_csv]}"
OUTPUT_CSV="${2:-outputs/synthetic_benchmark.csv}"

FEAT_DIMS=(128 256 512 1024 2048 4096 8192)
OUT_DIMS=(2 4 8 16 32 64 128)

TOTAL=$(( ${#FEAT_DIMS[@]} * ${#OUT_DIMS[@]} ))
CUR=0

echo "============================================================"
echo " GRID SEARCH – Quick Benchmark"
echo "============================================================"
echo " Graphs dir : $GRAPHS_DIR"
echo " Output CSV : $OUTPUT_CSV"
echo " feat_dims  : ${FEAT_DIMS[*]}"
echo " out_dims   : ${OUT_DIMS[*]}"
echo " Total runs : $TOTAL"
echo "============================================================"
echo

mkdir -p "$(dirname "$OUTPUT_CSV")"

for FD in "${FEAT_DIMS[@]}"; do
    for OD in "${OUT_DIMS[@]}"; do
        CUR=$(( CUR + 1 ))
        echo "── [$CUR/$TOTAL] feat_dim=$FD  out_dim=$OD ──"

        python3 scripts/benchmark_training_data.py "$GRAPHS_DIR" \
            --name synthetic \
            --output-csv "$OUTPUT_CSV" \
            --feat-dim "$FD" \
            --out-dim "$OD" \
            --skip-metrics \
            2>&1 | grep -E "(Quick|best=|ERROR|OOM|Resuming|saved)" || true

        echo
    done
done

echo "============================================================"
echo " GRID SEARCH COMPLETE"
echo " Results: $OUTPUT_CSV"
echo "============================================================"
