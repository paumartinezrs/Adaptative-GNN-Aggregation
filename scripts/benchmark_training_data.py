#!/usr/bin/env python3
"""
Step 2 – Quick-benchmark standalone graphs to build classifier training data.

Reads a directory of edgelist files (e.g. synthetic graphs), computes their
metrics, and runs a fast single-step benchmark for each aggregation strategy.

The resulting CSV has one row per graph with both metrics and timing, and is
used as training data for the strategy-selection classifier.

Usage
-----
    # Single (feat_dim, out_dim) combo:
    python scripts/benchmark_training_data.py /path/to/graphs \\
        --name synthetic --feat-dim 128 --out-dim 10

    # Grid search (recommended) – use the helper script:
    bash scripts/run_grid_search.sh /path/to/graphs

Outputs
-------
    outputs/synthetic_benchmark.csv
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import compute_flat_metrics, run_quick_benchmark
from lib.config import NODE_FEAT_DIM


def parse_args():
    p = argparse.ArgumentParser(
        description="Quick-benchmark a directory of edgelist files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("graphs_dir", help="Directory containing edgelist files")
    p.add_argument("--name", default="synthetic", help="Dataset label for filenames (default: synthetic)")
    p.add_argument("--output-dir", default="./outputs", help="Root output directory (default: ./outputs)")
    p.add_argument("--output-csv", default=None, help="Override benchmark CSV path")
    p.add_argument("--max-nodes", type=int, default=10000, help="Skip graphs with more nodes (default: 10000)")
    p.add_argument("--max-edges", type=int, default=50000, help="Skip graphs with more edges (default: 50000)")
    p.add_argument("--feat-dim", type=int, default=NODE_FEAT_DIM, help=f"Feature dimension (default: {NODE_FEAT_DIM})")
    p.add_argument("--out-dim", type=int, default=None, help="Output classes (default: random)")
    p.add_argument("--skip-metrics", action="store_true", help="Skip metrics computation, reuse existing CSV")
    return p.parse_args()


def main():
    args = parse_args()

    safe = args.name.lower().replace("-", "_")
    metrics_csv = Path(args.output_dir) / "metrics" / f"{safe}_partitions_metrics.csv"
    benchmark_csv = Path(args.output_csv) if args.output_csv else (
        Path(args.output_dir) / f"{safe}_benchmark.csv"
    )

    print("=" * 60)
    print(f"STEP 2: BENCHMARK TRAINING DATA – {args.name}")
    print("=" * 60)
    print(f"  Graphs dir : {args.graphs_dir}")
    print(f"  Metrics CSV: {metrics_csv}")
    print(f"  Output CSV : {benchmark_csv}")
    print(f"  feat_dim={args.feat_dim}  out_dim={args.out_dim or 'random'}")
    print(f"  max_nodes={args.max_nodes}  max_edges={args.max_edges}")

    # 1. Metrics
    if args.skip_metrics:
        if not metrics_csv.exists():
            print("ERROR: --skip-metrics requires an existing metrics CSV"); sys.exit(1)
        df_metrics = pd.read_csv(metrics_csv)
    else:
        print("\n=== Computing metrics ===")
        df_metrics = compute_flat_metrics(args.graphs_dir, metrics_csv, max_nodes=args.max_nodes)

    if len(df_metrics) == 0:
        print("No graphs to benchmark."); return

    # 2. Apply node/edge filters
    if "nodes" in df_metrics.columns and args.max_nodes:
        df_metrics = df_metrics[df_metrics["nodes"] <= args.max_nodes]
    if "edges" in df_metrics.columns and args.max_edges:
        df_metrics = df_metrics[df_metrics["edges"] <= args.max_edges]

    print(f"\nGraphs to benchmark: {len(df_metrics)}")

    # 3. Quick benchmark
    df_bench = run_quick_benchmark(
        df_metrics, args.graphs_dir, benchmark_csv,
        feature_dim=args.feat_dim, out_dim=args.out_dim,
    )

    if len(df_bench) > 0 and "best_strategy" in df_bench.columns:
        print("\nStrategy distribution:")
        for s, c in df_bench["best_strategy"].value_counts().items():
            print(f"  {s}: {c} ({100*c/len(df_bench):.1f}%)")

    print(f"\nResults saved to: {benchmark_csv}")


if __name__ == "__main__":
    main()
