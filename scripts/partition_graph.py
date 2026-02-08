#!/usr/bin/env python3
"""
Step 1 – Partition a real-world graph and compute sub-graph metrics.

Takes a dataset name, partitions it with every METIS config and KaHIP mode
for a range of K values, then computes topology metrics for each partition.

Usage
-----
    python scripts/partition_graph.py cora
    python scripts/partition_graph.py ogbn-arxiv
    python scripts/partition_graph.py computers --data-dir ./data

Outputs
-------
    outputs/partitions/<dataset>/          ← edgelists + mappings
    outputs/partitions/<dataset>/timing.csv ← partitioning times
    outputs/metrics/<dataset>_partitions_metrics.csv
"""

import sys
import gc
import argparse
from datetime import datetime
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import (
    load_graph_structure,
    build_graph_structures,
    compute_k_values,
    run_all_partitioning,
    compute_all_metrics,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Partition a graph dataset and compute partition metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("dataset", help="Dataset name (e.g. cora, ogbn-arxiv) or path to .edgelist")
    p.add_argument("--data-dir", default="./data", help="Root directory for PyG datasets (default: ./data)")
    p.add_argument("--output-dir", default="./outputs", help="Root output directory (default: ./outputs)")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset

    print("=" * 60)
    print(f"STEP 1: PARTITION GRAPH – {dataset_name}")
    print("=" * 60)

    # 1. Load graph topology
    print(f"\n[{datetime.now().time()}] Loading graph …")
    edge_index_np, num_nodes = load_graph_structure(dataset_name, data_dir=args.data_dir)
    num_edges = len(edge_index_np[0])
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Directed edges: {num_edges:,}")

    if num_nodes == 0 or num_edges == 0:
        print("ERROR: empty dataset"); sys.exit(1)

    # 2. K values
    k_values = compute_k_values(num_nodes)
    print(f"\nK values: {k_values}")
    for k in k_values:
        print(f"  K={k}: ~{num_nodes // k:,} nodes/partition")

    # 3. Output paths
    safe_name = dataset_name.lower().replace("-", "_").replace("/", "_")
    part_dir = Path(args.output_dir) / "partitions" / safe_name
    part_dir.mkdir(parents=True, exist_ok=True)
    timing_csv = part_dir / "timing.csv"
    metrics_csv = Path(args.output_dir) / "metrics" / f"{safe_name}_partitions_metrics.csv"

    # 4. Build structures
    G, xadj, adjncy = build_graph_structures(edge_index_np, num_nodes)
    del edge_index_np; gc.collect()

    # 5. Partition
    print(f"\n[{datetime.now().time()}] Partitioning …")
    run_all_partitioning(G, xadj, adjncy, num_nodes, k_values,
                         part_dir, timing_csv, dataset_name=safe_name)
    del G, xadj, adjncy; gc.collect()

    # 6. Metrics
    print(f"\n[{datetime.now().time()}] Computing metrics …")
    df = compute_all_metrics(part_dir, metrics_csv)
    print(f"  Partitions with metrics: {len(df)}")

    # Done
    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print("=" * 60)
    print(f"  Partitions : {part_dir}/")
    print(f"  Timing     : {timing_csv}")
    print(f"  Metrics    : {metrics_csv}")


if __name__ == "__main__":
    main()
