"""
Graph partitioning with METIS and KaHIP.

Provides helpers to:

1. Decide how many partitions (*K* values) are needed.
2. Build the data-structures expected by each partitioner.
3. Run METIS (multiple weight configurations) and KaHIP (multiple modes).
4. Save each sub-graph as an ``.edgelist`` file with a local→global
   ``.mapping`` so that the rest of the pipeline can work with small,
   0-indexed sub-graphs.
"""

import time
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd

from .config import (
    MAX_PARTITION_NODES,
    MIN_K,
    IMBALANCE,
    SEED,
    METIS_CONFIGS,
    KAHIP_MODES,
)


# ──────────────────────────────────────────────────────────────────────────────
# K-value computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_k_values(num_nodes, max_partition_size=None):
    """
    Return the list of *K* values (partition counts) to try.

    Strategy:
    * The minimum K is ``ceil(num_nodes / max_partition_size)``.
    * Powers of two starting from that minimum are generated.
    * If the exact minimum is not a power of two it is prepended.
    * Stops when partitions would have fewer than 100 nodes.
    """
    if max_partition_size is None:
        max_partition_size = MAX_PARTITION_NODES

    if num_nodes <= max_partition_size:
        return [2, 4]

    min_k_needed = max(MIN_K, int(np.ceil(num_nodes / max_partition_size)))

    k = MIN_K
    while k < min_k_needed:
        k *= 2

    k_values = []
    for _ in range(5):
        k_values.append(k)
        k *= 2
        if num_nodes / k < 100:
            break

    if min_k_needed not in k_values:
        k_values.insert(0, min_k_needed)

    return sorted(set(k_values))


# ──────────────────────────────────────────────────────────────────────────────
# Graph structure builders
# ──────────────────────────────────────────────────────────────────────────────

def build_graph_structures(edge_index_np, num_nodes):
    """
    Build the data-structures required by METIS (NetworkX) and KaHIP (CSR).

    Parameters
    ----------
    edge_index_np : ndarray (2, E)
    num_nodes : int

    Returns
    -------
    G : nx.Graph
    xadj : ndarray (N+1,)   – CSR row pointers
    adjncy : ndarray (2·E,) – CSR column indices
    """
    print(f"[{datetime.now().time()}] Building graph structures …")

    row = edge_index_np[0].astype(np.int32, copy=False)
    col = edge_index_np[1].astype(np.int32, copy=False)

    assert row.min() >= 0 and row.max() < num_nodes
    assert col.min() >= 0 and col.max() < num_nodes

    # Remove self-loops
    mask = row != col
    row, col = row[mask], col[mask]

    # Deduplicate (keep u < v)
    lo = np.minimum(row, col)
    hi = np.maximum(row, col)
    edges_unique = np.unique(np.stack([lo, hi], axis=1), axis=0)
    row, col = edges_unique[:, 0], edges_unique[:, 1]

    print(f"  Unique undirected edges (no self-loops): {len(row):,}")
    if len(row) == 0:
        raise ValueError("Graph has no valid edges after filtering.")

    # ── NetworkX / METIS ──
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges_unique)
    degree_counts = np.bincount(np.concatenate([row, col]), minlength=num_nodes)
    G.graph["_degree_counts"] = degree_counts

    print(f"  NetworkX: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    n_comp = nx.number_connected_components(G)
    if n_comp > 1:
        print(f"  Warning: graph has {n_comp} connected components")

    # ── KaHIP CSR ──
    full_row = np.concatenate([row, col]).astype(np.int32)
    full_col = np.concatenate([col, row]).astype(np.int32)

    idx = np.argsort(full_row)
    full_row = full_row[idx]
    adjncy = full_col[idx]

    degrees = np.bincount(full_row, minlength=num_nodes).astype(np.int32)
    xadj = np.zeros(num_nodes + 1, dtype=np.int32)
    xadj[1:] = np.cumsum(degrees)

    assert xadj[-1] == len(adjncy)
    print(f"  CSR: {len(adjncy):,} directed entries")

    return G, xadj, adjncy


# ──────────────────────────────────────────────────────────────────────────────
# METIS partitioning
# ──────────────────────────────────────────────────────────────────────────────

def partition_metis(G, k, config, output_dir):
    """
    Partition *G* into *k* parts with METIS using the given weight *config*.

    Returns (time_seconds, edge_cut, parts_array).
    """
    import metis

    num_nodes = G.number_of_nodes()
    if num_nodes < k:
        raise ValueError(f"Cannot partition {num_nodes} nodes into {k} parts")

    # Degree-based node weights (cached on graph)
    degree_weights = G.graph.get("_degree_weight_dict")
    if degree_weights is None:
        dc = G.graph.get("_degree_counts")
        if dc is None:
            dc = np.array([d for _, d in G.degree()], dtype=np.int32)
        mx = int(dc.max()) if dc.size else 0
        if mx == 0:
            norm = np.ones_like(dc, dtype=np.int32)
        else:
            norm = np.maximum(1, np.ceil(dc * 100 / mx)).astype(np.int32)
        degree_weights = {int(i): int(norm[i]) for i in range(len(norm))}
        G.graph["_degree_weight_dict"] = degree_weights

    nx.set_node_attributes(G, 1, "nodes")
    if "edges" in config:
        nx.set_node_attributes(G, degree_weights, "edges")
    else:
        nx.set_node_attributes(G, 1, "edges")

    G.graph["node_weight_attr"] = config
    nx.set_edge_attributes(G, 1, "weight")

    t0 = time.perf_counter()
    edgecut, parts = metis.part_graph(G, k)
    dt = time.perf_counter() - t0

    parts = np.asarray(parts, dtype=np.int32)
    assert len(parts) == num_nodes
    assert parts.min() >= 0 and parts.max() < k

    _save_partitions_nx(G, parts, k, output_dir)
    return dt, edgecut, parts


# ──────────────────────────────────────────────────────────────────────────────
# KaHIP partitioning
# ──────────────────────────────────────────────────────────────────────────────

def partition_kahip(xadj, adjncy, k, mode_name, output_dir, num_nodes):
    """
    Partition using KaHIP in the given *mode_name* (``fast``, ``eco``, ``strong``).

    Returns (time_seconds, edge_cut, parts_array).
    """
    import kahip

    if num_nodes < k:
        raise ValueError(f"Cannot partition {num_nodes} nodes into {k} parts")

    mode_map = {"fast": kahip.FAST, "eco": kahip.ECO, "strong": kahip.STRONG}
    mode_val = mode_map[mode_name]

    vwgt = np.ones(num_nodes, dtype=np.int32)
    adjcwgt = np.ones(len(adjncy), dtype=np.int32)

    t0 = time.perf_counter()
    edgecut, parts = kahip.kaffpa(
        vwgt, xadj, adjcwgt, adjncy,
        int(k), float(IMBALANCE), True, int(SEED), int(mode_val),
    )
    dt = time.perf_counter() - t0

    parts = np.asarray(parts, dtype=np.int32)
    assert len(parts) == num_nodes
    assert parts.min() >= 0 and parts.max() < k

    _save_partitions_csr(xadj, adjncy, parts, k, output_dir)
    return dt, edgecut, parts


# ──────────────────────────────────────────────────────────────────────────────
# Save helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_partitions_nx(G, parts, k, output_dir):
    """Write per-partition ``.edgelist`` and ``.mapping`` from a NetworkX graph."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_maps = [{} for _ in range(k)]
    counters = [0] * k
    files = [open(output_dir / f"part_{p}.edgelist", "w") for p in range(k)]

    try:
        for u, v in G.edges():
            pu, pv = int(parts[u]), int(parts[v])
            if pu != pv:
                continue
            p = pu
            for n in (u, v):
                if n not in local_maps[p]:
                    local_maps[p][n] = counters[p]
                    counters[p] += 1
            files[p].write(f"{local_maps[p][u]} {local_maps[p][v]}\n")
    finally:
        for f in files:
            f.close()

    _write_mappings(local_maps, k, output_dir)


def _save_partitions_csr(xadj, adjncy, parts, k, output_dir):
    """Write per-partition ``.edgelist`` and ``.mapping`` from a CSR graph."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_nodes = len(xadj) - 1
    local_maps = [{} for _ in range(k)]
    counters = [0] * k
    files = [open(output_dir / f"part_{p}.edgelist", "w") for p in range(k)]

    try:
        for u in range(num_nodes):
            pu = int(parts[u])
            for v in adjncy[int(xadj[u]):int(xadj[u + 1])]:
                v = int(v)
                if v <= u or parts[v] != pu:
                    continue
                for n in (u, v):
                    if n not in local_maps[pu]:
                        local_maps[pu][n] = counters[pu]
                        counters[pu] += 1
                files[pu].write(f"{local_maps[pu][u]} {local_maps[pu][v]}\n")
    finally:
        for f in files:
            f.close()

    _write_mappings(local_maps, k, output_dir)


def _write_mappings(local_maps, k, output_dir):
    """Write ``part_<p>.mapping`` (one global-id per line, ordered by local-id)."""
    for p in range(k):
        if not local_maps[p]:
            continue
        inv = {lid: gid for gid, lid in local_maps[p].items()}
        with open(output_dir / f"part_{p}.mapping", "w") as f:
            for lid in range(len(inv)):
                f.write(f"{inv[lid]}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Full partitioning sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_all_partitioning(
    G, xadj, adjncy, num_nodes, k_values, output_base_dir, csv_path,
    dataset_name=None,
):
    """
    Run every METIS config × KaHIP mode × K value and write results to CSV.

    This is the top-level convenience wrapper called by the partition script.
    """
    output_base_dir = Path(output_base_dir)
    csv_path = Path(csv_path)

    fieldnames = ["method", "config", "k", "partitioning_time_s", "edge_cut", "ts"]
    if dataset_name:
        fieldnames.insert(0, "dataset")

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    results = []
    total = (len(METIS_CONFIGS) + len(KAHIP_MODES)) * len(k_values)
    idx = 0

    try:
        # ── METIS ──
        print("\n=== METIS ===")
        for config in METIS_CONFIGS:
            cfg_name = "+".join(config)
            print(f"\n  Config: {cfg_name}")
            for k in k_values:
                idx += 1
                out_dir = output_base_dir / f"metis_{cfg_name}_k{k}"
                print(f"    [{idx}/{total}] K={k} …", end=" ")
                try:
                    dt, ec, _ = partition_metis(G, k, config, out_dir)
                    print(f"OK ({dt:.2f}s, edgecut={ec})")
                    row = {"method": "metis", "config": cfg_name, "k": k,
                           "partitioning_time_s": dt, "edge_cut": ec,
                           "ts": datetime.now().isoformat()}
                    if dataset_name:
                        row["dataset"] = dataset_name
                    results.append(row)
                    writer.writerow(row)
                    csv_file.flush()
                except Exception as e:
                    print(f"ERROR: {e}")

        # ── KaHIP ──
        print("\n=== KaHIP ===")
        for mode in KAHIP_MODES:
            print(f"\n  Mode: {mode}")
            for k in k_values:
                idx += 1
                out_dir = output_base_dir / f"kahip_{mode}_k{k}"
                print(f"    [{idx}/{total}] K={k} …", end=" ")
                try:
                    dt, ec, _ = partition_kahip(xadj, adjncy, k, mode, out_dir, num_nodes)
                    print(f"OK ({dt:.2f}s, edgecut={ec})")
                    row = {"method": "kahip", "config": mode, "k": k,
                           "partitioning_time_s": dt, "edge_cut": ec,
                           "ts": datetime.now().isoformat()}
                    if dataset_name:
                        row["dataset"] = dataset_name
                    results.append(row)
                    writer.writerow(row)
                    csv_file.flush()
                except Exception as e:
                    print(f"ERROR: {e}")
    finally:
        csv_file.close()

    return results
