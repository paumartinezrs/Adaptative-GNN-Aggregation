"""
Quick benchmark – measure aggregation-strategy timing with random features.

This module is used to generate **training data for the classifier**.
Each graph is loaded once, random features are allocated, and one training
step is timed for each of the three aggregation strategies.  The output CSV
contains both graph metrics and timing columns.

For the *full* benchmark that uses real dataset features and multiple epochs,
see ``scripts/benchmark_partitions.py``.
"""

import csv
import gc
import os
import random
import time
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree

from .config import NODE_FEAT_DIM, HIDDEN_DIM, BENCHMARK_REPEATS, SEED, OUT_DIM_RANGE
from .model import AdaptiveGCN


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_graph_to_tensors(path, device, feat_dim=NODE_FEAT_DIM,
                          load_dense=True, out_dim=10):
    """
    Load an ``.edgelist`` and return tensors for benchmarking.

    Features and labels are **random** (the point is to time aggregation,
    not to learn).  The graph is made undirected with self-loops so that
    it matches a real GNN sparse adjacency.

    Returns
    -------
    edge_index, adj_sparse, adj_dense, x, y, deg_inv_sqrt, num_nodes
    """
    G = nx.read_edgelist(str(path), nodetype=int)
    if G.number_of_nodes() == 0:
        raise ValueError(f"Empty graph: {path}")

    nodelist = list(G.nodes())
    mapping = {u: i for i, u in enumerate(nodelist)}
    G = nx.relabel_nodes(G, mapping)

    num_nodes = G.number_of_nodes()
    edges = np.array(list(G.edges()), dtype=np.int64)
    if edges.size == 0:
        raise ValueError(f"No edges: {path}")

    # Undirected + self-loops
    rev = edges[:, [1, 0]]
    self_loops = np.arange(num_nodes, dtype=np.int64).reshape(-1, 1)
    self_loops = np.hstack([self_loops, self_loops])
    all_edges = np.vstack([edges, rev, self_loops])
    edge_index = torch.from_numpy(all_edges.T).long().to(device)

    # Degree normalisation
    deg = degree(edge_index[1], num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    deg_inv_sqrt = deg_inv_sqrt.view(-1, 1).to(device)

    row, col = edge_index
    vals = torch.ones(row.size(0), device=device)
    adj_sparse = (
        torch.sparse_coo_tensor(torch.stack([row, col]), vals, (num_nodes, num_nodes))
        .coalesce()
        .to_sparse_csr()
    )

    adj_dense = None
    if load_dense:
        adj_dense = torch.zeros(num_nodes, num_nodes, dtype=torch.float32, device=device)
        adj_dense[row, col] = 1.0

    x = torch.randn(num_nodes, feat_dim, dtype=torch.float32, device=device)
    y = torch.randint(0, out_dim, (num_nodes,), device=device)

    return edge_index, adj_sparse, adj_dense, x, y, deg_inv_sqrt, num_nodes


def _time_strategy(fn, device, n_repeats=BENCHMARK_REPEATS):
    """Return the median wall-clock time (seconds) of calling *fn*."""
    times = []
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(n_repeats):
            torch.cuda.synchronize()
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1000.0)
    else:
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


# ──────────────────────────────────────────────────────────────────────────────
# Main entry-point
# ──────────────────────────────────────────────────────────────────────────────

def run_quick_benchmark(df_metrics, output_base_dir, benchmark_csv_path,
                        feature_dim=NODE_FEAT_DIM, out_dim=None):
    """
    Quick benchmark with random features for every graph listed in *df_metrics*.

    Each row in *df_metrics* must have ``strategy_dir``, ``partition_name``,
    and at least the metric columns produced by :func:`compute_partition_metrics`.

    Parameters
    ----------
    df_metrics : DataFrame
        Metrics table (one row per graph/partition).
    output_base_dir : str | Path
        Root directory where edgelist files live.
    benchmark_csv_path : str | Path
        Output CSV (supports incremental resume).
    feature_dim : int
        Feature dimension for random input.
    out_dim : int or None
        Number of output classes.  *None* → pick randomly from
        ``OUT_DIM_RANGE``.

    Returns
    -------
    DataFrame with metrics + timing + ``best_strategy`` columns.
    """
    if out_dim is None:
        out_dim = random.choice(OUT_DIM_RANGE)

    output_base_dir = Path(output_base_dir)
    benchmark_csv_path = Path(benchmark_csv_path)
    benchmark_csv_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== Quick benchmark ===")

    if df_metrics is None or len(df_metrics) == 0:
        print("  No graphs to benchmark.")
        return pd.DataFrame()

    if not torch.cuda.is_available():
        raise RuntimeError("Quick benchmark requires a CUDA GPU.")

    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda")
    print(f"  Device: {device}")
    print(f"  Feature dim: {feature_dim}, Out dim: {out_dim}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    meta_cols = ["strategy_dir", "strategy_base", "k_parts", "partition_name"]
    metric_cols = [c for c in df_metrics.columns if c not in meta_cols]
    csv_columns = meta_cols + [
        "feature_dim", "out_dim", "hidden_dim",
        "best_strategy", "step_s_edge_agg", "step_s_sparse_matmul", "step_s_dense_matmul",
    ] + metric_cols

    # ── Resume logic ──
    has_existing = benchmark_csv_path.exists() and benchmark_csv_path.stat().st_size > 0
    if has_existing:
        df_ex = pd.read_csv(benchmark_csv_path)
        all_rows = df_ex.to_dict(orient="records")
        done_keys, done_paths = set(), set()
        for _, r in df_ex.iterrows():
            rf = r.get("feature_dim")
            ro = r.get("out_dim")
            if rf != feature_dim or ro != out_dim:
                continue
            base = r.get("strategy_base")
            if base == "flat":
                pp = r.get("partition_path")
                if isinstance(pp, str) and pp.strip():
                    try:
                        done_paths.add((str(Path(pp).resolve()), feature_dim, out_dim))
                    except Exception:
                        done_paths.add((str(pp), feature_dim, out_dim))
            else:
                sd = r.get("strategy_dir")
                pn = r.get("partition_name")
                if sd is not None and pn is not None:
                    done_keys.add((sd, pn, feature_dim, out_dim))
        print(f"  Resuming – {len(done_keys) + len(done_paths)} already done")
    else:
        all_rows, done_keys, done_paths = [], set(), set()

    csv_file = benchmark_csv_path.open("a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    if not has_existing:
        writer.writeheader()
        csv_file.flush()
        os.fsync(csv_file.fileno())

    try:
        model = None
        optimizer = None
        refresh_counter = 0

        def _create_model():
            nonlocal model, optimizer
            if model is not None:
                del model, optimizer
                torch.cuda.empty_cache()
            model = AdaptiveGCN(feature_dim, HIDDEN_DIM, out_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        _create_model()

        def _train_step(x, structure, dis, y, strategy):
            model.train()
            optimizer.zero_grad()
            out = model(x.detach(), structure, dis, strategy)
            loss = F.cross_entropy(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total = len(df_metrics)
        pending = 0
        warmup_done = False

        for _, row in df_metrics.iterrows():
            sname = row["strategy_dir"]
            pname = row["partition_name"]
            key = (sname, pname, feature_dim, out_dim)

            # Skip if done
            pp_field = row.get("partition_path")
            rpp = None
            if isinstance(pp_field, str) and pp_field.strip():
                try:
                    rpp = str(Path(pp_field).resolve())
                except Exception:
                    rpp = str(pp_field)

            if row.get("strategy_base") == "flat":
                if rpp and (rpp, feature_dim, out_dim) in done_paths:
                    continue
            else:
                if key in done_keys:
                    continue
                if rpp and (rpp, feature_dim, out_dim) in done_paths:
                    continue

            pending += 1
            refresh_counter += 1
            if refresh_counter >= 20:
                _create_model()
                refresh_counter = 0
                warmup_done = False
                torch.cuda.empty_cache()

            # Resolve path
            if isinstance(pp_field, str) and pp_field.strip():
                p = Path(pp_field)
            else:
                p = output_base_dir / sname / pname

            print(f"  [{pending}/{total}] {sname}/{pname} …", end=" ")

            ei_cpu = as_cpu = x_cpu = y_cpu = d_cpu = None
            x_g = y_g = d_g = ei_g = as_g = ad_g = None

            try:
                ei_cpu, as_cpu, _, x_cpu, y_cpu, d_cpu, nn = load_graph_to_tensors(
                    p, torch.device("cpu"), feat_dim=feature_dim,
                    load_dense=False, out_dim=out_dim,
                )

                if not warmup_done:
                    for _ in range(max(3, BENCHMARK_REPEATS)):
                        x_g, y_g, d_g = x_cpu.to(device), y_cpu.to(device), d_cpu.to(device)
                        ei_g = ei_cpu.to(device)
                        _train_step(x_g, ei_g, d_g, y_g, "edge")
                        torch.cuda.synchronize(); del ei_g; ei_g = None
                        as_g = as_cpu.to(device)
                        _train_step(x_g, as_g, d_g, y_g, "sparse")
                        torch.cuda.synchronize(); del as_g; as_g = None
                        ei_t = ei_cpu.to(device)
                        ad_g = torch.zeros(nn, nn, dtype=torch.float32, device=device)
                        ad_g[ei_t[0], ei_t[1]] = 1.0; del ei_t
                        _train_step(x_g, ad_g, d_g, y_g, "dense")
                        torch.cuda.synchronize(); del ad_g; ad_g = None
                        del x_g, y_g, d_g; x_g = y_g = d_g = None
                    torch.cuda.synchronize(); torch.cuda.empty_cache()
                    warmup_done = True
                    print("(warmup) ", end="")

                x_g, y_g, d_g = x_cpu.to(device), y_cpu.to(device), d_cpu.to(device)

                ei_g = ei_cpu.to(device)
                t_edge = _time_strategy(lambda: _train_step(x_g, ei_g, d_g, y_g, "edge"), device)
                del ei_g; ei_g = None

                as_g = as_cpu.to(device)
                t_sparse = _time_strategy(lambda: _train_step(x_g, as_g, d_g, y_g, "sparse"), device)
                del as_g; as_g = None

                ei_t = ei_cpu.to(device)
                ad_g = torch.zeros(nn, nn, dtype=torch.float32, device=device)
                ad_g[ei_t[0], ei_t[1]] = 1.0; del ei_t
                t_dense = _time_strategy(lambda: _train_step(x_g, ad_g, d_g, y_g, "dense"), device)
                del ad_g; ad_g = None
                del x_g, y_g, d_g; x_g = y_g = d_g = None
                del ei_cpu, as_cpu, x_cpu, y_cpu, d_cpu
                ei_cpu = as_cpu = x_cpu = y_cpu = d_cpu = None

                times = {"edge_agg": t_edge, "sparse_matmul": t_sparse, "dense_matmul": t_dense}
                best = min(times, key=times.get)

                out_row = {
                    "strategy_dir": sname,
                    "strategy_base": row["strategy_base"],
                    "k_parts": row["k_parts"],
                    "partition_name": pname,
                    "feature_dim": feature_dim,
                    "out_dim": out_dim,
                    "hidden_dim": HIDDEN_DIM,
                    "best_strategy": best,
                    "step_s_edge_agg": t_edge,
                    "step_s_sparse_matmul": t_sparse,
                    "step_s_dense_matmul": t_dense,
                    **{k: row[k] for k in row.index if k not in meta_cols},
                }
                all_rows.append(out_row)
                done_keys.add(key)
                writer.writerow(out_row)
                csv_file.flush()
                os.fsync(csv_file.fileno())
                print(f"OK (best={best})")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("OOM – skipped")
                    gc.collect(); torch.cuda.empty_cache()
                else:
                    print(f"ERROR: {e}")
            except Exception as e:
                print(f"ERROR: {e}")
            finally:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                for v in (x_g, y_g, d_g, ei_g, as_g, ad_g,
                          ei_cpu, as_cpu, x_cpu, y_cpu, d_cpu):
                    if v is not None:
                        del v
                torch.cuda.empty_cache()

        return pd.DataFrame(all_rows)
    finally:
        csv_file.close()
