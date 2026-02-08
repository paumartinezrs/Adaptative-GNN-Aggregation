"""
Graph-level metrics computation and loading.

Metrics are computed per sub-graph (partition or standalone graph) and stored
in CSV files.  They are later used as features for the aggregation-strategy
classifier.
"""

import re
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd

from .config import CLUSTERING_TRIALS

try:
    import powerlaw
    _HAS_POWERLAW = True
except ImportError:
    _HAS_POWERLAW = False


# ──────────────────────────────────────────────────────────────────────────────
# Single-graph metrics
# ──────────────────────────────────────────────────────────────────────────────

# Columns produced by compute_partition_metrics()
METRIC_COLUMNS = [
    "nodes", "edges", "density", "density_log", "clustering",
    "max_degree", "mean_degree", "mixing", "diameter",
    "degree_pl_slope", "degree_slope_inverse",
]


def compute_partition_metrics(path) -> dict:
    """
    Compute topology metrics for a single graph stored as an ``.edgelist``.

    Returns a dict whose keys match :data:`METRIC_COLUMNS`.
    """
    from networkx.algorithms import approximation as nx_approx

    G = nx.read_edgelist(str(path), nodetype=int)
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        raise ValueError(f"Empty graph: {path}")

    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)
    density_log = np.log10(density) if density > 0 else -12.0

    degrees = [d for _, d in G.degree()]
    max_degree = float(max(degrees))
    mean_degree = 2.0 * m / n

    clustering = float(nx_approx.average_clustering(G, trials=CLUSTERING_TRIALS))

    try:
        mixing = float(nx.degree_assortativity_coefficient(G))
    except Exception:
        mixing = float("nan")

    try:
        gcc = max(nx.connected_components(G), key=len)
        diameter = float(nx_approx.diameter(G.subgraph(gcc)))
    except Exception:
        diameter = float("nan")

    # Power-law slope
    deg_seq = sorted(degrees)
    if _HAS_POWERLAW and len(deg_seq) >= 4:
        try:
            fit = powerlaw.Fit(deg_seq, verbose=False)
            pl_slope = float(fit.power_law.alpha)
        except Exception:
            pl_slope = 0.0
    else:
        pl_slope = 0.0

    if pl_slope >= 1:
        slope_inv = 1.0 / pl_slope
    elif 0 < pl_slope < 1:
        slope_inv = 1.0
    else:
        slope_inv = 0.0

    return {
        "nodes": n,
        "edges": m,
        "density": density,
        "density_log": density_log,
        "clustering": clustering,
        "max_degree": max_degree,
        "mean_degree": mean_degree,
        "mixing": mixing,
        "diameter": diameter,
        "degree_pl_slope": pl_slope,
        "degree_slope_inverse": slope_inv,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Batch metrics – partitioned graphs (strategy_dir/part_*.edgelist)
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(output_base_dir, metrics_csv_path):
    """
    Compute metrics for every sub-graph produced by the partitioning step.

    Supports incremental resumption: already-processed partitions are skipped
    when *metrics_csv_path* exists.

    Returns the full DataFrame.
    """
    output_base_dir = Path(output_base_dir)
    metrics_csv_path = Path(metrics_csv_path)
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== Computing partition metrics ===")

    strategy_dirs = sorted(p for p in output_base_dir.iterdir() if p.is_dir())

    rows, done_keys, done_paths = _load_existing_csv(metrics_csv_path)

    for strat_dir in strategy_dirs:
        strategy_name = strat_dir.name
        m = re.search(r"_k(\d+)", strategy_name)
        k_parts = int(m.group(1)) if m else None
        strategy_base = strategy_name.rsplit("_k", 1)[0]

        for path in sorted(strat_dir.glob("part_*.edgelist")):
            key = (strategy_name, path.name)
            resolved = str(path.resolve())
            if key in done_keys or resolved in done_paths:
                continue

            print(f"  {strategy_name}/{path.name} …", end=" ")
            try:
                metrics = compute_partition_metrics(path)
                row = {
                    "strategy_dir": strategy_name,
                    "strategy_base": strategy_base,
                    "k_parts": k_parts,
                    "partition_name": path.name,
                    "partition_path": resolved,
                    **metrics,
                }
                rows.append(row)
                done_keys.add(key)
                done_paths.add(resolved)
                pd.DataFrame(rows).to_csv(metrics_csv_path, index=False)
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Batch metrics – flat directory of edgelist files
# ──────────────────────────────────────────────────────────────────────────────

def compute_flat_metrics(base_dir, metrics_csv_path, max_nodes=None):
    """
    Compute metrics for every ``.edgelist`` / ``.txt`` in a flat directory.

    Used for standalone (synthetic) graphs that are not organised by strategy.
    """
    base_dir = Path(base_dir)
    metrics_csv_path = Path(metrics_csv_path)
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in base_dir.iterdir()
        if p.is_file() and p.suffix in (".edgelist", ".txt", "")
    )
    if not files:
        print("  No edgelist files found.")
        return pd.DataFrame()

    rows, _, done_paths = _load_existing_csv(metrics_csv_path, flat=True)

    for path in files:
        resolved = str(path.resolve())
        if resolved in done_paths:
            continue

        print(f"  {path.name} …", end=" ")
        try:
            if max_nodes is not None:
                n = _quick_node_count(path, max_nodes)
                if n > max_nodes:
                    print(f"skipped (>{max_nodes} nodes)")
                    continue
            metrics = compute_partition_metrics(path)
            row = {
                "strategy_dir": path.stem,
                "strategy_base": "flat",
                "k_parts": None,
                "partition_name": path.name,
                "partition_path": resolved,
                **metrics,
            }
            rows.append(row)
            done_paths.add(resolved)
            pd.DataFrame(rows).to_csv(metrics_csv_path, index=False)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# MetricsLoader – read pre-computed CSV at analysis/benchmark time
# ──────────────────────────────────────────────────────────────────────────────

class MetricsLoader:
    """Load pre-computed metrics from CSVs and look up per-partition rows."""

    def __init__(self, metrics_dir):
        self.metrics_dir = Path(metrics_dir)
        self._cache: dict[str, pd.DataFrame] = {}

    def load(self, dataset_name: str):
        if dataset_name in self._cache:
            return self._cache[dataset_name]

        candidates = [
            self.metrics_dir / f"{dataset_name}_partitions_metrics.csv",
            self.metrics_dir / f"{dataset_name.lower()}_partitions_metrics.csv",
            self.metrics_dir / f"{dataset_name.replace('-', '_')}_partitions_metrics.csv",
        ]
        for c in candidates:
            if c.exists():
                df = pd.read_csv(c)
                self._cache[dataset_name] = df
                return df
        print(f"  Warning: no metrics file for '{dataset_name}'")
        return None

    def get(self, dataset_name, strategy_dir, partition_name):
        """Return a dict of metrics for one partition, or ``None``."""
        df = self.load(dataset_name)
        if df is None:
            return None
        mask = (df["strategy_dir"] == strategy_dir) & (df["partition_name"] == partition_name)
        if mask.sum() == 0:
            return None
        row = df[mask].iloc[0]
        out = {}
        for col in METRIC_COLUMNS:
            if col in row:
                out[col] = row[col]
        for extra in ("strategy_base", "k_parts"):
            if extra in row:
                out[extra] = row[extra]
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_existing_csv(csv_path, flat=False):
    """Resume-helper: load rows/keys from an existing CSV."""
    rows, done_keys, done_paths = [], set(), set()
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        rows = df.to_dict(orient="records")
        if flat:
            if "partition_path" in df.columns:
                for p in df["partition_path"].dropna():
                    try:
                        done_paths.add(str(Path(p).resolve()))
                    except Exception:
                        done_paths.add(str(p))
        else:
            done_keys = set(zip(df["strategy_dir"], df["partition_name"]))
            if "partition_path" in df.columns:
                for p in df["partition_path"].dropna():
                    try:
                        done_paths.add(str(Path(p).resolve()))
                    except Exception:
                        done_paths.add(str(p))
        print(f"  Resuming from {len(rows)} already-processed entries")
    return rows, done_keys, done_paths


def _quick_node_count(path, limit):
    """Fast node count for an edgelist, stopping early at *limit*."""
    nodes = set()
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                nodes.add(int(parts[0]))
                nodes.add(int(parts[1]))
            except ValueError:
                continue
            if len(nodes) > limit:
                break
    return len(nodes)
