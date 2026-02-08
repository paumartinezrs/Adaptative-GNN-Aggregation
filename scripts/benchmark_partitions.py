#!/usr/bin/env python3
"""
Step 3 – Full multi-epoch benchmark of real-graph partitions.

For every dataset/partition-config combination, loads the full graph with
real features, trains an AdaptiveGCN over multiple epochs for each
aggregation strategy, and records per-partition timing statistics.

Usage
-----
    python scripts/benchmark_partitions.py \\
        --partitions-dir outputs/partitions \\
        --metrics-dir outputs/metrics

    # Single dataset:
    python scripts/benchmark_partitions.py \\
        --partitions-dir outputs/partitions \\
        --dataset computers

    # Normalised dimensions (fair cross-dataset comparison):
    python scripts/benchmark_partitions.py \\
        --partitions-dir outputs/partitions \\
        --normalize-dims --in-dim 128 --out-dim 10

Outputs
-------
    outputs/real_benchmark.csv
"""

import sys
import os
import gc
import csv
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import (
    load_dataset_pyg,
    DATASET_REGISTRY,
    STANDARD_IN_DIM,
    STANDARD_OUT_DIM,
    PartitionDataLoader,
    AdaptiveGCN,
    MetricsLoader,
    set_seed,
    save_row_to_csv,
)
from lib.config import DEFAULT_EPOCHS, DEFAULT_WARMUP_EPOCHS, DEFAULT_LR, DEFAULT_DROPOUT, HIDDEN_DIM

# Folder-name → DATASET_REGISTRY key
_FOLDER_MAP = {
    "flickr": "flickr", "physics": "physics", "computers": "computers",
    "ogbn_arxiv": "ogbn-arxiv", "cora": "cora", "pubmed": "pubmed",
    "citeseer": "citeseer", "actor": "actor", "corafull": "corafull",
    "elliptic": "elliptic",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-epoch benchmark of partitioned real graphs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--partitions-dir", required=True, help="Root partition directory")
    p.add_argument("--metrics-dir", default=None, help="Pre-computed metrics directory")
    p.add_argument("--data-dir", default="./data", help="Dataset root (default: ./data)")
    p.add_argument("--output", default="./outputs/real_benchmark.csv", help="Output CSV")
    p.add_argument("--dataset", nargs="+", default=None, help="Process only these datasets")
    p.add_argument("--strategies", nargs="+", default=["edge", "sparse", "dense"])
    p.add_argument("--normalize-dims", action="store_true")
    p.add_argument("--in-dim", type=int, default=STANDARD_IN_DIM)
    p.add_argument("--out-dim", type=int, default=STANDARD_OUT_DIM)
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, strategy, device):
    """Train one epoch; return per-partition times, avg loss, total nodes."""
    model.train()
    part_times, total_loss, total_nodes = {}, 0.0, 0

    for i in range(len(loader.partitions)):
        batch = loader.get_batch(i, strategy)
        if batch is None:
            continue
        x, y, mask, structure, deg_inv_sqrt = batch
        if mask.sum() == 0:
            continue

        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

        optimizer.zero_grad()
        out = model(x, structure, deg_inv_sqrt, strategy)
        loss = F.cross_entropy(out[mask], y[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        t1.record()
        torch.cuda.synchronize()
        part_times[i] = t0.elapsed_time(t1) / 1000.0
        total_loss += loss.item() * mask.sum().item()
        total_nodes += mask.sum().item()

    avg_loss = total_loss / total_nodes if total_nodes > 0 else 0.0
    return part_times, avg_loss, total_nodes


@torch.no_grad()
def _evaluate(model, data, device):
    """Global evaluation → (train_acc, val_acc, test_acc)."""
    from torch_geometric.utils import degree, to_torch_csr_tensor

    model.eval()
    adj = to_torch_csr_tensor(data.edge_index, size=(data.num_nodes, data.num_nodes)).to(device)
    deg = degree(data.edge_index[1], data.num_nodes, dtype=torch.float)
    dis = deg.pow(-0.5); dis[dis == float("inf")] = 0; dis = dis.view(-1, 1).to(device)

    out = model(data.x, adj, dis, "sparse")
    pred = out.argmax(1)

    def acc(m):
        return float((pred[m] == data.y[m]).sum() / m.sum()) if m.sum() > 0 else 0.0

    return acc(data.train_mask), acc(data.val_mask), acc(data.test_mask)


def _run_strategy(strategy, data, loader, in_dim, out_dim, cfg, device):
    """Run one strategy and return per-partition result dicts."""
    set_seed(cfg["seed"])
    model = AdaptiveGCN(in_dim, cfg["hidden_dim"], out_dim, dropout=cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Warmup
    for _ in range(cfg["warmup_epochs"]):
        _train_epoch(model, loader, opt, strategy, device)

    # Timed epochs
    n_parts = len(loader.partitions)
    all_times = {i: [] for i in range(n_parts)}
    for ep in range(cfg["epochs"]):
        pt, _, _ = _train_epoch(model, loader, opt, strategy, device)
        for pi, t in pt.items():
            all_times[pi].append(t)

    train_acc, val_acc, test_acc = _evaluate(model, data, device)

    results = []
    for pi in range(n_parts):
        times = all_times[pi]
        info = loader.partitions[pi]
        r = {
            "partition_idx": pi,
            "partition_name": info.get("partition_name", f"part_{pi}.edgelist"),
        }
        if times:
            r.update({
                f"{strategy}_median_time_s": round(np.median(times), 8),
                f"{strategy}_mean_time_s": round(np.mean(times), 8),
                f"{strategy}_std_time_s": round(np.std(times), 8),
                f"{strategy}_min_time_s": round(np.min(times), 8),
                f"{strategy}_max_time_s": round(np.max(times), 8),
                f"{strategy}_num_epochs": len(times),
            })
        else:
            r.update({
                f"{strategy}_median_time_s": None, f"{strategy}_mean_time_s": None,
                f"{strategy}_std_time_s": None, f"{strategy}_min_time_s": None,
                f"{strategy}_max_time_s": None, f"{strategy}_num_epochs": 0,
            })
        r["test_accuracy"] = round(test_acc, 4)
        r["val_accuracy"] = round(val_acc, 4)
        r["train_accuracy"] = round(train_acc, 4)
        results.append(r)

    return results


def _merge_results(all_results, strategies):
    """Merge per-strategy result lists into combined rows."""
    if not all_results:
        return []
    base = all_results[strategies[0]]
    merged = []
    for pi in range(len(base)):
        row = {"partition_idx": base[pi]["partition_idx"],
               "partition_name": base[pi]["partition_name"]}
        for s in strategies:
            for k, v in all_results[s][pi].items():
                if k.startswith(s):
                    row[k] = v
        row["test_accuracy"] = base[pi].get("test_accuracy")
        row["val_accuracy"] = base[pi].get("val_accuracy")
        row["train_accuracy"] = base[pi].get("train_accuracy")
        merged.append(row)
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────────────

def _discover_datasets(root):
    return [d.name for d in sorted(Path(root).iterdir()) if d.is_dir() and not d.name.startswith(".")]


def _discover_configs(root, dataset):
    dp = Path(root) / dataset
    if not dp.exists():
        return []
    return [d.name for d in sorted(dp.iterdir())
            if d.is_dir() and (d.name.startswith("kahip") or d.name.startswith("metis"))]


def _processed_configs(output_file):
    done = set()
    if not os.path.exists(output_file):
        return done
    try:
        with open(output_file) as f:
            for row in csv.DictReader(f):
                if "dataset" in row and "partition_config" in row:
                    done.add((row["dataset"], row["partition_config"]))
    except Exception:
        pass
    return done


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required."); sys.exit(1)
    device = torch.device("cuda")

    print("=" * 70)
    print("STEP 3: MULTI-EPOCH BENCHMARK")
    print("=" * 70)
    print(f"  Partitions : {args.partitions_dir}")
    print(f"  Metrics    : {args.metrics_dir}")
    print(f"  Output     : {args.output}")
    if args.normalize_dims:
        print(f"  Normalise  : in_dim={args.in_dim}, out_dim={args.out_dim}")

    cfg = {
        "hidden_dim": args.hidden_dim, "dropout": args.dropout,
        "lr": args.lr, "weight_decay": 0,
        "epochs": args.epochs, "warmup_epochs": args.warmup_epochs,
        "seed": args.seed,
    }

    metrics_loader = MetricsLoader(args.metrics_dir) if args.metrics_dir else None
    done = _processed_configs(args.output)
    datasets = args.dataset if args.dataset else _discover_datasets(args.partitions_dir)

    print(f"\n  Datasets: {datasets}")

    for folder in datasets:
        print(f"\n{'='*60}\n  DATASET: {folder}\n{'='*60}")

        ds_key = _FOLDER_MAP.get(folder)
        if ds_key is None:
            print(f"  '{folder}' not in folder map – skipping"); continue

        try:
            gc.collect(); torch.cuda.empty_cache()
            data, in_dim, out_dim = load_dataset_pyg(
                ds_key, args.data_dir, device,
                normalize_dims=args.normalize_dims,
                target_in_dim=args.in_dim, target_out_dim=args.out_dim,
            )
        except Exception as e:
            print(f"  Error loading dataset: {e}"); continue

        configs = _discover_configs(args.partitions_dir, folder)
        print(f"  Partition configs: {len(configs)}")

        for cfg_name in configs:
            if (folder, cfg_name) in done:
                print(f"  Skip {cfg_name} (already in CSV)"); continue

            print(f"\n  Config: {cfg_name}")
            part_path = Path(args.partitions_dir) / folder / cfg_name

            try:
                loader = PartitionDataLoader(
                    str(part_path), data.x, data.y, data.train_mask,
                    device, data.num_nodes,
                )
            except Exception as e:
                print(f"    Error: {e}"); continue

            if not loader.partitions:
                print("    No valid partitions"); continue

            all_res = {}
            for strat in args.strategies:
                print(f"    Strategy: {strat} …", end=" ", flush=True)
                try:
                    res = _run_strategy(strat, data, loader, in_dim, out_dim, cfg, device)
                    all_res[strat] = res
                    vt = [r[f"{strat}_median_time_s"] for r in res if r.get(f"{strat}_median_time_s") is not None]
                    print(f"OK (avg {np.mean(vt)*1000:.2f} ms)" if vt else "no times")
                except Exception as e:
                    print(f"ERROR: {e}"); continue

            merged = _merge_results(all_res, args.strategies)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for row in merged:
                if metrics_loader:
                    m = metrics_loader.get(folder, cfg_name, row["partition_name"])
                    if m:
                        for k, v in m.items():
                            row[f"metric_{k}"] = v
                row.update({
                    "timestamp": ts, "dataset": folder,
                    "partition_config": cfg_name,
                    "k_partitions": len(loader.partitions),
                    "hidden_dim": cfg["hidden_dim"], "epochs": cfg["epochs"],
                    "device": str(device), "normalized_dims": args.normalize_dims,
                    "model_in_dim": in_dim, "model_out_dim": out_dim,
                    "feature_dim": in_dim,
                })
                save_row_to_csv(args.output, row)

            print(f"    Saved {len(merged)} rows")
            del loader; gc.collect(); torch.cuda.empty_cache()

        del data; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE")
    print(f"  Results: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
