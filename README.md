# Adaptive GNN Aggregation

**Adaptive selection of GNN aggregation strategies for graph partitions.**

This project benchmarks three aggregation strategies for Graph Convolutional
Networks — *edge-index scatter*, *sparse matrix multiplication*, and *dense
matrix multiplication* — and trains a lightweight Random Forest classifier to
predict the fastest strategy per partition at inference time, yielding speedups over single fixed strategies.

---

## Workflow Overview

```
           ┌─────────────────────────────┐
           │  Step 0 (external)          │
           │  Generate synthetic         │
           │  graphs (e.g. GraphGalaxy)  │
           └──────────────┬──────────────┘
                          │ edgelist files
                          ▼
┌──────────────────────┐   ┌───────────────────────────┐
│  Step 1              │   │  Step 2                   │
│  partition_graph.py  │   │  benchmark_training_      │
│  ──────────────────  │   │  data.py                  │
│  Partition a real    │   │  ───────────────────────  │
│  graph with METIS &  │   │  Quick-benchmark each     │
│  KaHIP               │   │  synthetic graph →        │
│  → edgelists +       │   │  CSV with metrics +       │
│    metrics CSV       │   │  best strategy (training  │
└──────────┬───────────┘   │  data for classifier)     │
           │               └─────────────┬─────────────┘
           │                             │
           ▼                             │
┌──────────────────────┐                 │
│  Step 3              │                 │
│  benchmark_          │                 │
│  partitions.py       │                 │
│  ──────────────────  │                 │
│  Multi-epoch GNN     │                 │
│  benchmark using     │                 │
│  real features       │                 │
│  → results CSV       │                 │
└──────────┬───────────┘                 │
           │                             │
           ▼                             ▼
┌──────────────────────────────────────────────────────┐
│  Step 4: notebooks/analyze.ipynb                     │
│  ──────────────────────────────────────────────────  │
│  • Train Random Forest on synthetic benchmark data   │
│  • Predict best strategy per real partition          │
│  • Compute speedups, confusion matrix, feature       │
│    importance, heterogeneity analysis                │
└──────────────────────────────────────────────────────┘
```

---

## Installation

```bash
# 1. Create environment
conda create -n adaptive-gnn python=3.10 -y
conda activate adaptive-gnn

# 2. Install PyTorch + PyG (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Install METIS + KaHIP (system packages or from source)
#    METIS:  sudo apt install libmetis-dev && pip install metis
#    KaHIP:  https://github.com/KaHIP/KaHIP
```

---

## Usage

### Step 1 — Partition a real graph

```bash
python scripts/partition_graph.py \
    --dataset cora \
    --output-dir outputs/partitions/cora \
    --data-dir ./data
```

Produces:
- `outputs/partitions/cora/<method>_<config>/part_*.edgelist` — edgelist per partition
- `outputs/partitions/cora/<method>_<config>/mapping.csv` — global → local node mapping
- `outputs/metrics/cora_partitions_metrics.csv` — structural metrics per partition

Supported datasets: `cora`, `citeseer`, `pubmed`, `flickr`, `physics`,
`computers`, `ogbn-arxiv`, `reddit`, `elliptic`, `corafull`, `actor`,
`twitch-<region>`.

### Step 2 — Benchmark synthetic graphs (training data)

```bash
# Single run
python scripts/benchmark_training_data.py \
    --graph-dir /path/to/synthetic/edgelists \
    --output-csv outputs/synthetic_benchmark.csv \
    --feat-dim 128 --out-dim 10

# Grid search over feature/output dimensions
bash scripts/run_grid_search.sh /path/to/synthetic/edgelists outputs/synthetic_benchmark.csv
```

### Step 3 — Benchmark real-graph partitions

```bash
python scripts/benchmark_partitions.py \
    --dataset cora \
    --partition-dir outputs/partitions/cora \
    --output-csv outputs/real_benchmark.csv \
    --epochs 10 --warmup-epochs 3
```

### Step 4 — Analysis notebook

```bash
jupyter notebook notebooks/analyze.ipynb
```

Edit the `CONFIGURATION` cell at the top to point to your output CSVs, then
run all cells.

---

## Project Structure

```
adaptive-gnn-aggregation/
├── README.md
├── requirements.txt
├── lib/
│   ├── __init__.py          # Public API re-exports
│   ├── config.py            # All constants & hyperparameters
│   ├── utils.py             # Seed, CSV helpers
│   ├── model.py             # AdaptiveGCN (edge / sparse / dense)
│   ├── datasets.py          # Unified dataset loading (structure + PyG)
│   ├── partitioning.py      # METIS & KaHIP partitioning
│   ├── metrics.py           # Graph metrics computation & loading
│   ├── loader.py            # PartitionDataLoader (maps partitions to global features)
│   └── benchmark.py         # Quick single-step timing benchmark
├── scripts/
│   ├── partition_graph.py           # Step 1 CLI
│   ├── benchmark_training_data.py   # Step 2 CLI
│   ├── benchmark_partitions.py      # Step 3 CLI
│   └── run_grid_search.sh           # Grid search wrapper for Step 2
└── notebooks/
    └── analyze.ipynb                # Step 4: analysis & figures
```

---

## Model

**AdaptiveGCN** is a two-layer GCN that accepts a `strategy` parameter
controlling how neighbour aggregation is computed:

| Strategy | Implementation | Best when |
|----------|---------------|-----------|
| `edge`   | `index_add_` on edge index | Very sparse graphs |
| `sparse` | `torch.sparse.mm` with CSR/COO matrix | Medium density |
| `dense`  | `torch.matmul` with dense adjacency | Small, dense graphs |

All three produce the same mathematical result
($\hat{A} = D^{-1/2} A D^{-1/2}$); only the compute kernel differs.

---

## Output Format

### Synthetic benchmark CSV (`synthetic_benchmark.csv`)

| Column | Description |
|--------|-------------|
| `graph` | Graph filename |
| `nodes`, `edges` | Graph size |
| `density`, `clustering`, `max_degree`, … | Structural metrics |
| `feature_dim`, `out_dim` | Model configuration |
| `step_s_edge_agg` | Median time (s) for edge strategy |
| `step_s_sparse_matmul` | Median time (s) for sparse strategy |
| `step_s_dense_matmul` | Median time (s) for dense strategy |
| `best_strategy` | Fastest strategy label |

### Real benchmark CSV (`real_benchmark.csv`)

| Column | Description |
|--------|-------------|
| `dataset` | Source dataset |
| `partition_config` | Partitioning method + parameters |
| `partition_name` | Individual partition ID |
| `epoch` | Training epoch number |
| `edge_median_time_s` | Per-partition time with edge strategy |
| `sparse_median_time_s` | Per-partition time with sparse strategy |
| `dense_median_time_s` | Per-partition time with dense strategy |

---

## License

This code accompanies an academic thesis. Contact the author for licensing.
