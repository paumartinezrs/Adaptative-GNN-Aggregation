"""
Configuration constants for the Adaptive GNN Aggregation pipeline.

All tunable parameters are centralised here so that scripts share
a single source of truth.
"""

# ==============================================================================
# GRAPH PARTITIONING
# ==============================================================================

MAX_PARTITION_NODES = 20_000   # Target maximum nodes per partition
MIN_K = 2                      # Minimum number of partitions
IMBALANCE = 0.03               # Allowed partition imbalance (KaHIP)
SEED = 42

# Centrality-based node weights for METIS
CENTRALITY_SAMPLES = 64
CENTRALITY_SCALE = 100

# METIS weight configurations – each entry is a list of node-attribute names
# that METIS uses as multi-constraint weights.
METIS_CONFIGS = [
    ["nodes"],
    ["edges"],
    ["nodes", "edges"],
    ["centrality"],
    ["nodes", "centrality"],
    ["edges", "centrality"],
    ["nodes", "edges", "centrality"],
]

# KaHIP partitioning modes (increasing quality / runtime)
KAHIP_MODES = ["fast", "eco", "strong"]

# ==============================================================================
# GNN MODEL
# ==============================================================================

NODE_FEAT_DIM = 128            # Default input feature dimension
HIDDEN_DIM = 128               # Hidden layer width
BENCHMARK_REPEATS = 5          # Timing repetitions (quick benchmark)

# Output-dimension search space (mirrors real dataset class counts)
OUT_DIM_RANGE = [2, 4, 8, 16, 32, 64, 128]

# ==============================================================================
# NORMALISATION DEFAULTS (full benchmark with real features)
# ==============================================================================

STANDARD_IN_DIM = 128
STANDARD_OUT_DIM = 10

# ==============================================================================
# GRAPH METRICS
# ==============================================================================

CLUSTERING_TRIALS = 100        # Trials for approximate clustering coefficient

# ==============================================================================
# FULL BENCHMARK (multi-epoch, real features – benchmark_partitions.py)
# ==============================================================================

DEFAULT_EPOCHS = 10
DEFAULT_WARMUP_EPOCHS = 3
DEFAULT_LR = 0.005
DEFAULT_DROPOUT = 0.5
