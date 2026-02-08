"""
Adaptive GNN Aggregation – shared library.

Re-exports publicly used functions and classes so that scripts can write::

    from lib import load_graph_structure, AdaptiveGCN, ...
"""

from .config import *  # noqa: F401,F403

from .datasets import (
    DATASET_REGISTRY,
    list_datasets,
    load_graph_structure,
    load_dataset_pyg,
)
from .model import AdaptiveGCN, AdaptiveGCNLayer
from .partitioning import (
    compute_k_values,
    build_graph_structures,
    partition_metis,
    partition_kahip,
    run_all_partitioning,
)
from .metrics import (
    METRIC_COLUMNS,
    compute_partition_metrics,
    compute_all_metrics,
    compute_flat_metrics,
    MetricsLoader,
)
from .loader import PartitionDataLoader
from .benchmark import run_quick_benchmark, load_graph_to_tensors
from .utils import set_seed, save_row_to_csv
