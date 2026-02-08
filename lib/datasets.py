"""
Unified dataset loading.

Two public entry points cover every use-case in the pipeline:

* :func:`load_graph_structure` – returns a raw edge-index and node count.
  Used by the partitioning step that only needs topology.
* :func:`load_dataset_pyg` – returns a full PyG ``Data`` object with
  features, labels, and train/val/test masks.  Used by the multi-epoch
  benchmark that trains a real GNN.

Both functions accept dataset short-names (e.g. ``"cora"``, ``"computers"``,
``"ogbn-arxiv"``) **or** a filesystem path to an ``.edgelist`` file.
"""

import os
import gc
from pathlib import Path

import numpy as np
import torch
import networkx as nx

from .config import STANDARD_IN_DIM, STANDARD_OUT_DIM

# ──────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ──────────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "cora":       {"pyg_name": "Cora",       "loader": "planetoid"},
    "citeseer":   {"pyg_name": "CiteSeer",   "loader": "planetoid"},
    "pubmed":     {"pyg_name": "PubMed",     "loader": "planetoid"},
    "flickr":     {"pyg_name": "Flickr",     "loader": "flickr"},
    "physics":    {"pyg_name": "Physics",    "loader": "coauthor"},
    "computers":  {"pyg_name": "Computers",  "loader": "amazon"},
    "ogbn-arxiv": {"pyg_name": "ogbn-arxiv", "loader": "ogb"},
    "reddit":     {"pyg_name": "Reddit",     "loader": "reddit"},
    "elliptic":   {"pyg_name": "Elliptic",   "loader": "elliptic"},
    "corafull":   {"pyg_name": "CoraFull",   "loader": "corafull"},
    "actor":      {"pyg_name": "Actor",      "loader": "actor"},
}

# Short-name alias map → canonical key in DATASET_REGISTRY
_ALIASES = {
    "coauthor-physics": "physics",
    "amazon-computers": "computers",
    "ogbn_arxiv":       "ogbn-arxiv",
}


def list_datasets():
    """Return the list of supported dataset short-names."""
    return list(DATASET_REGISTRY.keys())


def _resolve(name: str):
    """Resolve a user-supplied name to a registry entry (or None)."""
    key = name.lower().strip().replace("_", "-")
    if key in DATASET_REGISTRY:
        return DATASET_REGISTRY[key]
    if key in _ALIASES:
        return DATASET_REGISTRY[_ALIASES[key]]
    # Twitch family: twitch-de, twitch-en, …
    if key.startswith("twitch"):
        parts = key.split("-")
        region = parts[1].upper() if len(parts) > 1 else "DE"
        return {"pyg_name": region, "loader": "twitch"}
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Internal: PyG dataset construction
# ──────────────────────────────────────────────────────────────────────────────

def _pyg_dataset(info, data_dir, transform=None):
    """Instantiate a PyG dataset object from a registry entry."""
    loader = info["loader"]
    name = info["pyg_name"]

    if loader == "planetoid":
        from torch_geometric.datasets import Planetoid
        return Planetoid(root=data_dir, name=name, transform=transform)

    if loader == "flickr":
        from torch_geometric.datasets import Flickr
        return Flickr(root=os.path.join(data_dir, "Flickr"), transform=transform)

    if loader == "coauthor":
        from torch_geometric.datasets import Coauthor
        return Coauthor(root=os.path.join(data_dir, "Coauthor"), name=name, transform=transform)

    if loader == "amazon":
        from torch_geometric.datasets import Amazon
        return Amazon(root=os.path.join(data_dir, "Amazon"), name=name, transform=transform)

    if loader == "twitch":
        from torch_geometric.datasets import Twitch
        return Twitch(root=os.path.join(data_dir, "Twitch"), name=name, transform=transform)

    if loader == "reddit":
        from torch_geometric.datasets import Reddit
        return Reddit(root=os.path.join(data_dir, "Reddit"), transform=transform)

    if loader == "elliptic":
        from torch_geometric.datasets import EllipticBitcoinDataset
        return EllipticBitcoinDataset(root=os.path.join(data_dir, "Elliptic"), transform=transform)

    if loader == "corafull":
        from torch_geometric.datasets import CoraFull
        return CoraFull(root=os.path.join(data_dir, "CoraFull"), transform=transform)

    if loader == "actor":
        from torch_geometric.datasets import Actor
        return Actor(root=os.path.join(data_dir, "Actor"), transform=transform)

    if loader == "ogb":
        from ogb.nodeproppred import PygNodePropPredDataset

        # Patch for PyTorch ≥ 2.4 (weights_only default changed)
        _orig = torch.load
        def _patched(*a, **kw):
            kw["weights_only"] = False
            return _orig(*a, **kw)
        torch.load = _patched
        try:
            return PygNodePropPredDataset(
                name=name, root=os.path.join(data_dir, "OGB"), transform=transform
            )
        finally:
            torch.load = _orig

    raise ValueError(f"Unknown loader type: {loader}")


# ──────────────────────────────────────────────────────────────────────────────
# Public: load graph structure (for partitioning)
# ──────────────────────────────────────────────────────────────────────────────

def load_graph_structure(dataset_name: str, data_dir: str = "./data"):
    """
    Load *only* graph topology – used by the partitioning step.

    Parameters
    ----------
    dataset_name : str
        A registered short-name (``"cora"``, ``"ogbn-arxiv"``, …) **or**
        a path to an ``.edgelist`` file.
    data_dir : str
        Root directory where PyG datasets are cached.

    Returns
    -------
    edge_index_np : ndarray, shape (2, E)
        Directed edge list as int32.
    num_nodes : int
    """
    # File path?
    if os.path.exists(dataset_name):
        return _load_edgelist(dataset_name)

    info = _resolve(dataset_name)
    if info is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'.\n"
            f"Supported: {list_datasets()} or a path to an .edgelist file."
        )

    print(f"  Loading {info['pyg_name']} …")
    dataset = _pyg_dataset(info, data_dir)
    data = dataset[0]

    edge_index_np = data.edge_index.numpy().astype(np.int32)
    num_nodes = int(data.num_nodes)

    del data, dataset
    gc.collect()
    return edge_index_np, num_nodes


def _load_edgelist(path):
    """Load an edgelist file and return bidirectional edge_index + num_nodes."""
    G = nx.read_edgelist(str(path), nodetype=int)
    if G.number_of_nodes() == 0:
        raise ValueError(f"Empty edgelist: {path}")

    nodes = list(G.nodes())
    if min(nodes) != 0 or max(nodes) != len(nodes) - 1:
        mapping = {n: i for i, n in enumerate(sorted(nodes))}
        G = nx.relabel_nodes(G, mapping)

    num_nodes = G.number_of_nodes()
    edges = np.array(list(G.edges()), dtype=np.int32)
    edge_index_np = np.vstack([edges.T, edges[:, [1, 0]].T])

    del G, edges
    gc.collect()
    return edge_index_np, num_nodes


# ──────────────────────────────────────────────────────────────────────────────
# Public: load full PyG dataset (for multi-epoch benchmark)
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset_pyg(
    dataset_name: str,
    data_dir: str = "./data",
    device: str = "cpu",
    normalize_dims: bool = False,
    target_in_dim: int = STANDARD_IN_DIM,
    target_out_dim: int = STANDARD_OUT_DIM,
):
    """
    Load a complete PyG ``Data`` object ready for GNN training.

    Applies ``ToUndirected`` + ``AddSelfLoops``, generates train/val/test
    masks when they are missing, and optionally normalises feature and label
    dimensions so that timing comparisons across datasets are fair.

    Parameters
    ----------
    dataset_name : str
        Registered short-name.
    data_dir : str
        Root directory for cached datasets.
    device : str
        ``"cpu"`` or ``"cuda"``.
    normalize_dims : bool
        If *True*, features are linearly projected to *target_in_dim* and
        labels are remapped to *target_out_dim* classes.
    target_in_dim, target_out_dim : int
        Dimensions used when *normalize_dims* is True.

    Returns
    -------
    data : Data
    in_dim : int
    out_dim : int
    """
    import torch_geometric.transforms as T

    info = _resolve(dataset_name)
    if info is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported: {list_datasets()}"
        )

    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])
    dataset = _pyg_dataset(info, data_dir, transform=transform)
    data = dataset[0]

    # Ensure y is 1-D
    if len(data.y.shape) > 1:
        data.y = data.y.squeeze()

    # ── masks ──────────────────────────────────────────────────────────────
    _ensure_masks(data)

    # Flatten multi-split masks (e.g. Actor)
    for attr in ("train_mask", "val_mask", "test_mask"):
        m = getattr(data, attr, None)
        if m is not None and len(m.shape) > 1:
            setattr(data, attr, m[:, 0])

    # OGB-specific split
    if info["loader"] == "ogb":
        split_idx = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[split_idx["train"]] = True
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[split_idx["valid"]] = True
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[split_idx["test"]] = True
        data.y = data.y.squeeze()

    data = data.to(device)

    original_in_dim = dataset.num_features
    original_out_dim = dataset.num_classes
    actual_num_classes = int(data.y.max().item()) + 1
    if actual_num_classes != original_out_dim:
        original_out_dim = actual_num_classes

    print(f"  Nodes: {data.num_nodes:,}  |  Edges: {data.edge_index.shape[1]:,}")
    print(f"  Features: {original_in_dim}  |  Classes: {original_out_dim}")

    if normalize_dims:
        data = _normalize_features(data, target_in_dim, device)
        data = _normalize_labels(data, target_out_dim)
        return data, target_in_dim, target_out_dim

    return data, original_in_dim, original_out_dim


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_masks(data):
    """Generate random 60/20/20 masks if any split is missing."""
    import torch_geometric.transforms as T

    need_split = False
    for attr in ("train_mask", "val_mask", "test_mask"):
        if not hasattr(data, attr) or getattr(data, attr) is None:
            need_split = True
            break
    if need_split:
        splitter = T.RandomNodeSplit(split="train_rest", num_val=0.2, num_test=0.2)
        data = splitter(data)
    return data


def _normalize_features(data, target_dim, device):
    """Project features to *target_dim* via a fixed random projection."""
    orig = data.x.shape[1]
    if orig == target_dim:
        return data
    torch.manual_seed(42)
    proj = torch.randn(orig, target_dim, device=device) / (orig ** 0.5)
    data.x = data.x @ proj
    return data


def _normalize_labels(data, target_classes):
    """Remap labels to *target_classes* classes (deterministic modulo)."""
    if data.y.max().item() + 1 == target_classes:
        return data
    data.y = data.y % target_classes
    return data
