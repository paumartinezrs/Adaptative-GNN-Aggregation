"""
Partition data-loader for the multi-epoch benchmark.

Reads the ``.edgelist`` + ``.mapping`` pairs produced by the partitioning step
and maps local node IDs back to the global feature / label tensors of the
full dataset so that each sub-graph can be trained independently.
"""

import torch
import networkx as nx
from pathlib import Path
from torch_geometric.utils import degree


class PartitionDataLoader:
    """
    Load every ``part_*.edgelist`` in *partitions_dir* and prepare
    the tensors needed for per-partition GNN training.

    Parameters
    ----------
    partitions_dir : str
        Directory containing ``part_0.edgelist``, ``part_0.mapping``, …
    global_x : Tensor (N, F)
        Global node features.
    global_y : Tensor (N,)
        Global node labels.
    global_mask : Tensor (N,)  bool
        Global train mask.
    device : torch.device
    global_num_nodes : int
    """

    def __init__(self, partitions_dir, global_x, global_y, global_mask,
                 device, global_num_nodes):
        self.partitions = []
        self.device = device
        self.global_x = global_x
        self.global_y = global_y
        self.global_mask = global_mask
        self.global_num_nodes = global_num_nodes

        path = Path(partitions_dir)
        if not path.exists():
            raise FileNotFoundError(f"Partition directory not found: {path}")

        files = sorted(
            path.glob("part_*.edgelist"),
            key=lambda x: int(x.stem.split("_")[1]),
        )
        print(f"  Processing {len(files)} partitions …")

        for idx, f in enumerate(files):
            try:
                data = self._load(f, idx)
                if data is not None:
                    self.partitions.append(data)
            except Exception as e:
                print(f"  Error in partition {idx}: {e}")

        print(f"  {len(self.partitions)} partitions loaded")

    # ──────────────────────────────────────────────────────────────────────
    # Internal loader
    # ──────────────────────────────────────────────────────────────────────

    def _load(self, p_file, idx):
        G = nx.read_edgelist(str(p_file), nodetype=int)
        if G.number_of_nodes() == 0:
            return None

        local_nodes = sorted(G.nodes())
        num_nodes = len(local_nodes)

        # ── mapping file ──
        map_file = p_file.with_suffix(".mapping")
        if map_file.exists():
            with open(map_file) as f:
                mapping_list = [int(line.strip()) for line in f]
            if len(mapping_list) < num_nodes:
                print(f"  Partition {idx}: mapping too short, skipping")
                return None
            mapping_list = mapping_list[:num_nodes]
            global_indices = mapping_list
        else:
            global_indices = local_nodes

        # Validate indices
        bad = [g for g in global_indices if g < 0 or g >= self.global_num_nodes]
        if bad:
            print(f"  Partition {idx}: {len(bad)} out-of-range indices, skipping")
            return None

        # Re-index edges to contiguous [0, num_nodes)
        node_map = {orig: new for new, orig in enumerate(local_nodes)}
        edges = []
        for u, v in G.edges():
            nu, nv = node_map[u], node_map[v]
            edges.append([nu, nv])
            edges.append([nv, nu])
        # Self-loops
        for i in range(num_nodes):
            edges.append([i, i])
        if not edges:
            return None

        edge_index = torch.tensor(edges, dtype=torch.long).t()

        # Symmetric normalisation: D^{-1/2}
        deg = degree(edge_index[1], num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        deg_inv_sqrt = deg_inv_sqrt.view(-1, 1)

        # Sparse CSR adjacency
        val = torch.ones(edge_index.shape[1], dtype=torch.float)
        adj_sparse = (
            torch.sparse_coo_tensor(edge_index, val, (num_nodes, num_nodes))
            .coalesce()
            .to_sparse_csr()
        )

        global_idx = torch.tensor(global_indices, dtype=torch.long).to(self.device)

        return {
            "global_idx": global_idx,
            "edge_index": edge_index.to(self.device),
            "adj_sparse": adj_sparse.to(self.device),
            "deg_inv_sqrt": deg_inv_sqrt.to(self.device),
            "num_nodes": num_nodes,
            "partition_id": idx,
            "partition_name": p_file.name,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Batch access
    # ──────────────────────────────────────────────────────────────────────

    def get_batch(self, idx, strategy):
        """
        Return ``(x, y, mask, structure, deg_inv_sqrt)`` for partition *idx*.

        *structure* depends on *strategy*:

        * ``"edge"``   → ``edge_index`` (2, E)
        * ``"sparse"`` → CSR adjacency matrix
        * ``"dense"``  → dense adjacency matrix
        """
        p = self.partitions[idx]

        x = self.global_x[p["global_idx"]]
        y = self.global_y[p["global_idx"]]
        mask = self.global_mask[p["global_idx"]]

        if strategy == "edge":
            structure = p["edge_index"]
        elif strategy == "sparse":
            structure = p["adj_sparse"]
        elif strategy == "dense":
            structure = p["adj_sparse"].to_dense()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return x, y, mask, structure, p["deg_inv_sqrt"]
