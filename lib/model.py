"""
Adaptive GCN with switchable aggregation strategy.

The model implements a standard two-layer GCN whose message-passing step
can use one of three aggregation back-ends at forward time:

* **edge**   – scatter-based edge aggregation  (``index_add_``)
* **sparse** – sparse matrix multiplication     (CSR ``matmul``)
* **dense**  – dense matrix multiplication      (``torch.matmul``)

The rest of the architecture (linear projections, dropout, …) is identical
regardless of the strategy, so timing differences isolate the aggregation cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGCNLayer(nn.Module):
    """Single GCN layer with selectable aggregation."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x, structure, deg_inv_sqrt, strategy="sparse"):
        # Symmetric normalisation: D^{-1/2} · x
        x_norm = x * deg_inv_sqrt

        if strategy == "edge":
            src, dst = structure
            x_agg = torch.zeros_like(x_norm)
            x_agg.index_add_(0, dst, x_norm[src])

        elif strategy == "sparse":
            if hasattr(structure, "matmul"):
                x_agg = structure.matmul(x_norm)
            elif structure.layout == torch.sparse_csr:
                x_agg = torch.sparse.mm(structure.to_sparse_coo(), x_norm)
            else:
                x_agg = structure @ x_norm

        elif strategy == "dense":
            x_agg = torch.matmul(structure, x_norm)

        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: edge, sparse, dense."
            )

        # Second half of symmetric normalisation
        x_agg = x_agg * deg_inv_sqrt
        return self.linear(x_agg)


class AdaptiveGCN(nn.Module):
    """Two-layer GCN whose aggregation strategy is selected at forward time."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.layer1 = AdaptiveGCNLayer(in_dim, hidden_dim)
        self.layer2 = AdaptiveGCNLayer(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, structure, deg_inv_sqrt, strategy="sparse"):
        x = self.layer1(x, structure, deg_inv_sqrt, strategy)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x, structure, deg_inv_sqrt, strategy)
        return x
