"""
GNN State Encoder
=================
2-layer GCN that processes the grid graph and produces per-node embeddings.
These give each MARL agent a topology-aware view of the grid state.

Outputs:
  - Per-node embeddings:  (N_nodes, embed_dim)
  - Global graph embedding (mean-pooled): (embed_dim,)
  - Agent-local embeddings for tie-switch buses: (2 * embed_dim,)

CASING: All bus names must be lowercase (matching engine convention).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class GNNEncoder(nn.Module):
    """2-layer GCN: node features -> hidden -> embeddings."""

    def __init__(self, in_features: int, hidden_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric required. pip install torch-geometric")
        self.embed_dim = embed_dim
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)

    def forward(self, data: "Data") -> torch.Tensor:
        """Return per-node embeddings (N_nodes, embed_dim)."""
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # (N_nodes, embed_dim)

    def get_global_embedding(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Mean-pool all node embeddings into a single graph-level vector."""
        return node_embeddings.mean(dim=0)  # (embed_dim,)

    def get_agent_local_embedding(
        self,
        node_embeddings: torch.Tensor,
        bus_to_idx: Dict[str, int],
        bus_from: str,
        bus_to: str,
    ) -> torch.Tensor:
        """
        Get concatenated embeddings for the two buses a tie switch connects.
        Returns (2 * embed_dim,) vector.
        Falls back to zeros if a bus isn't in the graph.
        """
        device = node_embeddings.device
        embed_dim = node_embeddings.shape[1]

        if bus_from in bus_to_idx:
            emb_from = node_embeddings[bus_to_idx[bus_from]]
        else:
            emb_from = torch.zeros(embed_dim, device=device)

        if bus_to in bus_to_idx:
            emb_to = node_embeddings[bus_to_idx[bus_to]]
        else:
            emb_to = torch.zeros(embed_dim, device=device)

        return torch.cat([emb_from, emb_to])  # (2 * embed_dim,)


def grid_state_to_pyg_data(
    bus_voltages: Dict[str, float],
    adjacency_edges: List[Tuple[str, str]],
    switch_states: Optional[Dict[str, bool]] = None,
    load_served: Optional[Dict[str, bool]] = None,
) -> Optional[Tuple["Data", Dict[str, int]]]:
    """
    Convert grid state to PyG Data object and bus-to-index mapping.

    Node features per bus (in_features depends on available data):
      - voltage magnitude (1)

    Returns (Data, bus_to_idx) or (None, None) if torch_geometric unavailable.

    IMPORTANT: bus_voltages keys and adjacency_edges buses must ALL
    be lowercase. The engine and grid_graph modules guarantee this.
    """
    if not HAS_PYG:
        return None, None

    # Build bus index -- keys are already lowercase from engine
    bus_names = sorted(bus_voltages.keys())
    bus_to_idx = {b: i for i, b in enumerate(bus_names)}
    N = len(bus_names)

    # Node features: voltage magnitude
    x = torch.zeros(N, 1, dtype=torch.float32)
    for b, v in bus_voltages.items():
        x[bus_to_idx[b], 0] = v

    # Edge index -- edges are already lowercase from grid_graph
    src, dst = [], []
    for b1, b2 in adjacency_edges:
        if b1 in bus_to_idx and b2 in bus_to_idx:
            i1, i2 = bus_to_idx[b1], bus_to_idx[b2]
            src.extend([i1, i2])   # undirected
            dst.extend([i2, i1])

    if not src:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data, bus_to_idx
