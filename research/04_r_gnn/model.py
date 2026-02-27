"""
Recurrent Graph Neural Network (R-GNN) — Spatial Branch.
Architecture:
  For each timestep t: GCNConv × 2 → GRUCell (per-node recurrence)
  Final hidden: global mean pool → graph-level feature [B, GRU_HIDDEN]
  Also exposes per-node features for hybrid fusion.

IT22577924 — Karunanayake K.P.A.W.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import add_self_loops, degree
    PYGEOMETRIC = True
except ImportError:
    PYGEOMETRIC = False

from config import (
    NODE_FEAT, EDGE_FEAT, GCN_HIDDEN, GCN_LAYERS, GCN_DROPOUT,
    GRU_HIDDEN, T_STEPS,
    N_CLASSES_DETECTION, N_CLASSES_TYPE, N_CLASSES_PHASE,
)


class FallbackGCNConv(nn.Module):
    """Minimal GCN layer when torch_geometric is unavailable."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch)

    def forward(self, x, edge_index, edge_attr=None):
        return F.relu(self.lin(x))


def _gcn_conv(in_ch, out_ch):
    if PYGEOMETRIC:
        return GCNConv(in_ch, out_ch)
    return FallbackGCNConv(in_ch, out_ch)


class RGNNModel(nn.Module):
    """
    Recurrent GNN for spatio-temporal fault diagnostics.

    Parameters
    ----------
    n_buses : int  — number of graph nodes
    """

    def __init__(self, n_buses: int):
        super().__init__()
        self.n_buses = n_buses

        # GCN message-passing layers (shared across timesteps)
        self.gcn_layers = nn.ModuleList()
        in_ch = NODE_FEAT
        for _ in range(GCN_LAYERS):
            self.gcn_layers.append(_gcn_conv(in_ch, GCN_HIDDEN))
            in_ch = GCN_HIDDEN

        self.gcn_dropout = nn.Dropout(GCN_DROPOUT)

        # GRU cell: per-node recurrence over time
        self.gru = nn.GRUCell(input_size=GCN_HIDDEN, hidden_size=GRU_HIDDEN)

        # Graph-level output heads (after global pooling)
        self.head_detection = nn.Linear(GRU_HIDDEN, N_CLASSES_DETECTION)
        self.head_type      = nn.Linear(GRU_HIDDEN, N_CLASSES_TYPE)
        self.head_phase     = nn.Linear(GRU_HIDDEN, N_CLASSES_PHASE)

        # Node-level location head: each node predicts a scalar score for
        # being the fault location; argmax over nodes = predicted fault bus.
        self.head_location  = nn.Linear(GRU_HIDDEN, 1)

    def forward(self, x_seq, edge_index, edge_attr=None, batch=None):
        """
        Parameters
        ----------
        x_seq      : [T, N_total_nodes, NODE_FEAT]  — stacked node features across time
                     OR for single-graph: [B*T, N_buses, NODE_FEAT]
        edge_index : [2, E] — graph connectivity (same for all t in a sample)
        edge_attr  : [E, EDGE_FEAT] — edge features
        batch      : [N_total_nodes] — batch assignment for PyG mini-batching

        For simplicity in training loop, we accept [B, T, N, F] and process each sample.
        Returns
        -------
        dict with logits: detection, type, phase, location, node_feats, graph_feat
        """
        # x_seq: [B, T, N, Fin]
        B, T, N, Fin = x_seq.shape
        device = x_seq.device

        # Initialize GRU hidden state per node: [B*N, GRU_HIDDEN]
        h = torch.zeros(B * N, GRU_HIDDEN, device=device)

        for t in range(T):
            # x_t: [B, N, Fin] → reshape to [B*N, Fin]
            x_t = x_seq[:, t, :, :]   # [B, N, Fin]

            # Process each sample in batch through GCN (edge_index is per-sample)
            # For batched GCN we replicate edge_index across batch
            x_t_flat = x_t.reshape(B * N, Fin)

            # Offset edge indices for batch
            # edge_index: [2, E] — apply to all batch items with offset
            E = edge_index.shape[1]
            offsets = torch.arange(B, device=device).repeat_interleave(E) * N
            src_batch = edge_index[0].repeat(B) + offsets
            dst_batch = edge_index[1].repeat(B) + offsets
            ei_batch  = torch.stack([src_batch, dst_batch], dim=0)

            # GCN forward
            z = x_t_flat
            for gcn in self.gcn_layers:
                z = gcn(z, ei_batch)
                z = F.relu(z)
                z = self.gcn_dropout(z)

            # GRU update
            h = self.gru(z, h)   # [B*N, GRU_HIDDEN]

        # Per-node final hidden state
        node_feats = h.reshape(B, N, GRU_HIDDEN)   # [B, N, GRU_HIDDEN]

        # Graph-level feature via mean pooling
        graph_feat = node_feats.mean(dim=1)          # [B, GRU_HIDDEN]

        # Per-node location score → [B, N]: each node votes for itself as
        # the fault location; CrossEntropyLoss treats node index as class.
        loc_logits = self.head_location(node_feats).squeeze(-1)  # [B, N]

        return {
            "detection":  self.head_detection(graph_feat),
            "type":       self.head_type(graph_feat),
            "phase":      self.head_phase(graph_feat),
            "location":   loc_logits,
            "graph_feat": graph_feat,      # [B, GRU_HIDDEN]
            "node_feats": node_feats,      # [B, N, GRU_HIDDEN]
        }
