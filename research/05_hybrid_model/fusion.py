"""
Bidirectional Hierarchical Attention Fusion (BHAF) Layer.
Replaces degenerate single-token cross-attention with meaningful
bidirectional cross-attention using per-node spatial features.

Key fix: Uses per-node features [B, N_buses, 128] as multi-token K/V
sequence so that attention weights are non-trivial.

Architecture:
  Direction 1 (T→S): temporal query attends to individual bus nodes
  Direction 2 (S→T): each node queries the temporal representation
  Gated combination + multi-scale fusion from DenseNet taps

IT22577924 — Karunanayake K.P.A.W.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (CNN_T_DIM, RGNN_DIM, FUSED_DIM, FUSION_HEADS,
                    FUSION_DROPOUT, MULTISCALE_INTERMEDIATE)


class CrossAttentionBlock(nn.Module):
    """Single direction cross-attention with pre-norm, residual, and FFN."""

    def __init__(self, q_dim: int, kv_dim: int, d_attn: int,
                 n_heads: int, dropout: float):
        super().__init__()
        self.proj_q = nn.Linear(q_dim, d_attn)
        self.proj_k = nn.Linear(kv_dim, d_attn)
        self.proj_v = nn.Linear(kv_dim, d_attn)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_attn, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_attn)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_attn, d_attn * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_attn * 2, d_attn),
        )
        self.norm2 = nn.LayerNorm(d_attn)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        query     : [B, Lq, q_dim]
        key_value : [B, Lkv, kv_dim]

        Returns
        -------
        [B, Lq, d_attn]
        """
        Q = self.proj_q(query)
        K = self.proj_k(key_value)
        V = self.proj_v(key_value)

        attn_out, _ = self.attn(Q, K, V)
        h = self.norm1(Q + self.dropout1(attn_out))
        h = self.norm2(h + self.ffn(h))
        return h


class BidirectionalHierarchicalFusion(nn.Module):
    """
    BHAF: Bidirectional cross-attention with gated combination
    and multi-scale DenseNet feature fusion.

    Inputs:
      - h_temporal  : [B, CNN_T_DIM]        — from CNN-Transformer
      - node_feats  : [B, N_buses, RGNN_DIM] — from R-GNN per-node
      - f_early     : [B, F_early_dim]       — DenseNet Block 1 tap
      - f_mid       : [B, F_mid_dim]         — DenseNet Block 2 tap

    Outputs:
      - h_fused     : [B, FUSED_DIM]         — final fused representation
      - h_st        : [B, N_buses, FUSED_DIM] — per-node enriched features (for location)
    """

    def __init__(self, f_early_dim: int = 208, f_mid_dim: int = 392):
        super().__init__()
        d_attn = FUSED_DIM  # 192

        # Direction 1: Temporal → Spatial (T→S)
        # Q = temporal [B, 1, CNN_T_DIM], K/V = node_feats [B, N, RGNN_DIM]
        self.cross_ts = CrossAttentionBlock(
            q_dim=CNN_T_DIM, kv_dim=RGNN_DIM,
            d_attn=d_attn, n_heads=FUSION_HEADS, dropout=FUSION_DROPOUT)

        # Direction 2: Spatial → Temporal (S→T)
        # Q = node_feats [B, N, RGNN_DIM], K/V = temporal [B, 1, CNN_T_DIM]
        self.cross_st = CrossAttentionBlock(
            q_dim=RGNN_DIM, kv_dim=CNN_T_DIM,
            d_attn=d_attn, n_heads=FUSION_HEADS, dropout=FUSION_DROPOUT)

        # Expand single temporal token → 8 learned tokens for S→T K/V
        self.n_temporal_tokens = 8
        self.temporal_expand = nn.Sequential(
            nn.Linear(CNN_T_DIM, CNN_T_DIM * self.n_temporal_tokens),
            nn.GELU(),
        )

        # Gated combination
        self.gate_proj = nn.Linear(d_attn * 2, d_attn)

        # Multi-scale fusion: two-stage projection (792→492→192)
        ms_input_dim = f_early_dim + f_mid_dim + d_attn
        ms_inter = MULTISCALE_INTERMEDIATE  # 492
        self.multiscale_proj = nn.Sequential(
            nn.Linear(ms_input_dim, ms_inter),
            nn.GELU(),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(ms_inter, FUSED_DIM),
        )
        self.multiscale_ffn = nn.Sequential(
            nn.Linear(FUSED_DIM, FUSED_DIM * 2),
            nn.GELU(),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(FUSED_DIM * 2, FUSED_DIM),
        )
        self.multiscale_norm = nn.LayerNorm(FUSED_DIM)

    def forward(self, h_temporal: torch.Tensor, node_feats: torch.Tensor,
                f_early: torch.Tensor, f_mid: torch.Tensor) -> tuple:
        """
        Returns
        -------
        h_fused : [B, FUSED_DIM]             — graph-level fused feature
        h_st    : [B, N_buses, FUSED_DIM]    — per-node enriched features
        """
        B = h_temporal.shape[0]

        # Add sequence dimension for attention
        h_t_seq = h_temporal.unsqueeze(1)  # [B, 1, CNN_T_DIM]

        # Direction 1: T→S — temporal attends to spatial nodes
        h_ts = self.cross_ts(h_t_seq, node_feats)  # [B, 1, FUSED_DIM]
        h_ts = h_ts.squeeze(1)                       # [B, FUSED_DIM]

        # Direction 2: S→T — expand temporal to 8 tokens so attention is non-trivial
        h_t_expanded = self.temporal_expand(h_temporal)        # [B, CNN_T_DIM*8]
        h_t_expanded = h_t_expanded.view(B, self.n_temporal_tokens, -1)  # [B, 8, CNN_T_DIM]
        h_st = self.cross_st(node_feats, h_t_expanded)        # [B, N, FUSED_DIM]

        # Gated combination
        h_st_mean = h_st.mean(dim=1)                 # [B, FUSED_DIM]
        gate_input = torch.cat([h_ts, h_st_mean], dim=-1)  # [B, 2*FUSED_DIM]
        gate = torch.sigmoid(self.gate_proj(gate_input))    # [B, FUSED_DIM]
        h_gated = gate * h_ts + (1 - gate) * h_st_mean     # [B, FUSED_DIM]

        # Multi-scale fusion with DenseNet taps
        ms_input = torch.cat([f_early, f_mid, h_gated], dim=-1)
        h_ms = self.multiscale_proj(ms_input)         # [B, FUSED_DIM]
        h_fused = self.multiscale_norm(h_ms + self.multiscale_ffn(h_ms))

        return h_fused, h_st
