"""
Graph-Informed Attention Fusion Layer.
Combines temporal (CNN-Transformer) and spatial (R-GNN) features via cross-attention.
IT22577924 — Karunanayake K.P.A.W.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CNN_T_DIM, RGNN_DIM, FUSED_DIM, FUSION_HEADS, FUSION_DROPOUT


class GraphInformedFusion(nn.Module):
    """
    Cross-attention fusion of temporal and spatial branch features.

    Temporal feature acts as Query.
    Spatial (graph) feature acts as Key and Value.

    Output: fused vector [B, FUSED_DIM]
    """

    def __init__(self):
        super().__init__()
        # Project both branches to the same dimension for cross-attention
        self.proj_temporal = nn.Linear(CNN_T_DIM, FUSED_DIM)
        self.proj_spatial  = nn.Linear(RGNN_DIM,  FUSED_DIM)

        self.attn = nn.MultiheadAttention(
            embed_dim=FUSED_DIM,
            num_heads=FUSION_HEADS,
            dropout=FUSION_DROPOUT,
            batch_first=True,
        )

        self.norm  = nn.LayerNorm(FUSED_DIM)
        self.dropout = nn.Dropout(FUSION_DROPOUT)

        # Final fusion MLP
        self.ffn = nn.Sequential(
            nn.Linear(FUSED_DIM, FUSED_DIM),
            nn.ReLU(),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(FUSED_DIM, FUSED_DIM),
        )
        self.norm2 = nn.LayerNorm(FUSED_DIM)

    def forward(self, temporal_feat: torch.Tensor,
                spatial_feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        temporal_feat : [B, CNN_T_DIM]  — from CNN-Transformer
        spatial_feat  : [B, RGNN_DIM]   — from R-GNN global pool

        Returns
        -------
        fused : [B, FUSED_DIM]
        """
        # Project and add sequence dim for MHA: [B, 1, FUSED_DIM]
        Q = self.proj_temporal(temporal_feat).unsqueeze(1)
        K = self.proj_spatial(spatial_feat).unsqueeze(1)
        V = K

        # Cross-attention
        attn_out, _ = self.attn(Q, K, V)   # [B, 1, FUSED_DIM]
        attn_out = attn_out.squeeze(1)       # [B, FUSED_DIM]

        # Residual + LayerNorm (on projected temporal)
        h = self.norm(Q.squeeze(1) + self.dropout(attn_out))

        # FFN
        h = self.norm2(h + self.ffn(h))
        return h
