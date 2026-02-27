"""
CNN-Transformer Model (Temporal Branch).
Architecture:
  [B, T, F] → Conv1D stack → Transformer Encoder → Global AvgPool → [B, D_MODEL]
  → 4 classification heads (multi-task)
IT22577924 — Karunanayake K.P.A.W.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    CNN_CHANNELS, CNN_KERNEL, CNN_DROPOUT,
    D_MODEL, N_HEAD, N_LAYERS, DIM_FF, TRANS_DROPOUT,
    N_CLASSES_DETECTION, N_CLASSES_TYPE, N_CLASSES_PHASE,
)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CNNEncoder(nn.Module):
    """1D-CNN feature extractor over the time dimension."""

    def __init__(self, in_features: int, channels: list = CNN_CHANNELS,
                 kernel: int = CNN_KERNEL, dropout: float = CNN_DROPOUT):
        super().__init__()
        layers = []
        in_ch = in_features
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
                nn.ReLU(),
                nn.BatchNorm1d(out_ch),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x):
        # x: [B, T, F] → permute → [B, F, T]
        x = x.permute(0, 2, 1)
        x = self.net(x)
        # → [B, out_channels, T'] → permute → [B, T', out_channels]
        return x.permute(0, 2, 1)


class CNNTransformerModel(nn.Module):
    """
    Full CNN-Transformer model for multi-task fault diagnostics.

    Parameters
    ----------
    in_features    : number of input features per timestep (N_branches × 3)
    n_buses        : number of buses (for location head output size)
    """

    def __init__(self, in_features: int, n_buses: int):
        super().__init__()
        self.cnn_encoder = CNNEncoder(in_features)
        d = self.cnn_encoder.out_channels  # == D_MODEL after last conv layer

        # Project to d_model if CNN output differs
        self.proj = nn.Linear(d, D_MODEL) if d != D_MODEL else nn.Identity()
        self.pos_enc = PositionalEncoding(D_MODEL, dropout=TRANS_DROPOUT)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=DIM_FF,
            dropout=TRANS_DROPOUT, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

        # Output heads
        self.head_detection = nn.Linear(D_MODEL, N_CLASSES_DETECTION)
        self.head_type      = nn.Linear(D_MODEL, N_CLASSES_TYPE)
        self.head_phase     = nn.Linear(D_MODEL, N_CLASSES_PHASE)
        self.head_location  = nn.Linear(D_MODEL, n_buses + 1)   # +1 for "no fault" class

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : [B, T, F]  — normalized current sequences

        Returns
        -------
        dict with logits: detection, type, phase, location
        """
        h = self.cnn_encoder(x)      # [B, T', d]
        h = self.proj(h)             # [B, T', D_MODEL]
        h = self.pos_enc(h)
        h = self.transformer(h)      # [B, T', D_MODEL]
        feat = h.mean(dim=1)         # global average pool → [B, D_MODEL]

        return {
            "detection": self.head_detection(feat),
            "type":      self.head_type(feat),
            "phase":     self.head_phase(feat),
            "location":  self.head_location(feat),
            "feat":      feat,       # expose for hybrid model
        }
