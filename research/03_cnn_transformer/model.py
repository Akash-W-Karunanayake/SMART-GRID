"""
CNN-Transformer Model (Temporal Branch) — v2 with DenseNet-121-1D backbone.

Architecture:
  [B, T, F] → 1D DenseNet-121 (growth=24) → multi-scale taps
  → Project → Transformer Encoder → GAP → h_temporal [B, D_MODEL]
  → 4 classification heads (multi-task)

Stochastic depth regularization applied to DenseBlock layers.

IT22577924 — Karunanayake K.P.A.W.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    DENSE_GROWTH_RATE, DENSE_BLOCK_LAYERS, DENSE_DROP_RATE,
    D_MODEL, N_HEAD, N_LAYERS, DIM_FF, TRANS_DROPOUT,
    N_CLASSES_DETECTION, N_CLASSES_TYPE, N_CLASSES_PHASE,
)


# ---------------------------------------------------------------------------
# DenseNet-1D building blocks
# ---------------------------------------------------------------------------

class _DenseLayer(nn.Module):
    """Single BN-ReLU-Conv1d layer with bottleneck and stochastic depth."""

    def __init__(self, in_channels: int, growth_rate: int, drop_rate: float = 0.0,
                 survival_prob: float = 1.0):
        super().__init__()
        mid = 4 * growth_rate
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, mid, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(mid)
        self.conv2 = nn.Conv1d(mid, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.survival_prob = survival_prob

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        # Stochastic depth — zero out contribution but keep channel count
        if self.training and self.survival_prob < 1.0:
            if torch.rand(1).item() > self.survival_prob:
                return torch.cat([x, torch.zeros_like(out)], dim=1)
            out = out / self.survival_prob  # scale to compensate for expected dropout
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, n_layers: int, in_channels: int, growth_rate: int,
                 drop_rate: float = 0.0, stoch_start: float = 1.0,
                 stoch_end: float = 1.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # Linear interpolation of survival probability
            surv = stoch_start + (stoch_end - stoch_start) * i / max(n_layers - 1, 1)
            self.layers.append(
                _DenseLayer(in_channels + i * growth_rate, growth_rate,
                           drop_rate, surv)
            )
        self.out_channels = in_channels + n_layers * growth_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    """BN-ReLU-Conv1x1-AvgPool transition between DenseBlocks."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_ch)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))


class DenseNet1D(nn.Module):
    """
    1D DenseNet-121 with reduced growth rate (24) for T=20 inputs.

    Produces multi-scale taps: F_early, F_mid, F_deep
    and final feature via GAP.
    """

    def __init__(self, in_features: int, growth_rate: int = DENSE_GROWTH_RATE,
                 block_layers: list = None, drop_rate: float = DENSE_DROP_RATE):
        super().__init__()
        if block_layers is None:
            block_layers = DENSE_BLOCK_LAYERS  # [6, 12, 24, 16]

        # Initial convolution
        init_ch = 64
        self.conv0 = nn.Conv1d(in_features, init_ch, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn0 = nn.BatchNorm1d(init_ch)

        # Stochastic depth: linearly increase drop prob from 0 to 0.2
        total_layers = sum(block_layers)
        layer_counter = 0

        # DenseBlock 1
        self.block1 = _DenseBlock(
            block_layers[0], init_ch, growth_rate, drop_rate,
            stoch_start=1.0,
            stoch_end=1.0 - 0.2 * block_layers[0] / total_layers)
        ch1 = self.block1.out_channels  # 64 + 6*24 = 208
        self.f_early_dim = ch1
        layer_counter += block_layers[0]

        # Transition 1 (halve channels, pool)
        self.trans1 = _Transition(ch1, ch1 // 2, pool=True)
        ch = ch1 // 2  # 104

        # DenseBlock 2
        s_start = 1.0 - 0.2 * layer_counter / total_layers
        self.block2 = _DenseBlock(
            block_layers[1], ch, growth_rate, drop_rate,
            stoch_start=s_start,
            stoch_end=1.0 - 0.2 * (layer_counter + block_layers[1]) / total_layers)
        ch2 = self.block2.out_channels  # 104 + 12*24 = 392
        self.f_mid_dim = ch2
        layer_counter += block_layers[1]

        # Transition 2 (halve channels, pool)
        self.trans2 = _Transition(ch2, ch2 // 2, pool=True)
        ch = ch2 // 2  # 196

        # DenseBlock 3
        s_start = 1.0 - 0.2 * layer_counter / total_layers
        self.block3 = _DenseBlock(
            block_layers[2], ch, growth_rate, drop_rate,
            stoch_start=s_start,
            stoch_end=1.0 - 0.2 * (layer_counter + block_layers[2]) / total_layers)
        ch3 = self.block3.out_channels  # 196 + 24*24 = 772
        layer_counter += block_layers[2]

        # Transition 3 (halve channels, NO pool — T=5 is minimum)
        self.trans3 = _Transition(ch3, ch3 // 2, pool=False)
        ch = ch3 // 2  # 386

        # DenseBlock 4
        s_start = 1.0 - 0.2 * layer_counter / total_layers
        self.block4 = _DenseBlock(
            block_layers[3], ch, growth_rate, drop_rate,
            stoch_start=s_start, stoch_end=0.8)
        ch4 = self.block4.out_channels  # 386 + 16*24 = 770
        self.f_deep_dim = ch4

        self.final_bn = nn.BatchNorm1d(ch4)

    def forward(self, x):
        """
        Parameters
        ----------
        x : [B, C_in, T]

        Returns
        -------
        f_early       : [B, f_early_dim]       — pooled (for hybrid fusion)
        f_mid         : [B, f_mid_dim]          — pooled (for hybrid fusion)
        f_deep        : [B, f_deep_dim, T']     — full temporal map (for Transformer)
        f_deep_pooled : [B, f_deep_dim]         — pooled (for hybrid fusion)
        """
        x = F.relu(self.bn0(self.conv0(x)))

        # Block 1 → early tap (pooled)
        x = self.block1(x)
        f_early = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        x = self.trans1(x)

        # Block 2 → mid tap (pooled)
        x = self.block2(x)
        f_mid = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        x = self.trans2(x)

        # Block 3
        x = self.block3(x)
        x = self.trans3(x)

        # Block 4 → deep tap (keep full temporal map)
        x = self.block4(x)
        x = F.relu(self.final_bn(x))
        f_deep = x                                          # [B, ch4, T']
        f_deep_pooled = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # [B, ch4]

        return f_early, f_mid, f_deep, f_deep_pooled


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Full CNN-Transformer Model
# ---------------------------------------------------------------------------

class CNNTransformerModel(nn.Module):
    """
    DenseNet-121-1D + Transformer encoder for multi-task fault diagnostics.

    Exposes multi-scale features (f_early, f_mid, f_deep) for hybrid fusion,
    plus the final temporal representation h_temporal.

    Parameters
    ----------
    in_features    : number of input features per timestep (N_branches × 6)
    n_buses        : number of buses (for location head output size)
    """

    def __init__(self, in_features: int, n_buses: int):
        super().__init__()

        self.densenet = DenseNet1D(in_features)

        # Project deep features to d_model for Transformer
        self.proj_deep = nn.Linear(self.densenet.f_deep_dim, D_MODEL)
        self.pos_enc = PositionalEncoding(D_MODEL, dropout=TRANS_DROPOUT)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=DIM_FF,
            dropout=TRANS_DROPOUT, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

        # Output heads
        self.head_detection = nn.Sequential(
            nn.Linear(D_MODEL, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, N_CLASSES_DETECTION)
        )
        self.head_type = nn.Sequential(
            nn.Linear(D_MODEL, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, N_CLASSES_TYPE)
        )
        self.head_phase = nn.Sequential(
            nn.Linear(D_MODEL, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, N_CLASSES_PHASE)
        )
        self.head_location = nn.Sequential(
            nn.Linear(D_MODEL, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, n_buses + 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        Parameters
        ----------
        x : [B, T, F]  — normalized current sequences (6 features per branch)

        Returns
        -------
        dict with logits: detection, type, phase, location
             and features: feat, f_early, f_mid, f_deep
        """
        # DenseNet expects [B, C, T]
        x_perm = x.permute(0, 2, 1)  # [B, F, T]
        f_early, f_mid, f_deep, f_deep_pooled = self.densenet(x_perm)

        # Transformer on deep temporal features
        # f_deep: [B, ch4, T'] — full temporal map
        f_deep_seq = f_deep.permute(0, 2, 1)    # [B, T', ch4]
        h = self.proj_deep(f_deep_seq)           # [B, T', D_MODEL]
        h = self.pos_enc(h)
        h = self.transformer(h)                  # [B, T', D_MODEL]
        feat = h.mean(dim=1)                     # [B, D_MODEL] — GAP after Transformer

        return {
            "detection": self.head_detection(feat),
            "type":      self.head_type(feat),
            "phase":     self.head_phase(feat),
            "location":  self.head_location(feat),
            "feat":      feat,              # h_temporal [B, D_MODEL] for hybrid fusion
            "f_early":   f_early,           # [B, 208] multi-scale tap (pooled)
            "f_mid":     f_mid,             # [B, 392] multi-scale tap (pooled)
            "f_deep":    f_deep_pooled,     # [B, 770] multi-scale tap (pooled)
        }
